"""Probe 1 — would giving the document title weight in retrieval help?

Today the title never reaches ranking. What gets embedded is
``breadcrumbs + "\\n\\n" + body`` (ingest_worker.py:201), the Qdrant payload has
no title field, and the reranker sees only ``child.text``
(search_service.py:161). The title is fetched at search_service.py:186 — after
ranking is finished — purely to fill ``source_title`` for display. So a chunk
reading "Biaya UKT 500000" ranks on its own words even when it sits inside a
document about outbound student mobility.

Four rankings over the *same* retrieved candidates, so no re-ingestion:

    A  baseline     rerank on child.text                (what production does)
    B  rerank-input rerank on "title\\n" + child.text     (smallest real diff)
    C  score blend  (1-w)*norm(rerank_A) + w*cos(q,title)
    D  title-only   cos(q, title) alone                 (diagnostic)

    python -m evals.probes.probe1_title [--limit N]

Writes results/probe1_title.csv and results/probe1_title.summary.json.
Read-only. Everything expensive is cached under cache/.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from ._common import (
    quiet_logs,
    CACHE_DIR,
    Candidate,
    Query,
    Retriever,
    cached_dense,
    cosine,
    doc_ranking,
    get_reranker,
    hit_at_k,
    load_child_texts,
    load_documents,
    load_student_questions,
    load_subset_a_queries,
    minmax,
    reciprocal_rank,
    save_dense,
    write_results,
)

BLEND_WEIGHTS = (0.1, 0.2, 0.3)
# Below this cosine, the query and the document's title are talking about
# different things. Calibrated from the observed distribution, reported in full
# so the cut can be re-judged.
TITLE_MISMATCH_MAX_COS = 0.45


class RerankCache:
    """Rerank scores keyed by (variant, query). One HTTP call per query."""

    def __init__(self, name: str) -> None:
        self.path = CACHE_DIR / f"{name}.jsonl"
        self.data: Dict[str, List[float]] = {}
        if self.path.exists():
            with open(self.path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        rec = json.loads(line)
                        self.data[rec["k"]] = rec["s"]
            print(f"  rerank cache: {len(self.data)} entries")

    def get(self, key: str) -> List[float] | None:
        return self.data.get(key)

    def put(self, key: str, scores: List[float]) -> None:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.data[key] = scores
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"k": key, "s": scores}) + "\n")


async def rerank_scores(
    reranker, cache: RerankCache, variant: str, query: Query, docs: Sequence[str]
) -> List[float]:
    """Cross-encoder score per candidate, in candidate order."""
    key = f"{variant}::{query.qid}"
    hit = cache.get(key)
    if hit is not None and len(hit) == len(docs):
        return hit
    results = await reranker.rerank(query=query.text, documents=list(docs))
    scores = [0.0] * len(docs)
    for r in results:
        if 0 <= r.index < len(scores):
            scores[r.index] = float(r.score)
    cache.put(key, scores)
    return scores


def rank_by(cands: Sequence[Candidate], scores: Sequence[float]) -> List[Candidate]:
    order = sorted(range(len(cands)), key=lambda i: scores[i], reverse=True)
    return [cands[i] for i in order]


def score_set(queries: Sequence[Query], rankings: Dict[str, Dict[str, List[Candidate]]]) -> Dict[str, dict]:
    """Document-level Hit@k and MRR per variant, over queries that have gold."""
    labelled = [q for q in queries if q.gold_doc_ids]
    out: Dict[str, dict] = {}
    for variant, per_query in rankings.items():
        if not labelled:
            out[variant] = {}
            continue
        hits = {k: [] for k in (1, 3, 5)}
        rrs = []
        for q in labelled:
            docs = doc_ranking(per_query[q.qid])
            for k in hits:
                hits[k].append(hit_at_k(docs, q.gold_doc_ids, k))
            rrs.append(reciprocal_rank(docs, q.gold_doc_ids))
        out[variant] = {
            "n": len(labelled),
            "hit@1": round(sum(hits[1]) / len(labelled), 4),
            "hit@3": round(sum(hits[3]) / len(labelled), 4),
            "hit@5": round(sum(hits[5]) / len(labelled), 4),
            "mrr": round(sum(rrs) / len(labelled), 4),
        }
    return out


async def build_rankings(
    queries: Sequence[Query],
    cands_by_qid: Dict[str, List[Candidate]],
    child_text: Dict[str, str],
    titles: Dict[str, str],
    q_vec: Dict[str, np.ndarray],
    title_vec: Dict[str, np.ndarray],
    reranker,
    cache: RerankCache,
) -> Tuple[Dict[str, Dict[str, List[Candidate]]], Dict[str, dict]]:
    """All four variants plus the per-query diagnostic row."""
    rankings: Dict[str, Dict[str, List[Candidate]]] = {
        "A_baseline": {}, "B_rerank_title": {}, "D_title_only": {},
        **{f"C_blend_w{w}": {} for w in BLEND_WEIGHTS},
    }
    diagnostics: Dict[str, dict] = {}

    for n, q in enumerate(queries, 1):
        cands = cands_by_qid.get(q.qid, [])
        if not cands:
            for v in rankings:
                rankings[v][q.qid] = []
            continue

        plain = [child_text.get(c.chunk_id, "") for c in cands]
        titled = [
            f"{titles.get(c.doc_id, '')}\n{child_text.get(c.chunk_id, '')}"
            for c in cands
        ]

        s_a = await rerank_scores(reranker, cache, "A", q, plain)
        s_b = await rerank_scores(reranker, cache, "B", q, titled)

        qv = q_vec[q.qid]
        s_title = [
            float(qv @ title_vec[c.doc_id]) if c.doc_id in title_vec else 0.0
            for c in cands
        ]

        rankings["A_baseline"][q.qid] = rank_by(cands, s_a)
        rankings["B_rerank_title"][q.qid] = rank_by(cands, s_b)
        rankings["D_title_only"][q.qid] = rank_by(cands, s_title)

        norm_a = minmax(s_a)
        for w in BLEND_WEIGHTS:
            blended = [(1 - w) * norm_a[i] + w * s_title[i] for i in range(len(cands))]
            rankings[f"C_blend_w{w}"][q.qid] = rank_by(cands, blended)

        top_a = rankings["A_baseline"][q.qid][0]
        top_b = rankings["B_rerank_title"][q.qid][0]
        diagnostics[q.qid] = {
            "top1_doc_A": top_a.doc_id,
            "top1_title_A": titles.get(top_a.doc_id, ""),
            "top1_title_cos_A": round(s_title[cands.index(top_a)], 4),
            "top1_doc_B": top_b.doc_id,
            "top1_changed_B": int(top_a.doc_id != top_b.doc_id),
            "max_title_cos": round(max(s_title), 4) if s_title else 0.0,
        }

        if n % 25 == 0:
            print(f"    scored {n}/{len(queries)}")

    return rankings, diagnostics


async def run_set(
    label: str,
    queries: Sequence[Query],
    retriever: Retriever,
    docs_meta,
    reranker,
    cache: RerankCache,
) -> Tuple[dict, List[dict]]:
    print(f"\n  [{label}] {len(queries)} queries")

    cands_by_qid = await retriever.retrieve(queries)
    chunk_ids = sorted({c.chunk_id for cs in cands_by_qid.values() for c in cs})
    child_text = await load_child_texts(chunk_ids)
    print(f"    {len(chunk_ids)} distinct candidate chunks")

    titles = {d.doc_id: d.title for d in docs_meta.values()}

    # Title vectors: computed once for the whole corpus, cached across probes.
    cached = cached_dense("title_emb")
    if cached is None:
        print(f"    embedding {len(titles)} document titles ...")
        ids = list(titles.keys())
        vecs = await retriever.embed([titles[i] for i in ids])
        save_dense("title_emb", ids, vecs)
        cached = (ids, np.asarray(vecs, dtype="float32"))
    t_ids, t_mat = cached
    t_mat = t_mat / (np.linalg.norm(t_mat, axis=1, keepdims=True) + 1e-12)
    title_vec = {i: t_mat[n] for n, i in enumerate(t_ids)}

    # Query vectors for the blend.
    qcache = cached_dense(f"query_emb_{label}")
    if qcache is None or set(qcache[0]) != {q.qid for q in queries}:
        print(f"    embedding {len(queries)} queries ...")
        vecs = await retriever.embed([q.text for q in queries])
        save_dense(f"query_emb_{label}", [q.qid for q in queries], vecs)
        qcache = ([q.qid for q in queries], np.asarray(vecs, dtype="float32"))
    q_ids, q_mat = qcache
    q_mat = q_mat / (np.linalg.norm(q_mat, axis=1, keepdims=True) + 1e-12)
    q_vec = {i: q_mat[n] for n, i in enumerate(q_ids)}

    rankings, diagnostics = await build_rankings(
        queries, cands_by_qid, child_text, titles, q_vec, title_vec, reranker, cache
    )

    metrics = score_set(queries, rankings)
    changed = sum(d["top1_changed_B"] for d in diagnostics.values())
    mismatch = [
        (q.qid, d["top1_title_cos_A"], d["top1_title_A"])
        for q in queries
        for d in [diagnostics.get(q.qid)]
        if d and d["top1_title_cos_A"] < TITLE_MISMATCH_MAX_COS
    ]
    cos_values = [d["top1_title_cos_A"] for d in diagnostics.values()]

    summary = {
        "queries": len(queries),
        "labelled": sum(1 for q in queries if q.gold_doc_ids),
        "metrics": metrics,
        "top1_doc_changed_by_B": changed,
        "top1_doc_changed_pct": round(changed / max(1, len(queries)), 4),
        "title_mismatch": {
            "threshold_cos": TITLE_MISMATCH_MAX_COS,
            "count": len(mismatch),
            "pct": round(len(mismatch) / max(1, len(diagnostics)), 4),
            "cos_deciles": [
                round(float(np.percentile(cos_values, p)), 3) for p in range(10, 100, 10)
            ] if cos_values else [],
            "worst": [
                {"qid": q, "cos": c, "title": t[:110]}
                for q, c, t in sorted(mismatch, key=lambda x: x[1])[:10]
            ],
        },
    }

    rows: List[dict] = []
    for q in queries:
        d = diagnostics.get(q.qid, {})
        row = {
            "set": label, "qid": q.qid, "register": q.register,
            "category": q.category, "question": q.text,
            "gold_doc_ids": "|".join(q.gold_doc_ids),
        }
        for variant, per_query in rankings.items():
            ranked = doc_ranking(per_query.get(q.qid, []))
            row[f"{variant}_top1"] = ranked[0] if ranked else ""
            if q.gold_doc_ids:
                row[f"{variant}_hit1"] = hit_at_k(ranked, q.gold_doc_ids, 1)
                row[f"{variant}_rr"] = round(reciprocal_rank(ranked, q.gold_doc_ids), 4)
        row.update(d)
        rows.append(row)

    return summary, rows


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None,
                    help="cap each query set, for a smoke run")
    args = ap.parse_args()

    quiet_logs()
    print("Probe 1 — title priority\n")
    docs_meta = await load_documents()
    print(f"  corpus: {len(docs_meta)} documents")

    retriever = Retriever()
    reranker = get_reranker()
    cache = RerankCache("rerank_scores")

    subset_a = load_subset_a_queries(limit=args.limit)
    students = load_student_questions()[: args.limit] if args.limit else load_student_questions()

    try:
        sum_a, rows_a = await run_set("subset_a", subset_a, retriever, docs_meta, reranker, cache)
        sum_s, rows_s = await run_set("student", students, retriever, docs_meta, reranker, cache)
    finally:
        await retriever.close()
        await reranker.close()

    write_results("probe1_title", rows_a + rows_s,
                  {"subset_a": sum_a, "student": sum_s})

    print("\n  Document-level retrieval, subset_a (labelled)")
    print(f"    {'variant':18s} {'hit@1':>7s} {'hit@3':>7s} {'hit@5':>7s} {'MRR':>7s}")
    for variant, m in sum_a["metrics"].items():
        if m:
            print(f"    {variant:18s} {m['hit@1']:7.3f} {m['hit@3']:7.3f} "
                  f"{m['hit@5']:7.3f} {m['mrr']:7.3f}")

    base = sum_a["metrics"]["A_baseline"]
    print(f"\n    deltas vs baseline (hit@1 / MRR)")
    for variant, m in sum_a["metrics"].items():
        if m and variant != "A_baseline":
            print(f"      {variant:18s} {m['hit@1'] - base['hit@1']:+.3f} "
                  f"{m['mrr'] - base['mrr']:+.3f}")

    for label, s in (("subset_a", sum_a), ("student", sum_s)):
        tm = s["title_mismatch"]
        print(f"\n  [{label}] top-1 doc changed by variant B: "
              f"{s['top1_doc_changed_by_B']}/{s['queries']} ({s['top1_doc_changed_pct']:.1%})")
        print(f"    cos(query, title of top-1) deciles: {tm['cos_deciles']}")
        print(f"    below {tm['threshold_cos']}: {tm['count']} ({tm['pct']:.1%})")

    print("\n  Worst title mismatches on subset_a (chunk matched, title did not):")
    for w in sum_a["title_mismatch"]["worst"][:5]:
        print(f"    cos={w['cos']:.3f}  {w['qid']}  {w['title']}")


if __name__ == "__main__":
    asyncio.run(main())
