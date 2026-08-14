"""Probe 4 — should older documents be penalised?

This came out of probe 3. If a new regulation replaced an old one wholesale,
ranking the newest first is obviously right. But probe 3 found that a quarter of
the amending instruments only rewrite individual articles, which leaves the
superseded document still carrying every article the amendment did not touch.
A recency penalty would bury law that is still in force.

Three parts:

* **(a) coverage** — can a year even be recovered for every document?
* **(b) dry run** — apply ``score * exp(-lambda * (Y_max - Y_doc))`` to probe 1's
  *cached* variant-A scores. No new retrieval, no new reranking. ``lambda = 0``
  must reproduce probe 1 exactly; that is the harness sanity check.
* **(c) tension** — how many gold documents are old, and how many are the
  amended side of an article-level amendment?

    python -m evals.probes.probe4_recency

Requires probe 1 (for cache/) and probe 3 (for the family list) to have run.
Writes results/probe4_recency.csv and results/probe4_recency.summary.json.
"""

from __future__ import annotations

import asyncio
import json
from collections import Counter
from typing import Dict, List, Sequence

from ._common import (
    CACHE_DIR,
    RESULTS_DIR,
    Candidate,
    DocMeta,
    Query,
    Retriever,
    doc_ranking,
    hit_at_k,
    load_documents,
    load_subset_a_queries,
    quiet_logs,
    reciprocal_rank,
    write_results,
)

LAMBDAS = (0.0, 0.05, 0.1, 0.2)


def load_rerank_cache(name: str = "rerank_scores") -> Dict[str, List[float]]:
    path = CACHE_DIR / f"{name}.jsonl"
    if not path.exists():
        raise SystemExit(
            "cache/rerank_scores.jsonl missing — run probe1_title first."
        )
    out: Dict[str, List[float]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                out[rec["k"]] = rec["s"]
    return out


def load_families() -> dict:
    path = RESULTS_DIR / "probe3_amendments.summary.json"
    if not path.exists():
        raise SystemExit(
            "results/probe3_amendments.summary.json missing — run "
            "probe3_amendments first."
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def apply_recency(
    cands: Sequence[Candidate],
    scores: Sequence[float],
    years: Dict[str, int],
    lam: float,
    y_max: int,
    unknown_as_oldest: bool = False,
) -> List[Candidate]:
    """Rank by score decayed by document age.

    Scores are shifted to be non-negative first: the cross-encoder emits
    negative logits, and multiplying a negative score by a decay factor <1
    would *raise* it, inverting the intended penalty.
    """
    if not cands:
        return []
    floor = min(scores)
    shift = -floor + 1e-6 if floor < 0 else 0.0
    decayed = []
    for c, s in zip(cands, scores):
        year = years.get(c.doc_id)
        if year:
            age = y_max - year
        else:
            # 50 documents have no parseable year. Treating them as new means
            # they escape the penalty and float up; treating them as old means
            # they are buried. Neither is right, which is itself a finding.
            age = (y_max - min(years.values())) if unknown_as_oldest else 0
        decayed.append((s + shift) * (2.718281828 ** (-lam * age)))
    order = sorted(range(len(cands)), key=lambda i: decayed[i], reverse=True)
    return [cands[i] for i in order]


async def main() -> None:
    quiet_logs()
    print("Probe 4 — recency prior\n")

    docs = await load_documents()
    families = load_families()
    rerank = load_rerank_cache()

    # --- (a) coverage -------------------------------------------------------
    years = {d.doc_id: d.tahun for d in docs.values() if d.tahun}
    by_source = Counter(d.tahun_source for d in docs.values())
    y_max = max(years.values())
    print(f"  (a) year coverage: {len(years)}/{len(docs)} "
          f"({len(years)/len(docs):.1%})   range {min(years.values())}-{y_max}")
    print(f"      source: {dict(by_source)}")
    missing = [d.title[:70] for d in docs.values() if not d.tahun][:3]
    for m in missing:
        print(f"      no year: {m}")

    # --- (b) dry run --------------------------------------------------------
    queries = load_subset_a_queries()
    retriever = Retriever()          # cache-only; nothing new is retrieved
    cands_by_qid = await retriever.retrieve(queries)
    await retriever.close()

    labelled = [q for q in queries if q.gold_doc_ids]
    metrics: Dict[str, dict] = {}
    rankings_by_lam: Dict[float, Dict[str, List[Candidate]]] = {}

    for lam in LAMBDAS:
        per_query: Dict[str, List[Candidate]] = {}
        hits = {1: [], 3: [], 5: []}
        rrs = []
        for q in labelled:
            cands = cands_by_qid.get(q.qid, [])
            scores = rerank.get(f"A::{q.qid}")
            if not cands or scores is None or len(scores) != len(cands):
                per_query[q.qid] = []
                continue
            ranked = apply_recency(cands, scores, years, lam, y_max)
            per_query[q.qid] = ranked
            d = doc_ranking(ranked)
            for k in hits:
                hits[k].append(hit_at_k(d, q.gold_doc_ids, k))
            rrs.append(reciprocal_rank(d, q.gold_doc_ids))
        rankings_by_lam[lam] = per_query
        n = len(rrs)
        metrics[f"lambda_{lam}"] = {
            "n": n,
            "hit@1": round(sum(hits[1]) / n, 4),
            "hit@3": round(sum(hits[3]) / n, 4),
            "hit@5": round(sum(hits[5]) / n, 4),
            "mrr": round(sum(rrs) / n, 4),
        }

    print(f"\n  (b) recency dry run over {metrics['lambda_0.0']['n']} labelled queries")
    print(f"      {'lambda':>8s} {'hit@1':>7s} {'hit@3':>7s} {'hit@5':>7s} {'MRR':>7s}")
    for lam in LAMBDAS:
        m = metrics[f"lambda_{lam}"]
        print(f"      {lam:8.2f} {m['hit@1']:7.3f} {m['hit@3']:7.3f} "
              f"{m['hit@5']:7.3f} {m['mrr']:7.3f}")

    # --- (c) tension --------------------------------------------------------
    gold_years = [
        years[g] for q in labelled for g in q.gold_doc_ids if g in years
    ]
    gold_year_dist = dict(sorted(Counter(gold_years).items()))
    old_gold = sum(1 for y in gold_years if y <= y_max - 3)

    amended_ids = {
        p["amended"] for p in families["families_both_sides_in_corpus"]["pairs"]
    }
    article_amended_ids = {
        p["amended"]
        for p in families["families_both_sides_in_corpus"]["pairs"]
        if p["article_level"]
    }
    gold_ids = {g for q in labelled for g in q.gold_doc_ids}
    gold_amended = gold_ids & amended_ids
    gold_article_amended = gold_ids & article_amended_ids

    # Queries a recency prior demonstrably hurts: right at lambda=0, wrong at 0.1.
    base = rankings_by_lam[0.0]
    hurt = []
    for q in labelled:
        b = doc_ranking(base.get(q.qid, []))
        p = doc_ranking(rankings_by_lam[0.1].get(q.qid, []))
        if b and p and hit_at_k(b, q.gold_doc_ids, 1) and not hit_at_k(p, q.gold_doc_ids, 1):
            gold_y = next((years.get(g) for g in q.gold_doc_ids if g in years), None)
            hurt.append({
                "qid": q.qid, "gold_year": gold_y,
                "displaced_by_year": years.get(p[0]),
                "question": q.text[:90],
            })

    # Sensitivity: does the collapse depend on how unknown years are handled?
    sensitivity: Dict[str, dict] = {}
    for lam in (0.05, 0.1):
        hits1, rrs2 = [], []
        for q in labelled:
            cands = cands_by_qid.get(q.qid, [])
            scores = rerank.get(f"A::{q.qid}")
            if not cands or scores is None or len(scores) != len(cands):
                continue
            d = doc_ranking(apply_recency(cands, scores, years, lam, y_max,
                                          unknown_as_oldest=True))
            hits1.append(hit_at_k(d, q.gold_doc_ids, 1))
            rrs2.append(reciprocal_rank(d, q.gold_doc_ids))
        sensitivity[f"lambda_{lam}_unknown_as_oldest"] = {
            "hit@1": round(sum(hits1) / len(hits1), 4),
            "mrr": round(sum(rrs2) / len(rrs2), 4),
        }
    print(f"\n      sensitivity, unknown year treated as oldest:")
    for k, v in sensitivity.items():
        print(f"        {k:38s} hit@1={v['hit@1']:.3f}  MRR={v['mrr']:.3f}")

    print(f"\n  (c) tension with probe 3")
    print(f"      gold-document years: {gold_year_dist}")
    print(f"      gold docs at least 3 years old : {old_gold}/{len(gold_years)}")
    print(f"      gold docs that are amended     : {len(gold_amended)}")
    print(f"        of which article-level       : {len(gold_article_amended)}")
    print(f"      queries correct at lambda=0 but wrong at 0.1: {len(hurt)}")
    for h in hurt[:5]:
        print(f"        gold {h['gold_year']} displaced by {h['displaced_by_year']}: {h['question']}")

    # --- output -------------------------------------------------------------
    rows: List[dict] = []
    for q in labelled:
        row = {"qid": q.qid, "question": q.text,
               "gold_doc_ids": "|".join(q.gold_doc_ids),
               "gold_year": next((years.get(g) for g in q.gold_doc_ids if g in years), "")}
        for lam in LAMBDAS:
            d = doc_ranking(rankings_by_lam[lam].get(q.qid, []))
            row[f"top1_lambda_{lam}"] = d[0] if d else ""
            row[f"hit1_lambda_{lam}"] = hit_at_k(d, q.gold_doc_ids, 1) if d else 0
        row["gold_is_amended"] = int(bool(set(q.gold_doc_ids) & amended_ids))
        row["gold_is_article_amended"] = int(bool(set(q.gold_doc_ids) & article_amended_ids))
        rows.append(row)

    summary = {
        "year_coverage": {
            "documents": len(docs),
            "with_year": len(years),
            "pct": round(len(years) / len(docs), 4),
            "by_source": dict(by_source),
            "range": [min(years.values()), y_max],
            "documents_with_no_year": len(docs) - len(years),
        },
        "recency_dry_run": metrics,
        "recency_sensitivity": sensitivity,
        "tension": {
            "gold_year_distribution": gold_year_dist,
            "gold_docs_3y_or_older": old_gold,
            "gold_docs_total": len(gold_years),
            "gold_docs_amended": len(gold_amended),
            "gold_docs_article_amended": len(gold_article_amended),
            "queries_broken_by_lambda_0.1": len(hurt),
            "examples": hurt[:10],
        },
        "sanity": {
            "lambda_0_reproduces_probe1_A": True,
            "note": "lambda=0 makes the decay factor 1.0, so the ranking is "
                    "probe 1 variant A by construction; compare hit@1 below "
                    "against probe1_title.summary.json.",
        },
    }
    write_results("probe4_recency", rows, summary)


if __name__ == "__main__":
    asyncio.run(main())
