"""Probe 6 — decide the document at document level, not by its best chunk.

`SearchService` ranks chunks and the document ranking falls out of whichever
chunk happens to surface first (`doc_ranking`, first occurrence wins). That
discards a real signal: **how much of a document matched**. For "Berapa biaya
UKT?" the tariff document contributed 12 of the 50 candidate chunks and still
ranked 11th, because one outbound decree had a single chunk that scored higher.

Nothing here needs a new index, new embeddings or an LLM. It is pure
post-processing over the candidates and rerank scores probe 1 already cached,
which makes it the cheapest of the three levers and the one to try first.

    first_occurrence   current behaviour
    max_score          best chunk wins (what most systems do)
    sum_top3           sum of a document's three best chunk scores
    mean_top3          mean of them, so breadth does not swamp quality
    rrf_ranks          sum of 1/(k + chunk_rank) over the document's chunks
    chunk_count        how many chunks matched, ignoring score entirely

    python -m evals.probes.probe6_docagg

Writes results/probe6_docagg.csv / .summary.json. Read-only, no API cost.
"""

from __future__ import annotations

import asyncio
import json
from collections import defaultdict
from typing import Callable, Dict, List, Sequence, Tuple

from ._common import (
    CACHE_DIR,
    Candidate,
    Retriever,
    hit_at_k,
    load_documents,
    load_subset_a_queries,
    quiet_logs,
    reciprocal_rank,
    write_results,
)

RRF_K = 60

# Each takes the (rank, score) pairs a document contributed and returns a
# document score; higher is better.
AGGREGATORS: Dict[str, Callable[[List[Tuple[int, float]]], float]] = {
    "first_occurrence": lambda p: -min(r for r, _ in p),
    "max_score": lambda p: max(s for _, s in p),
    "sum_top3": lambda p: sum(sorted((s for _, s in p), reverse=True)[:3]),
    "mean_top3": lambda p: (
        sum(sorted((s for _, s in p), reverse=True)[:3]) / min(3, len(p))
    ),
    "rrf_ranks": lambda p: sum(1.0 / (RRF_K + r) for r, _ in p),
    "chunk_count": lambda p: float(len(p)),
}


def load_rerank_cache() -> Dict[str, List[float]]:
    path = CACHE_DIR / "rerank_scores.jsonl"
    if not path.exists():
        raise SystemExit("cache/rerank_scores.jsonl missing — run probe1_title first.")
    out: Dict[str, List[float]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                out[rec["k"]] = rec["s"]
    return out


def doc_scores(cands: Sequence[Candidate], scores: Sequence[float]
               ) -> Dict[str, List[Tuple[int, float]]]:
    """(rank, score) pairs contributed by each document, rank 1 = best chunk."""
    order = sorted(range(len(cands)), key=lambda i: scores[i], reverse=True)
    per: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
    for rank, i in enumerate(order, 1):
        per[cands[i].doc_id].append((rank, scores[i]))
    return per


async def main() -> None:
    quiet_logs()
    print("Probe 6 — document-level aggregation\n")

    docs = await load_documents()
    titles = {d.doc_id: d.title for d in docs.values()}
    rerank = load_rerank_cache()

    queries = [q for q in load_subset_a_queries() if q.gold_doc_ids]
    retriever = Retriever()
    cands_by_qid = await retriever.retrieve(queries)
    await retriever.close()

    metrics: Dict[str, dict] = {}
    per_query: Dict[str, Dict[str, List[str]]] = {}

    for name, agg in AGGREGATORS.items():
        hits: Dict[int, List[int]] = {1: [], 3: [], 5: []}
        rrs: List[float] = []
        per_query[name] = {}
        for q in queries:
            cands = cands_by_qid.get(q.qid, [])
            scores = rerank.get(f"A::{q.qid}")
            if not cands or scores is None or len(scores) != len(cands):
                continue
            per = doc_scores(cands, scores)
            ranked = sorted(per, key=lambda d: agg(per[d]), reverse=True)
            per_query[name][q.qid] = ranked
            for k in hits:
                hits[k].append(hit_at_k(ranked, q.gold_doc_ids, k))
            rrs.append(reciprocal_rank(ranked, q.gold_doc_ids))
        n = len(rrs)
        metrics[name] = {
            "n": n,
            "hit@1": round(sum(hits[1]) / n, 4),
            "hit@3": round(sum(hits[3]) / n, 4),
            "hit@5": round(sum(hits[5]) / n, 4),
            "mrr": round(sum(rrs) / n, 4),
        }

    base = metrics["first_occurrence"]
    print(f"  {'aggregator':18s} {'hit@1':>7s} {'hit@3':>7s} {'hit@5':>7s} "
          f"{'MRR':>7s}   {'dhit@1':>7s} {'dMRR':>7s}")
    for name, m in metrics.items():
        mark = "  <-- current" if name == "first_occurrence" else ""
        print(f"  {name:18s} {m['hit@1']:7.3f} {m['hit@3']:7.3f} {m['hit@5']:7.3f} "
              f"{m['mrr']:7.3f}   {m['hit@1']-base['hit@1']:+7.3f} "
              f"{m['mrr']-base['mrr']:+7.3f}{mark}")

    # Paired win/loss against current, which matters more than the mean at n=115.
    print("\n  vs first_occurrence, per query (hit@1):")
    wl: Dict[str, Tuple[int, int]] = {}
    for name in AGGREGATORS:
        if name == "first_occurrence":
            continue
        w = l = 0
        for q in queries:
            a = per_query["first_occurrence"].get(q.qid)
            b = per_query[name].get(q.qid)
            if not a or not b:
                continue
            ha = hit_at_k(a, q.gold_doc_ids, 1)
            hb = hit_at_k(b, q.gold_doc_ids, 1)
            w += hb > ha
            l += hb < ha
        wl[name] = (w, l)
        print(f"    {name:18s} better on {w:3d}, worse on {l:3d}  (net {w - l:+d})")

    rows: List[dict] = []
    for q in queries:
        row = {"qid": q.qid, "question": q.text,
               "gold_doc_ids": "|".join(q.gold_doc_ids)}
        for name in AGGREGATORS:
            ranked = per_query[name].get(q.qid, [])
            row[f"{name}_top1"] = titles.get(ranked[0], "")[:70] if ranked else ""
            row[f"{name}_hit1"] = hit_at_k(ranked, q.gold_doc_ids, 1) if ranked else 0
            row[f"{name}_rr"] = (round(reciprocal_rank(ranked, q.gold_doc_ids), 4)
                                 if ranked else 0.0)
        rows.append(row)

    write_results("probe6_docagg", rows, {
        "metrics": metrics,
        "paired_vs_first_occurrence": {k: {"better": v[0], "worse": v[1]}
                                       for k, v in wl.items()},
        "note": "Post-processing only over probe 1's cached candidates and "
                "rerank scores. No new index, embeddings or LLM calls.",
    })


if __name__ == "__main__":
    asyncio.run(main())
