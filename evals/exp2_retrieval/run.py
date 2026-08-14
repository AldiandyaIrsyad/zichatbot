"""Experiment 2 — Retrieval Quality Evaluation.

Evaluates hybrid retrieval (dense + sparse with RRF fusion) against
dense-only and sparse-only baselines using Subset A (RAG QA triplets).

Metrics: Hit Rate@k (k=1,3,5) and MRR, reported per category and overall.

Usage:
    python -m evals.exp2_retrieval.run \\
        --dataset evals/data/subset_a.csv \\
        --api-url http://localhost:8000
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import httpx

from evals._shared.csv_export import write_results_csv
from evals._shared.dataset import load_subset_a, SubsetARow
from evals._shared.metrics import (
    RetrievalMetrics,
    compute_retrieval_metrics,
    hit_rate_at_k,
    reciprocal_rank_at_k,
)

DEFAULT_OUTPUT_CSV = "evals/data/results/exp2_retrieval.csv"


@dataclass
class CategoryResult:
    """Retrieval metrics for a single category."""

    category: str
    metrics: RetrievalMetrics
    sample_count: int


async def retrieve(
    api_url: str,
    query: str,
    top_k: int,
    mode: str,
    rerank: bool = True,
    hyde: Optional[bool] = None,
) -> List[str]:
    """Retrieve document IDs from the KB search endpoint.

    ``hyde`` selects the endpoint (HyDE ablation, E6): ``None`` (default) hits
    ``/api/kb/search``, which is structurally HyDE-off and preserves the
    original behaviour. ``True``/``False`` hit ``/api/chat/search`` with the
    ``hyde`` toggle — the chat-domain endpoint that can inject the production
    ``HyDEExpander`` (``/api/kb/search`` cannot, per the ``kb/ ⇏ chat/``
    dependency rule). Routing both HyDE conditions through the one chat endpoint
    makes HyDE the single varied factor; ``hyde=False`` there reproduces
    ``/api/kb/search``.

    ``/api/kb/search`` returns CHUNK-level hits, each carrying the doc_id it
    came from — several of the top-k chunks routinely belong to the same
    document (measured: top-5 averages only ~3.8 unique documents across all
    three modes). This function returns the raw chunk-level list; callers
    computing Hit Rate@k/MRR@k should deduplicate first (see dedup_doc_ids) so
    "@k" means "top-k distinct documents", not "top-k chunks that may repeat
    the same 2-3 documents".

    Args:
        rerank: Whether the cross-encoder reranker runs. With it on, all three
            modes are really "mode -> rerank", so the comparison is between
            reranked variants rather than between the fusion strategies
            themselves.

    Returns:
        List of retrieved doc_ids in ranked (chunk-level) order.
    """
    params: Dict[str, Any] = {"q": query, "top_k": top_k, "mode": mode, "rerank": str(rerank).lower()}
    if hyde is None:
        endpoint = "/api/kb/search"
    else:
        endpoint = "/api/chat/search"
        params["hyde"] = str(hyde).lower()
    # The HyDE ensemble adds three concurrent LLM generations and a second
    # dense retrieval call per query, so allow more headroom than pure retrieval.
    timeout = 30.0 if hyde is None else 120.0
    try:
        async with httpx.AsyncClient(base_url=api_url, timeout=timeout) as client:
            response = await client.get(endpoint, params=params)
            if response.status_code != 200:
                return []
            results = response.json()
            return [r.get("doc_id", "") for r in results]
    except httpx.HTTPError:
        # A slow/failed HyDE call is a retrieval miss for this one query, not a
        # run-ending crash.
        return []


def dedup_doc_ids(doc_ids: List[str]) -> List[str]:
    """Collapse a chunk-level doc_id list to unique documents, rank-preserved.

    Keeps the first (best-ranked) occurrence of each doc_id and drops
    later repeats, so a document's rank in the deduped list is the best
    rank any of its chunks achieved.

    Returns:
        Unique doc_ids in first-occurrence (best-rank) order.
    """
    seen: set = set()
    unique: List[str] = []
    for doc_id in doc_ids:
        if doc_id and doc_id not in seen:
            seen.add(doc_id)
            unique.append(doc_id)
    return unique


async def evaluate_mode(
    api_url: str,
    dataset: List[SubsetARow],
    mode: str,
    top_k: int = 5,
    rerank: bool = True,
    hyde: Optional[bool] = None,
    concurrency: int = 1,
) -> Tuple[RetrievalMetrics, List[Tuple[SubsetARow, List[str], List[str]]]]:
    """Evaluate a single retrieval mode over the dataset.

    Deduplicates each query's chunk-level results to unique documents before
    computing Hit Rate@k/MRR@k, so "@k" is measured over document rank, not
    chunk rank.

    Returns:
        Tuple of (aggregated RetrievalMetrics computed on deduped doc
        rankings, per-query raw results as (row, raw_chunk_doc_ids,
        deduped_doc_ids) triples — deduped_doc_ids is what metrics were
        computed from; raw_chunk_doc_ids is kept for CSV transparency and for
        compute_per_category to regroup without re-querying).
    """
    if concurrency < 1:
        raise ValueError("concurrency must be at least 1")
    semaphore = asyncio.Semaphore(concurrency)

    async def retrieve_one(row: SubsetARow) -> Tuple[SubsetARow, List[str], List[str]]:
        async with semaphore:
            retrieved_ids = await retrieve(api_url, row.question, top_k, mode, rerank, hyde)
        return row, retrieved_ids, dedup_doc_ids(retrieved_ids)

    # gather preserves input order, retaining exact row alignment for paired tests.
    raw_per_query = list(await asyncio.gather(*(retrieve_one(row) for row in dataset)))
    results_per_query = [(deduped_ids, row.gold_doc_ids) for row, _raw_ids, deduped_ids in raw_per_query]
    return compute_retrieval_metrics(results_per_query), raw_per_query


def compute_per_category(
    raw_per_query: List[Tuple[SubsetARow, List[str], List[str]]],
) -> List[CategoryResult]:
    """Regroup an already-computed evaluate_mode() pass by category.

    Regroups the single pass's raw per-query results rather than re-running
    every query through the API a second time, so per-category numbers are
    guaranteed consistent with the overall figure they are a breakdown of (and
    Exp2's API/HyDE cost is not doubled).

    Args:
        raw_per_query: The (row, raw_chunk_doc_ids, deduped_doc_ids)
            triples returned by evaluate_mode.

    Returns:
        List of CategoryResult, one per category.
    """
    by_category: Dict[str, List[Tuple[List[str], List[str]]]] = defaultdict(list)
    for row, _raw_ids, deduped_ids in raw_per_query:
        by_category[row.category].append((deduped_ids, row.gold_doc_ids))

    results: List[CategoryResult] = []
    for category, pairs in sorted(by_category.items()):
        results.append(
            CategoryResult(
                category=category,
                metrics=compute_retrieval_metrics(pairs),
                sample_count=len(pairs),
            )
        )
    return results


def print_report(
    mode_name: str,
    overall: RetrievalMetrics,
    per_category: List[CategoryResult],
) -> None:
    """Print a formatted retrieval evaluation report."""
    print(f"\n{'=' * 70}")
    print(f"  Retrieval Mode: {mode_name}")
    print(f"{'=' * 70}")
    print(f"  Total Queries: {overall.query_count}")
    print()
    print(f"  {'Metric':<20} {'Value':>10}")
    print(f"  {'-' * 32}")
    print(f"  {'Hit Rate@1':<20} {overall.hit_rate_at_1:>10.4f}")
    print(f"  {'Hit Rate@3':<20} {overall.hit_rate_at_3:>10.4f}")
    print(f"  {'Hit Rate@5':<20} {overall.hit_rate_at_5:>10.4f}")
    print(f"  {'MRR@1':<20} {overall.mrr_at_1:>10.4f}")
    print(f"  {'MRR@3':<20} {overall.mrr_at_3:>10.4f}")
    print(f"  {'MRR@5':<20} {overall.mrr_at_5:>10.4f}")

    if per_category:
        print(f"\n  Per Category:")
        print(f"  {'Category':<25} {'N':>5} {'HR@1':>8} {'HR@3':>8} {'HR@5':>8} {'MRR@5':>8}")
        print(f"  {'-' * 70}")
        for r in per_category:
            m = r.metrics
            print(
                f"  {r.category:<25} {r.sample_count:>5} "
                f"{m.hit_rate_at_1:>8.4f} {m.hit_rate_at_3:>8.4f} "
                f"{m.hit_rate_at_5:>8.4f} {m.mrr_at_5:>8.4f}"
            )
    print()


async def async_main(args: argparse.Namespace) -> None:
    """Async entry point for Experiment 2."""
    try:
        dataset = load_subset_a(args.dataset)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    # Filter out out-of-domain rows: they have source_doc_id="NONE" and
    # always score 0 on retrieval, which would pollute the metrics without
    # providing meaningful signal.
    in_domain_dataset = [r for r in dataset if r.category != "out-of-domain"]
    skipped = len(dataset) - len(in_domain_dataset)
    print(f"Loaded {len(dataset)} samples from Subset A ({skipped} out-of-domain rows filtered)")
    dataset = in_domain_dataset

    modes_to_eval: List[str] = []
    if args.mode == "all":
        modes_to_eval = ["hybrid", "dense", "sparse"]
    else:
        modes_to_eval = [args.mode]

    # Report both rerank conditions, so the table separates "which fusion
    # strategy is better" from "how much of the difference the reranker was
    # absorbing".
    rerank_conditions: List[bool] = {
        "both": [True, False],
        "on": [True],
        "off": [False],
    }[args.rerank]

    # E6 HyDE ablation. None = the original /api/kb/search path (HyDE-off by
    # construction); True/False route through /api/chat/search so HyDE is the
    # single varied factor. 'both' reports HyDE off then on on that endpoint.
    hyde_conditions: List[Optional[bool]] = {
        "off": [None],
        "on": [True],
        "both": [False, True],
    }[args.hyde]

    csv_rows = []
    for hyde in hyde_conditions:
        for rerank in rerank_conditions:
            for mode in modes_to_eval:
                hyde_label = "" if hyde is None else f", hyde {'on' if hyde else 'off'}"
                label = f"{mode} (rerank {'on' if rerank else 'off'}{hyde_label})"
                print(f"\nEvaluating mode: {label}")
                overall, raw_per_query = await evaluate_mode(
                    args.api_url, dataset, mode, args.top_k, rerank, hyde, args.concurrency
                )
                per_category = compute_per_category(raw_per_query)
                print_report(label.upper(), overall, per_category)

                for row, retrieved_ids, deduped_ids in raw_per_query:
                    relevant = row.gold_doc_ids
                    csv_rows.append({
                        "question": row.question,
                        "category": row.category,
                        "mode": mode,
                        "rerank": rerank,
                        "hyde": "" if hyde is None else str(hyde).lower(),
                        "source_doc_id": row.source_doc_id,
                        "retrieved_doc_ids": "|".join(retrieved_ids),
                        "retrieved_doc_ids_dedup": "|".join(deduped_ids),
                        "hit_at_1": hit_rate_at_k(deduped_ids, relevant, 1),
                        "hit_at_3": hit_rate_at_k(deduped_ids, relevant, 3),
                        "hit_at_5": hit_rate_at_k(deduped_ids, relevant, 5),
                        "reciprocal_rank": reciprocal_rank_at_k(deduped_ids, relevant, args.top_k),
                    })

    write_results_csv(args.output_csv, csv_rows)


def main() -> None:
    """Entry point for Experiment 2."""
    parser = argparse.ArgumentParser(
        description="Experiment 2: Retrieval Quality Evaluation (hybrid vs dense vs sparse)."
    )
    parser.add_argument("--dataset", required=True, help="Path to Subset A CSV")
    parser.add_argument(
        "--api-url",
        default="http://localhost:8000",
        help="Base URL of the running application (for KB search)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Maximum k for Hit Rate@k (default: 5)",
    )
    parser.add_argument(
        "--mode",
        default="all",
        choices=["all", "hybrid", "dense", "sparse"],
        help="Retrieval mode to evaluate (default: all)",
    )
    parser.add_argument(
        "--rerank",
        default="both",
        choices=["both", "on", "off"],
        help="Cross-encoder reranker condition. With the reranker on, every mode is "
        "really 'mode -> rerank', so the ablation compares reranked variants rather "
        "than the fusion strategies themselves. Default 'both' reports each mode twice.",
    )
    parser.add_argument(
        "--hyde",
        default="off",
        choices=["off", "on", "both"],
        help="HyDE retrieval-ensemble ablation (E6). 'off' (default) uses /api/kb/search "
        "(HyDE structurally off — the original behaviour). 'on'/'both' route through "
        "/api/chat/search, which injects the production three-passage HyDE ensemble; 'both' "
        "reports HyDE off then on on that endpoint so HyDE is the single varied factor. "
        "HyDE-on adds three concurrent LLM calls and one dense retrieval call per query.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Maximum concurrent retrieval requests (default: 1; use a small value for hosted HyDE)",
    )
    parser.add_argument(
        "--output-csv",
        default=DEFAULT_OUTPUT_CSV,
        help="Path to write raw per-row results CSV",
    )
    args = parser.parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
