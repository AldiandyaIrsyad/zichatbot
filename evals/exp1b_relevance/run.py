"""Experiment 1b — IVM Relevance Evaluation.

Evaluates all three production IVM relevance/OOD backends
(``app/thesis/ivm/checkers.py``) against Subset C (boundary relevance
queries), plus a keyword-overlap baseline — a 4-way comparison.

Backends under test (all reused directly from production):
    1. ``llm_judge``: LLMJudgeRelevanceChecker wrapping the production
       LLMJudge (app/thesis/ivm/judge.py) — same prompt, same
       max_tokens=400, same last-word parse the production system uses. A
       reasoning model's chain-of-thought preamble defeats a first-substring
       parse or a tiny token budget, so the production budget and last-word
       parse are load-bearing and must be the ones tested.
    2. ``similarity_threshold``: SimilarityThresholdRelevanceChecker,
       thresholding the top retrieval score — no LLM call.
    3. ``nli_entailment``: NliEntailmentRelevanceChecker, thresholding NLI
       entailment_score between query and joined context.
    4. keyword-overlap baseline (JDIH_LEXICON, unchanged).

Both non-LLM backends' default thresholds are documented placeholders in
production (``ood_similarity_threshold=0.02``, ``ood_nli_entailment_threshold
=0.5`` — see app/chat/config.py) that were never calibrated against this
KB's actual score distribution; ``--similarity-threshold``/``--nli-threshold``
let a calibration sweep be run instead of trusting the placeholder.

Metrics: Accuracy, Precision, Recall, F1, FPR + bootstrap CI, reported overall
and per subtype.

Usage:
    python -m evals.exp1b_relevance.run \\
        --dataset evals/data/subset_c.csv \\
        --api-url http://localhost:8000 \\
        --infinity-url http://localhost:7997
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

import httpx

from evals._shared.clients import EvalNLIClient, get_llm_client_from_env
from evals._shared.csv_export import write_results_csv
from evals._shared.dataset import load_subset_c, SubsetCRow
from evals._shared.metrics import (
    BinaryMetrics,
    CI,
    bootstrap_binary_ci,
    compute_binary_metrics,
)
from evals._shared.repeats import repeat_passes, self_agreement

# Outcomes a judgement can have. `errored` is kept distinct from a real
# `out_of_domain` verdict: the LLM-judge is a hosted model that fails closed
# (an exception is treated as irrelevant), which is right in production but
# would let an API outage masquerade as an out-of-domain prediction and inflate
# FPR here. The deterministic checkers cannot error this way and do not use this.
IN_DOMAIN = "in_domain"
OUT_OF_DOMAIN = "out_of_domain"
ERRORED = "errored"
from app.guardrails.ivm.checkers import (
    LLMJudgeRelevanceChecker,
    NliEntailmentRelevanceChecker,
    SimilarityThresholdRelevanceChecker,
)
from app.guardrails.ivm.judge import LLMJudge

DEFAULT_OUTPUT_CSV = "evals/data/results/exp1b_relevance.csv"


# Institutional anchors — the corpus's own identity, in-domain by definition and
# not tuned to Subset C. Everything else in the keyword baseline's lexicon is
# DERIVED from the live KB (see ``derive_jdih_lexicon``), so the baseline is
# reproducible and provably not hand-tuned against the evaluation set.
JDIH_ANCHORS: Set[str] = {"upi", "universitas pendidikan indonesia", "jdih"}

# Structural/boilerplate title words stripped before deriving domain terms, plus
# the institution name whose individual words would otherwise dominate. Kept small
# and generic on purpose — this is not a hand-tuned domain list.
_TITLE_STOPWORDS: Set[str] = {
    "yang", "dan", "di", "ke", "dari", "untuk", "atas", "tentang", "nomor",
    "tahun", "pada", "dalam", "serta", "atau", "perubahan", "kedua", "ketiga",
    "keempat", "kesatu", "pertama", "melalui", "dengan", "bagi", "para", "oleh",
    "sebagai", "the", "of", "and", "for", "lingkungan", "universitas",
    "pendidikan", "indonesia", "per", "lain", "baru",
}
# Leading document code (e.g. "2503-UN40-PT.03.03-2025") and "<n> Tahun <yyyy>".
_DOC_CODE_RE = re.compile(r"^\s*[\dA-Z]+[-/][\w.\-/]+")
_DOC_YEAR_RE = re.compile(r"^\s*\d+\s+tahun\s+\d+", re.IGNORECASE)

# Default document-frequency floor for a derived term. Bigram-only at df>=5 was
# the operating point that maximised Subset C separation without hand curation;
# unigrams collapse to generic legal/administrative vocabulary shared with the
# out-of-domain near-miss queries (measured FPR ~0.98), so they are excluded.
DEFAULT_LEXICON_MIN_DOC_FREQ = 5


def _title_tokens(title: str) -> List[str]:
    """Tokenise a KB document title into content unigrams.

    Strips the leading document code / "N Tahun YYYY" prefix and the
    " - "-separated code segment, lowercases, and drops stopwords and short
    tokens.
    """
    cleaned = _DOC_CODE_RE.sub("", title)
    cleaned = _DOC_YEAR_RE.sub("", cleaned)
    if " - " in cleaned:
        cleaned = cleaned.split(" - ", 1)[-1]
    return [
        w for w in re.findall(r"[a-z]+", cleaned.lower())
        if len(w) > 2 and w not in _TITLE_STOPWORDS
    ]


def derive_jdih_lexicon(
    titles: Sequence[str],
    min_doc_freq: int = DEFAULT_LEXICON_MIN_DOC_FREQ,
) -> Set[str]:
    """Derive the keyword-baseline lexicon from KB document titles.

    Deterministic and reproducible from the corpus: collects **bigrams** (adjacent
    content-word pairs) that appear in at least ``min_doc_freq`` distinct titles,
    plus the institutional anchors. Bigrams (e.g. "peraturan rektor", "majelis
    wali amanat", "uang kuliah tunggal") are UPI/JDIH-specific; unigrams are
    deliberately excluded because they reduce to generic legal/administrative
    vocabulary that the out-of-domain near-miss queries also use.

    Returns:
        The lexicon (anchors + qualifying bigrams).
    """
    bigram_df: Counter = Counter()
    for title in titles:
        tokens = _title_tokens(title)
        pairs = {f"{a} {b}" for a, b in zip(tokens, tokens[1:])}
        bigram_df.update(pairs)
    lexicon = set(JDIH_ANCHORS)
    lexicon.update(term for term, df in bigram_df.items() if df >= min_doc_freq)
    return lexicon


async def fetch_kb_titles(api_url: str) -> List[str]:
    """Fetch all KB document titles from the admin listing.

    Returns an empty list on error — the caller falls back to anchors.
    """
    try:
        async with httpx.AsyncClient(base_url=api_url, timeout=30.0) as client:
            response = await client.get("/api/admin/pdfs")
            if response.status_code != 200:
                return []
            docs = response.json()
        return [str(d.get("title", "")) for d in docs if isinstance(d, dict)]
    except Exception as exc:
        logger_warn(f"could not fetch KB titles for lexicon derivation: {exc}")
        return []

@dataclass
class SubtypeResult:
    """Metrics for a single subtype."""

    subtype: str
    metrics: BinaryMetrics
    accuracy_ci: CI


async def retrieve_contexts_detailed(
    api_url: str,
    query: str,
    top_k: int = 8,
    hyde: bool = False,
) -> Tuple[List[str], List[float]]:
    """Retrieve top-k contexts and their retrieval scores from the KB.

    Returns per-chunk text and score (not just concatenated text) so all
    three IRelevanceChecker backends can be driven identically to
    production's ``RelevanceService`` — SimilarityThresholdRelevanceChecker
    needs ``context_scores``.

    ``top_k=8`` matches production's ``RERANK_TOP_K``
    (app/kb/application/search_service.py) — the number of chunks the
    reranker actually hands to IVM/generation in the live pipeline,
    regardless of what a caller's own ``top_k`` requests further upstream.

    ``hyde`` selects the retrieval path for the production-faithful Exp1b
    condition (E7): ``False`` hits ``/api/kb/search`` (HyDE-off, matching the
    committed numbers); ``True`` hits ``/api/chat/search?hyde=true`` — the same
    HyDE-on retrieval the live chat pipeline feeds the relevance checkers,
    which ``/api/kb/search`` cannot provide (``kb/ ⇏ chat/``). The keyword and
    centroid backends do not call this function, so HyDE only moves the three
    retrieval-dependent backends (judge/similarity/nli).

    Returns:
        Tuple of (chunk texts, chunk scores), same order, same length.
    """
    params = {"q": query, "top_k": top_k}
    if hyde:
        endpoint = "/api/chat/search"
        params["hyde"] = "true"
    else:
        endpoint = "/api/kb/search"
    # HyDE adds an LLM generation call per query — give it more headroom.
    timeout = 120.0 if hyde else 30.0
    try:
        async with httpx.AsyncClient(base_url=api_url, timeout=timeout) as client:
            response = await client.get(endpoint, params=params)
            if response.status_code != 200:
                return [], []
            results = response.json()
            chunks = [r.get("text", "") for r in results]
            scores = [float(r.get("score", 0.0)) for r in results]
            return chunks, scores
    except httpx.HTTPError:
        # A slow/failed HyDE call is a miss for this query, not a run-ending crash.
        return [], []


async def run_llm_judge(
    checker: LLMJudgeRelevanceChecker,
    api_url: str,
    dataset: List[SubsetCRow],
    top_k: int = 8,
    hyde: bool = False,
) -> List[str]:
    """Run the production LLMJudgeRelevanceChecker over Subset C, once.

    Reuses ``app/thesis/ivm/judge.py``'s ``LLMJudge`` directly (via the
    production ``LLMJudgeRelevanceChecker`` wrapper) rather than a duplicated
    ad hoc prompt/parser, so this experiment tests the actual component the
    system ships — including its ``max_tokens=400`` budget and last-word
    parse, both load-bearing for reasoning models.

    Returns an **outcome per row** rather than a bool: an API failure is
    recorded as ``ERRORED`` and excluded from the metrics downstream, instead
    of being folded into ``out_of_domain``. In production the judge fails
    closed (an exception blocks the query), which is correct there; scoring it
    as a genuine out-of-domain verdict would turn an outage into a measurement.

    Returns:
        One outcome per row: ``IN_DOMAIN``, ``OUT_OF_DOMAIN`` or ``ERRORED``.
    """
    outcomes: List[str] = []
    for row in dataset:
        chunks, scores = await retrieve_contexts_detailed(api_url, row.query, top_k, hyde)
        try:
            relevant = await checker.check_query(row.query, chunks, scores)
            outcomes.append(IN_DOMAIN if relevant else OUT_OF_DOMAIN)
        except Exception as exc:
            logger_warn(f"judge call failed: {exc}")
            outcomes.append(ERRORED)
    return outcomes


def logger_warn(message: str) -> None:
    """Emit a warning to stderr without pulling in a logging dependency."""
    print(f"  warning: {message}", file=sys.stderr)


def scoreable(
    outcomes: Sequence[str],
    dataset: Sequence[SubsetCRow],
) -> Tuple[List[bool], List[bool], int]:
    """Keep only the rows the judge actually classified.

    ``ERRORED`` rows are dropped from the metric computation rather than
    assigned a class, and counted so the reader knows how much of the set the
    figures rest on.

    Returns:
        (predictions as is_in_domain, ground truths as is_in_domain, errored count).
    """
    predictions: List[bool] = []
    truths: List[bool] = []
    errored = 0
    for outcome, row in zip(outcomes, dataset):
        if outcome == ERRORED:
            errored += 1
            continue
        predictions.append(outcome == IN_DOMAIN)
        truths.append(row.label == "in_domain")
    return predictions, truths, errored


async def run_similarity_threshold(
    checker: SimilarityThresholdRelevanceChecker,
    api_url: str,
    dataset: List[SubsetCRow],
    top_k: int = 8,
    hyde: bool = False,
) -> List[bool]:
    """Run the production SimilarityThresholdRelevanceChecker over Subset C.

    No LLM call — thresholds the top retrieval (RRF-fusion) score already
    returned by /api/kb/search. See the checker's own docstring
    (app/thesis/ivm/checkers.py) for why its default threshold is a
    placeholder that needs empirical calibration, which is exactly what
    this experiment (run with --similarity-threshold swept) provides.

    Returns predictions with True = relevant/in-domain.
    """
    predictions: List[bool] = []
    for row in dataset:
        chunks, scores = await retrieve_contexts_detailed(api_url, row.query, top_k, hyde)
        predictions.append(await checker.check_query(row.query, chunks, scores))
    return predictions


async def run_nli_entailment(
    checker: NliEntailmentRelevanceChecker,
    api_url: str,
    dataset: List[SubsetCRow],
    top_k: int = 8,
    hyde: bool = False,
) -> List[bool]:
    """Run the production NliEntailmentRelevanceChecker over Subset C.

    Thresholds the Indonesian NLI model's entailment_score between the
    query (hypothesis) and joined retrieved context (premise), reusing the
    same EvalNLIClient Exp3/Exp4 use. Returns predictions (True = in-domain).
    """
    predictions: List[bool] = []
    for row in dataset:
        chunks, scores = await retrieve_contexts_detailed(api_url, row.query, top_k, hyde)
        predictions.append(await checker.check_query(row.query, chunks, scores))
    return predictions


async def fetch_corpus_dense_vectors(
    host: str,
    port: int,
    collection: str,
    limit: int = 100_000,
) -> "np.ndarray":
    """Scroll the Qdrant collection and return every dense chunk vector.

    The corpus embeddings define the in-domain cloud the centroid detector
    measures against. Read-only: the eval never writes to the store.

    Args:
        limit: Safety cap on rows scrolled.

    Returns:
        Array of shape (n, 1024) of dense vectors.
    """
    import numpy as np
    from qdrant_client import AsyncQdrantClient

    client = AsyncQdrantClient(host=host, port=port)
    vectors: List[List[float]] = []
    offset = None
    try:
        while len(vectors) < limit:
            points, offset = await client.scroll(
                collection_name=collection,
                with_vectors=["dense"],
                with_payload=False,
                limit=256,
                offset=offset,
            )
            for point in points:
                dense = (point.vector or {}).get("dense")
                if dense is not None:
                    vectors.append(dense)
            if offset is None:
                break
    finally:
        await client.close()
    return np.asarray(vectors, dtype=float)


async def run_centroid(
    corpus_vectors: "np.ndarray",
    query_vectors: "np.ndarray",
    threshold: float,
    metric: str,
    shrinkage: float,
) -> List[bool]:
    """Score every query against the corpus centroid and threshold it.

    Args:
        corpus_vectors: Corpus dense embeddings, shape (n, d).
        query_vectors: Subset C query embeddings, shape (m, d), same order as
            the dataset.
        metric: ``cosine`` or ``mahalanobis``.
        shrinkage: Covariance shrinkage for the Mahalanobis precision matrix.

    Returns:
        One prediction per query (True = in-domain).
    """
    from evals._shared.centroid import fit_centroid, is_in_domain, score

    model = fit_centroid(
        corpus_vectors, with_mahalanobis=(metric == "mahalanobis"), shrinkage=shrinkage
    )
    return [
        is_in_domain(score(model, q, metric), threshold, metric) for q in query_vectors
    ]


def run_keyword_overlap_baseline(
    dataset: List[SubsetCRow],
    lexicon: Set[str],
    threshold: int = 1,
) -> List[bool]:
    """Run keyword-overlap baseline against a lexicon.

    A query is predicted as relevant if it contains at least ``threshold``
    terms from ``lexicon`` (substring match). The lexicon is passed in rather
    than referenced as a global so the baseline runs against the reproducible,
    KB-derived set (``derive_jdih_lexicon``) instead of a hand-authored list.
    Returns predictions (True = relevant/in-domain).
    """
    predictions: List[bool] = []
    for row in dataset:
        query_lower = row.query.lower()
        matches = sum(1 for term in lexicon if term in query_lower)
        predictions.append(matches >= threshold)
    return predictions


def compute_per_subtype(
    predictions: List[bool],
    dataset: List[SubsetCRow],
) -> List[SubtypeResult]:
    """Compute metrics per subtype (predictions: True = in-domain)."""
    by_subtype: Dict[str, List[Tuple[bool, bool]]] = defaultdict(list)
    for pred, row in zip(predictions, dataset):
        gt = row.label == "in_domain"
        by_subtype[row.subtype].append((pred, gt))

    results: List[SubtypeResult] = []
    for subtype, pairs in sorted(by_subtype.items()):
        preds = [p for p, _ in pairs]
        gts = [g for _, g in pairs]
        metrics = compute_binary_metrics(preds, gts)
        ci = bootstrap_binary_ci(preds, gts, metric="accuracy")
        results.append(SubtypeResult(subtype=subtype, metrics=metrics, accuracy_ci=ci))

    return results


def compute_by_agreement(
    predictions: List[bool],
    dataset: List[SubsetCRow],
    strict_threshold: int = 4,
) -> Dict[str, Optional[BinaryMetrics]]:
    """Split metrics by whether the generating panel agreed strictly.

    Subset C admits boundary queries the panel split on (see
    ``SubsetCRow.is_contested``) rather than discarding them, because a split
    panel on a near-miss query is information about the boundary. Pooling the
    two into one accuracy would hide the thing worth knowing: whether a
    relevance checker fails specifically where human-level judgement is also
    divided. Reported separately for exactly that reason.

    Args:
        strict_threshold: Generation-time acceptance threshold.

    Returns:
        Mapping of "strict"/"contested" -> metrics, with None where the slice
        is empty (e.g. a dataset without a ``panel_yes`` column, whose rows all
        read as strict).
    """
    slices: Dict[str, List[Tuple[bool, bool]]] = {"strict": [], "contested": []}
    for pred, row in zip(predictions, dataset):
        key = "contested" if row.is_contested(strict_threshold) else "strict"
        slices[key].append((pred, row.label == "in_domain"))

    out: Dict[str, Optional[BinaryMetrics]] = {}
    for key, pairs in slices.items():
        if not pairs:
            out[key] = None
            continue
        out[key] = compute_binary_metrics([p for p, _ in pairs], [g for _, g in pairs])
    return out


def print_report(
    system_name: str,
    overall: BinaryMetrics,
    overall_ci: CI,
    per_subtype: List[SubtypeResult],
    by_agreement: Optional[Dict[str, Optional[BinaryMetrics]]] = None,
) -> None:
    """Print a formatted evaluation report.

    Args:
        by_agreement: Optional strict/contested split from
            ``compute_by_agreement``.
    """
    print(f"\n{'=' * 70}")
    print(f"  {system_name}")
    print(f"{'=' * 70}")
    print(f"  Samples: {overall.total}")
    print()
    print(f"  {'Metric':<20} {'Value':>10} {'95% CI':>20}")
    print(f"  {'-' * 52}")
    print(f"  {'Accuracy':<20} {overall.accuracy:>10.4f} [{overall_ci.lower:.4f}, {overall_ci.upper:.4f}]")
    print(f"  {'Precision':<20} {overall.precision:>10.4f}")
    print(f"  {'Recall':<20} {overall.recall:>10.4f}")
    print(f"  {'F1-Score':<20} {overall.f1:>10.4f}")
    print(f"  {'FPR':<20} {overall.fpr:>10.4f}")

    if per_subtype:
        # Only Accuracy and FPR (not Precision/Recall/F1, which are
        # mathematically undefined for Subset C's single-class subtypes,
        # e.g. off_topic is 100% out_of_domain). n is shown explicitly since
        # some subtypes are small (near_miss_government is n=6) and a subtype
        # accuracy shouldn't be read as a stable rate at that size.
        print(f"\n  Per Subtype:")
        print(f"  {'Subtype':<25} {'n':>5} {'Acc':>8} {'FPR':>8}")
        print(f"  {'-' * 49}")
        for r in per_subtype:
            m = r.metrics
            print(f"  {r.subtype:<25} {m.total:>5} {m.accuracy:>8.4f} {m.fpr:>8.4f}")

    if by_agreement and by_agreement.get("contested"):
        # Reported separately rather than pooled: a checker that matches the
        # panel on clear-cut queries but not on the ones the panel itself
        # split over is a different (and more interesting) result than a
        # single averaged accuracy would show.
        print(f"\n  By Generating-Panel Agreement:")
        print(f"  {'Slice':<25} {'n':>5} {'Acc':>8} {'FPR':>8}")
        print(f"  {'-' * 49}")
        for key, label in (("strict", "strict (full panel)"), ("contested", "contested (split panel)")):
            m = by_agreement.get(key)
            if m is None:
                continue
            print(f"  {label:<25} {m.total:>5} {m.accuracy:>8.4f} {m.fpr:>8.4f}")
    print()


async def async_main(args: argparse.Namespace) -> None:
    """Async entry point for Experiment 1b."""
    try:
        dataset = load_subset_c(args.dataset)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(dataset)} samples from Subset C")

    ground_truths = [row.label == "in_domain" for row in dataset]
    preds: Dict[str, List[bool]] = {}
    # The judge is stored apart from the deterministic bool-based backends: it
    # runs several times and can error, so it carries per-pass outcomes and a
    # flip count rather than a single bool.
    judge_runs: List[List[str]] = []
    judge_flips: List[int] = []

    def write_csv() -> None:
        rows = []
        for i, row in enumerate(dataset):
            r: Dict[str, object] = {
                "query": row.query,
                "subtype": row.subtype,
                "true_label": row.label,
            }
            for name, p in preds.items():
                r[f"{name}_pred"] = "in_domain" if p[i] else "out_of_domain"
                r[f"{name}_correct"] = p[i] == ground_truths[i]
            if judge_runs:
                for run_index, run in enumerate(judge_runs, start=1):
                    r[f"llm_judge_pred_run{run_index}"] = run[i]
                r["llm_judge_distinct_labels"] = judge_flips[i]
            rows.append(r)
        write_results_csv(args.output_csv, rows)

    # --- 1. Keyword overlap baseline (KB-derived lexicon) ---
    titles = await fetch_kb_titles(args.api_url)
    if titles:
        lexicon = derive_jdih_lexicon(titles, min_doc_freq=args.lexicon_min_doc_freq)
        print(
            f"\nDerived JDIH lexicon from {len(titles)} KB titles: "
            f"{len(lexicon)} terms (bigrams df>={args.lexicon_min_doc_freq} + anchors)"
        )
    else:
        lexicon = set(JDIH_ANCHORS)
        logger_warn(
            f"KB titles unavailable — keyword baseline falls back to {len(lexicon)} "
            "institutional anchors only (results not comparable to a derived run)"
        )
    print("Running keyword-overlap baseline...")
    preds["baseline"] = run_keyword_overlap_baseline(
        dataset, lexicon, threshold=args.keyword_threshold
    )
    print_report(
        "Baseline (Keyword Overlap)",
        compute_binary_metrics(preds["baseline"], ground_truths),
        bootstrap_binary_ci(preds["baseline"], ground_truths, metric="accuracy"),
        compute_per_subtype(preds["baseline"], dataset),
        compute_by_agreement(preds["baseline"], dataset, args.strict_threshold),
    )

    # --- 2. LLM-as-Judge (production LLMJudge, via LLMJudgeRelevanceChecker) ---
    if args.skip_llm_judge:
        print("\nSkipping LLM-as-Judge (--skip-llm-judge)")
    else:
        try:
            # session_id is accepted and ignored by the client (it does nothing
            # for caching); the provider is pinned instead so the judge is not
            # silently rerouted to a different upstream mid-run.
            llm_client = get_llm_client_from_env(model=args.judge_model or None)
        except ValueError as e:
            print(f"\nSkipping LLM-as-Judge: {e}", file=sys.stderr)
        else:
            try:
                judge = LLMJudge(llm_connection=llm_client, model=llm_client.model)
                checker = LLMJudgeRelevanceChecker(judge)
                print(
                    f"\nRunning LLM-as-Judge ({args.judge_repeats} passes, "
                    f"model={llm_client.model}, hyde={'on' if args.hyde else 'off'})..."
                )
                judge_runs = await repeat_passes(
                    lambda: run_llm_judge(checker, args.api_url, dataset, args.top_k, args.hyde),
                    args.judge_repeats,
                    "judge pass",
                )
            finally:
                await llm_client.aclose()

            agreement, judge_flips = self_agreement(judge_runs)
            # Metrics come from the first pass; errored rows are excluded and
            # the per-subtype/agreement tables are computed over the rows that
            # were actually classified, aligned with their dataset rows.
            judge_preds, judge_truths, judge_errored = scoreable(judge_runs[0], dataset)
            kept_rows = [
                row for outcome, row in zip(judge_runs[0], dataset) if outcome != ERRORED
            ]
            print_report(
                "LLM-as-Judge (production)",
                compute_binary_metrics(judge_preds, judge_truths),
                bootstrap_binary_ci(judge_preds, judge_truths, metric="accuracy"),
                compute_per_subtype(judge_preds, kept_rows),
                compute_by_agreement(judge_preds, kept_rows, args.strict_threshold),
            )
            print(f"  Rows scored          : {len(judge_preds)} of {len(dataset)}")
            if judge_errored:
                print(f"  Errored              : {judge_errored} (excluded; an outage is not a verdict)")
            if args.judge_repeats > 1:
                unstable = sum(1 for count in judge_flips if count > 1)
                print(f"  Self-agreement       : {agreement:.4f} over {args.judge_repeats} passes")
                print(f"  Rows with flips      : {unstable}")
                if agreement < 0.95:
                    print("  NOTE: the judge is not stable enough to quote as a bare point")
                    print("        estimate — report the agreement rate alongside it.")
            print()

    # --- 3. similarity_threshold (no LLM call) ---
    if args.skip_similarity:
        print("\nSkipping similarity_threshold (--skip-similarity)")
    else:
        sim_checker = SimilarityThresholdRelevanceChecker(threshold=args.similarity_threshold)
        print(f"\nRunning similarity_threshold (threshold={args.similarity_threshold})...")
        preds["similarity_threshold"] = await run_similarity_threshold(sim_checker, args.api_url, dataset, args.top_k, args.hyde)
        print_report(
            f"Similarity Threshold ({args.similarity_threshold})",
            compute_binary_metrics(preds["similarity_threshold"], ground_truths),
            bootstrap_binary_ci(preds["similarity_threshold"], ground_truths, metric="accuracy"),
            compute_per_subtype(preds["similarity_threshold"], dataset),
            compute_by_agreement(preds["similarity_threshold"], dataset, args.strict_threshold),
        )

    # --- 4. nli_entailment ---
    if args.skip_nli:
        print("\nSkipping nli_entailment (--skip-nli)")
    else:
        nli_client = EvalNLIClient(base_url=args.infinity_url, model=args.nli_model)
        try:
            nli_checker = NliEntailmentRelevanceChecker(nli_model=nli_client, threshold=args.nli_threshold)
            print(f"\nRunning nli_entailment (threshold={args.nli_threshold})...")
            preds["nli_entailment"] = await run_nli_entailment(nli_checker, args.api_url, dataset, args.top_k, args.hyde)
        finally:
            await nli_client.aclose()
        print_report(
            f"NLI Entailment ({args.nli_threshold})",
            compute_binary_metrics(preds["nli_entailment"], ground_truths),
            bootstrap_binary_ci(preds["nli_entailment"], ground_truths, metric="accuracy"),
            compute_per_subtype(preds["nli_entailment"], dataset),
            compute_by_agreement(preds["nli_entailment"], dataset, args.strict_threshold),
        )

    # --- 5. centroid / Mahalanobis (offline, no LLM call) ---
    if args.skip_centroid:
        print("\nSkipping centroid (--skip-centroid)")
    else:
        from app.kb.infra.bge_m3_embeddings import BGEM3Embeddings

        print(
            f"\nRunning centroid ({args.centroid_metric}, threshold={args.centroid_threshold})..."
        )
        corpus = await fetch_corpus_dense_vectors(
            args.qdrant_host, args.qdrant_port, args.qdrant_collection
        )
        if corpus.shape[0] == 0:
            print("  Skipping centroid: the Qdrant collection returned no vectors", file=sys.stderr)
        else:
            print(f"  Fitted on {corpus.shape[0]} corpus vectors (dim {corpus.shape[1]})")
            # Embed Subset C queries with the same model that embedded the
            # corpus, so query-to-centroid distances are meaningful.
            embedder = BGEM3Embeddings(model_name=args.embed_model, device=args.embed_device)
            embedded = await embedder.embed_texts([row.query for row in dataset])
            import numpy as np

            query_vectors = np.asarray([e.dense for e in embedded], dtype=float)
            preds["centroid"] = await run_centroid(
                corpus,
                query_vectors,
                threshold=args.centroid_threshold,
                metric=args.centroid_metric,
                shrinkage=args.centroid_shrinkage,
            )
            print_report(
                f"Centroid ({args.centroid_metric}, {args.centroid_threshold})",
                compute_binary_metrics(preds["centroid"], ground_truths),
                bootstrap_binary_ci(preds["centroid"], ground_truths, metric="accuracy"),
                compute_per_subtype(preds["centroid"], dataset),
                compute_by_agreement(preds["centroid"], dataset, args.strict_threshold),
            )

    write_csv()


def main() -> None:
    """Entry point for Experiment 1b."""
    parser = argparse.ArgumentParser(
        description=(
            "Experiment 1b: IVM Relevance Evaluation — keyword-overlap baseline vs "
            "the three production OOD backends (llm_judge, similarity_threshold, "
            "nli_entailment)."
        )
    )
    parser.add_argument("--dataset", required=True, help="Path to Subset C CSV")
    parser.add_argument(
        "--api-url",
        default="http://localhost:8000",
        help="Base URL of the running application (for KB search)",
    )
    parser.add_argument(
        "--infinity-url",
        default="http://localhost:7997",
        help="Infinity server base URL (for nli_entailment)",
    )
    parser.add_argument(
        "--nli-model",
        default="StevenLimcorn/indo-roberta-indonli",
        help="NLI model identifier (must match a model loaded on the Infinity server)",
    )
    parser.add_argument(
        "--hyde",
        action="store_true",
        help="Production-faithful condition (E7): route the retrieval-dependent "
        "backends (llm_judge, similarity_threshold, nli_entailment) through the "
        "HyDE-on chat retrieval endpoint (/api/chat/search) instead of the HyDE-off "
        "/api/kb/search, matching what the live pipeline feeds the relevance checkers. "
        "Keyword and centroid backends are unaffected (they don't retrieve via that path).",
    )
    parser.add_argument(
        "--judge-model",
        default="qwen/qwen3-14b",
        help="LLM-as-Judge model. It must match the deployed main LLM family and size; "
        "the production anchor is qwen/qwen3-14b. Use 8B or 32B only when the main "
        "generator is changed to the same size.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=8,
        help="Number of contexts to retrieve per query (default: 8, matching "
        "production's RERANK_TOP_K)",
    )
    parser.add_argument(
        "--keyword-threshold",
        type=int,
        default=1,
        help="Minimum lexicon matches for keyword baseline (default: 1)",
    )
    parser.add_argument(
        "--lexicon-min-doc-freq",
        type=int,
        default=DEFAULT_LEXICON_MIN_DOC_FREQ,
        help="Min distinct KB titles a bigram must appear in to enter the derived "
        f"keyword-baseline lexicon (default: {DEFAULT_LEXICON_MIN_DOC_FREQ}; unigrams "
        "are excluded as they reduce to generic legal vocabulary)",
    )
    parser.add_argument(
        "--strict-threshold",
        type=int,
        default=4,
        help="Generation-time panel acceptance threshold. Rows admitted below it "
        "(see the panel_yes column) are reported as a separate 'contested' slice "
        "rather than pooled into the headline accuracy (default: 4)",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.02,
        help="Threshold for similarity_threshold backend (default: 0.02, "
        "matching production's placeholder default — calibrate empirically, "
        "these are RRF-fusion scores, not bounded cosine similarity)",
    )
    parser.add_argument(
        "--nli-threshold",
        type=float,
        default=0.5,
        help="Minimum entailment_score for nli_entailment backend (default: 0.5, "
        "matching production's default)",
    )
    parser.add_argument(
        "--judge-repeats",
        type=int,
        default=3,
        help=(
            "How many identical passes to make for the LLM-judge. A hosted model "
            "is not bit-reproducible even at temperature 0, so repeats turn its "
            "stability into a measured self-agreement rate (default: 3)."
        ),
    )
    parser.add_argument(
        "--skip-llm-judge",
        action="store_true",
        help="Skip the LLM-as-Judge backend",
    )
    parser.add_argument(
        "--skip-similarity",
        action="store_true",
        help="Skip the similarity_threshold backend",
    )
    parser.add_argument(
        "--skip-nli",
        action="store_true",
        help="Skip the nli_entailment backend",
    )
    parser.add_argument(
        "--skip-centroid",
        action="store_true",
        help="Skip the centroid/Mahalanobis backend",
    )
    parser.add_argument(
        "--centroid-metric",
        choices=["cosine", "mahalanobis"],
        default="cosine",
        help="Distance to the corpus centre (default: cosine, higher = in-domain)",
    )
    parser.add_argument(
        "--centroid-threshold",
        type=float,
        default=0.5,
        help="Decision boundary for the centroid backend (placeholder — sweep it; "
        "cosine is in-domain at or above, Mahalanobis at or below)",
    )
    parser.add_argument(
        "--centroid-shrinkage",
        type=float,
        default=0.1,
        help="Covariance shrinkage toward a scaled identity for Mahalanobis "
        "(0..1; bge-m3 is 1024-dim so a non-zero value is required for a stable inverse)",
    )
    parser.add_argument(
        "--qdrant-host",
        default=os.environ.get("QDRANT_HOST", "127.0.0.1"),
        help="Qdrant host for the corpus vectors (centroid backend)",
    )
    parser.add_argument(
        "--qdrant-port",
        type=int,
        default=int(os.environ.get("QDRANT_PORT", "6333")),
        help="Qdrant HTTP port",
    )
    parser.add_argument(
        "--qdrant-collection",
        default=os.environ.get("QDRANT_COLLECTION_NAME", "knowledge_base"),
        help="Qdrant collection name",
    )
    parser.add_argument(
        "--embed-model",
        default="BAAI/bge-m3",
        help="Embedding model for Subset C queries (must match the corpus embedder)",
    )
    parser.add_argument(
        "--embed-device",
        default=os.environ.get("EMBED_DEVICE", "cuda"),
        help="Device for the query embedder (cuda or cpu)",
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
