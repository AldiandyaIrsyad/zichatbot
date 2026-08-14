"""Experiment 4 — End-to-End RAG Pipeline Evaluation.

Evaluates the full RAG pipeline (IVM → retrieval → generation → RAM) against
Subset A (RAG QA triplets). Compares against a no-guardrail baseline
(skip IVM + RAM, same retrieval + generation).

Metrics:
    - BERTScore F1 — answer quality vs ground truth
    - Faithfulness — proportion of supported sentences
    - Abstention Accuracy — correct refusal of out-of-domain queries

Usage:
    python -m evals.exp4_end_to_end.run \\
        --dataset evals/data/subset_a.csv \\
        --api-url http://localhost:8000
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import sys
import time
from urllib.parse import urlencode
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import httpx

from evals._shared.clients import EvalNLIClient
from evals._shared.csv_export import write_results_csv
from evals._shared.dataset import load_subset_a, SubsetARow
from evals._shared.metrics import (
    CI,
    abstention_accuracy,
    bert_score_f1,
    bootstrap_ci,
    faithfulness,
    wilson_interval,
)
from app.guardrails.ram.text_utils import split_sentences

DEFAULT_OUTPUT_CSV = "evals/data/results/exp4_end_to_end.csv"


# Citation format: *(STATUS: SCORE; SOURCE; Page N; DocID:ID; Evidence:"snippet")*
# The DocID and Evidence segments are optional and must be accounted for
# even when unused here, otherwise the pattern fails to reach the closing
# ")" and silently matches nothing for any citation that includes them.
CITATION_PATTERN = re.compile(
    r"\*?\s*\("
    r"(?P<status>Supported|Contradiction|Neutral)"
    r":\s*(?P<score>[\d.]+)"
    r";\s*(?P<source>[^;]+)"
    r"(?:;\s*Page\s+(?P<page>\d+))?"
    r"(?:;\s*DocID:(?P<doc_id>[\w-]+))?"
    r'(?:;\s*Evidence:"(?P<evidence>[^"]*)")?'
    r"\)\s*\*?"
)

# Detects an English chain-of-thought preamble at the start of a response
# that should have been Indonesian-only per CHAT_SYSTEM_PROMPT — a known
# symptom of LLMConnection.stream_chat's reasoning-delta fallback (the same
# mechanism that affects the relevance judge). This is a DETECTOR, not a
# text-repair tool — there is no structural signal in the NDJSON stream that
# separates reasoning from real content (both are emitted as the same "chunk"
# event type), so guessing where the real answer starts risks corrupting
# responses that don't match this pattern. Flagged rows are reported, not
# silently stripped or dropped, unless --exclude-contaminated is passed.
REASONING_PREAMBLE_PATTERN = re.compile(
    r"^\s*(okay|alright|let(?:'|’)s see|let me|the user is ask|i need to|"
    r"looking at|i should|we need to|hmm)\b",
    re.IGNORECASE,
)


def detect_reasoning_contamination(text: str) -> bool:
    """Flag a response likely to open with an un-stripped English CoT preamble.

    Returns True if the response opens with a recognized reasoning-preamble
    phrase (see REASONING_PREAMBLE_PATTERN).
    """
    return bool(text) and bool(REASONING_PREAMBLE_PATTERN.match(text.strip()))


def strip_citation_markup(text: str) -> str:
    """Remove RAM citation markers from a response before scoring it.

    BERTScore should measure similarity of the substantive answer to the
    reference, not of machine-generated markup — citation markup is a large
    fraction of full-pipeline response characters while the baseline condition
    has 0%, so leaving it in would make the two conditions' BERTScore not
    comparable.

    Returns:
        Text with citation markers removed, whitespace-collapsed.
    """
    cleaned = CITATION_PATTERN.sub(" ", text)
    return re.sub(r"\s+", " ", cleaned).strip()


@dataclass
class PipelineResult:
    """Result of running the full pipeline on a single query.

    Attributes:
        question: The input question.
        response: Full system response text.
        citations: List of parsed citation tuples (status, score, source, page).
        abstained: Whether the system genuinely refused/warned (an IVM
            block or explicit pipeline rejection — the "error" NDJSON
            event). Distinct from ``errored``: a request timeout or non-200
            response is an infrastructure failure, not a correct out-of-domain
            refusal.
        errored: Whether the request itself failed (timeout, non-200) —
            no verdict from the pipeline either way, so this should not be
            counted as a correct (or incorrect) abstention.
        reasoning_contaminated: Whether the response was flagged by
            detect_reasoning_contamination — reported, not acted on
            automatically; see --exclude-contaminated.
        category: Question category.
        ground_truth: Ground-truth answer.
        retrieved_context: Context chunks the pipeline retrieved for this
            query (from the stream's "context" event). Captured regardless
            of whether guardrails ran, so faithfulness can be computed
            post-hoc for pipeline runs where RAM didn't run live — see
            compute_posthoc_faithfulness.
        latency_s: Wall-clock seconds for the streaming request. The
            guardrails cost real time (an IVM judge call plus one NLI call per
            sentence), so the quality numbers are only half the trade-off
            without this.
    """

    question: str
    response: str
    citations: List[Tuple[str, float, str, Optional[int]]] = field(default_factory=list)
    abstained: bool = False
    errored: bool = False
    reasoning_contaminated: bool = False
    category: str = ""
    ground_truth: str = ""
    retrieved_context: str = ""
    latency_s: float = 0.0


@dataclass
class E2EMetrics:
    """Aggregated end-to-end metrics.

    Attributes:
        bertscore_f1: BERTScore F1 (answer quality), computed on
            citation-stripped candidates.
        bertscore_f1_ci: Bootstrap CI for BERTScore F1.
        faithfulness_score: PRIMARY faithfulness metric — post-hoc NLI
            check over every sentence, computed the same way for both
            pipeline conditions (see compute_posthoc_faithfulness).
            Populated by the caller (async_main), not by compute_e2e_metrics
            itself, since it requires an NLI client shared across both
            conditions.
        faithfulness_ci: Bootstrap CI for faithfulness_score.
        citation_faithfulness_score: SECONDARY/diagnostic faithfulness —
            the citation-marker-based measure. Only meaningful for the
            full-pipeline condition (RAM only emits citations there); always
            0.0 for the no-guardrail baseline since it never produces
            citations. Kept for comparison, not as the headline number.
        citation_faithfulness_ci: Bootstrap CI for citation_faithfulness_score.
        abstention_acc: Abstention accuracy for out-of-domain queries —
            counts only genuine refusals (PipelineResult.abstained), not
            infra errors.
        abstention_ci: Bootstrap CI for abstention accuracy.
        false_refusal_rate: Fraction of IN-DOMAIN queries the system
            wrongly refused — the counterpart metric so "refuse everything"
            can't score a perfect Abstention Accuracy without visibly costing
            something here.
        false_refusal_ci: Wilson CI for false_refusal_rate.
        error_rate: Fraction of ALL queries where the request itself
            failed (timeout/non-200) rather than returning any verdict.
        contamination_rate: Fraction of in-domain scored responses flagged
            by detect_reasoning_contamination — reported for transparency;
            see --exclude-contaminated.
        latency_mean_s: Mean wall-clock seconds per query, over non-errored
            queries. The cost side of the latency trade-off: guardrails buy
            their quality/safety gains with an IVM judge call plus one NLI
            call per generated sentence.
        latency_p50_s: Median latency — reported alongside the mean because a
            few slow queries skew the mean badly at this n.
        latency_p95_s: 95th-percentile latency.
        total_queries: Total number of queries evaluated.
        out_of_domain_count: Number of out-of-domain queries.
    """

    bertscore_f1: float = 0.0
    bertscore_f1_ci: CI = field(default_factory=lambda: CI(0.0, 0.0, 0.0))
    faithfulness_score: float = 0.0
    faithfulness_ci: CI = field(default_factory=lambda: CI(0.0, 0.0, 0.0))
    citation_faithfulness_score: float = 0.0
    citation_faithfulness_ci: CI = field(default_factory=lambda: CI(0.0, 0.0, 0.0))
    abstention_acc: float = 0.0
    abstention_ci: CI = field(default_factory=lambda: CI(0.0, 0.0, 0.0))
    false_refusal_rate: float = 0.0
    false_refusal_ci: CI = field(default_factory=lambda: CI(0.0, 0.0, 0.0))
    error_rate: float = 0.0
    contamination_rate: float = 0.0
    latency_mean_s: float = 0.0
    latency_p50_s: float = 0.0
    latency_p95_s: float = 0.0
    total_queries: int = 0
    out_of_domain_count: int = 0


def parse_citations(response: str) -> List[Tuple[str, float, str, Optional[int]]]:
    """Parse citation markers from a system response.

    Extracts all citations matching the format
    ``*(STATUS: SCORE; SOURCE; Page N)*``.

    Returns:
        List of (status, score, source, page) tuples.
    """
    citations: List[Tuple[str, float, str, Optional[int]]] = []
    for match in CITATION_PATTERN.finditer(response):
        status = match.group("status")
        try:
            score = float(match.group("score"))
        except ValueError:
            score = 0.0
        source = match.group("source").strip()
        page_str = match.group("page")
        page = int(page_str) if page_str else None
        citations.append((status, score, source, page))
    return citations


def extract_sentence_labels(citations: List[Tuple[str, float, str, Optional[int]]]) -> List[str]:
    """Map citation statuses to faithfulness labels."""
    label_map = {
        "Supported": "supported",
        "Contradiction": "not_supported",
        "Neutral": "partially_supported",
    }
    return [label_map.get(c[0], "partially_supported") for c in citations]


async def run_pipeline(
    api_url: str,
    row: SubsetARow,
    session_id: Optional[str] = None,
    skip_guardrails: bool = False,
    skip_ivm: Optional[bool] = None,
    skip_ram: Optional[bool] = None,
) -> PipelineResult:
    """Run the full RAG pipeline on a single query via the chat API.

    Args:
        skip_guardrails: If True, bypass IVM + RAM together (the original
            baseline arm).
        skip_ivm: If set, bypass only the IVM safety/relevance checks.
        skip_ram: If set, bypass only the RAM per-sentence assessment.
            Together with skip_ivm this makes the guardrail conditions a
            4-cell ablation (none / IVM-only / RAM-only / both) rather than a
            single on-off block, which is what lets an effect be attributed
            to one of the two modules.

    Returns:
        PipelineResult with response, parsed citations, and elapsed latency.
    """
    async with httpx.AsyncClient(base_url=api_url, timeout=120.0) as client:
        # Create a session
        if session_id is None:
            try:
                resp = await client.post("/api/chat/sessions")
                if resp.status_code in (200, 201):
                    session_id = resp.json().get("id")
            except Exception:
                session_id = None

        # Send the message to the streaming endpoint
        stream_url = f"/api/chat/sessions/{session_id}/stream"
        params: Dict[str, str] = {}
        if skip_guardrails:
            params["skip_guardrails"] = "true"
        if skip_ivm is not None:
            params["skip_ivm"] = str(skip_ivm).lower()
        if skip_ram is not None:
            params["skip_ram"] = str(skip_ram).lower()
        if params:
            stream_url += "?" + urlencode(params)

        started = time.perf_counter()
        try:
            resp = await client.post(
                stream_url,
                json={"message": row.question},
                timeout=120.0,
            )
            if resp.status_code != 200:
                # Infra/HTTP failure, not a pipeline verdict — errored, not
                # abstained: crediting this as a correct out-of-domain refusal
                # would inflate Abstention Accuracy on infrastructure noise.
                return PipelineResult(
                    question=row.question,
                    response="",
                    errored=True,
                    category=row.category,
                    ground_truth=row.ground_truth_answer,
                    latency_s=time.perf_counter() - started,
                )

            # The response is an NDJSON stream
            response_text = ""
            retrieved_context = ""
            content_type = resp.headers.get("content-type", "")
            if "application/x-ndjson" in content_type or "text/plain" in content_type:
                for line in resp.text.strip().split("\n"):
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                        chunk_type = chunk.get("type")
                        if chunk_type == "chunk":
                            response_text += chunk.get("content", "")
                        elif chunk_type == "context":
                            retrieved_context = chunk.get("content", "")
                        elif chunk_type == "error":
                            content = chunk.get("content", "")
                            # ChatService uses the same event type for policy
                            # refusals and caught infrastructure exceptions. The
                            # generic exception message is not an abstention.
                            infra_error = content.strip().lower() == "an unexpected error occurred."
                            return PipelineResult(
                                question=row.question,
                                response=content,
                                abstained=not infra_error,
                                errored=infra_error,
                                category=row.category,
                                ground_truth=row.ground_truth_answer,
                                latency_s=time.perf_counter() - started,
                            )
                        elif chunk_type == "done":
                            break
                    except json.JSONDecodeError:
                        response_text += line
            else:
                data = resp.json()
                response_text = data.get("content", data.get("response", ""))
        except httpx.TimeoutException:
            # Infra failure, not a pipeline verdict — see the non-200 case
            # above.
            return PipelineResult(
                question=row.question,
                response="",
                errored=True,
                category=row.category,
                ground_truth=row.ground_truth_answer,
                latency_s=time.perf_counter() - started,
            )

    citations = parse_citations(response_text)
    return PipelineResult(
        question=row.question,
        response=response_text,
        citations=citations,
        abstained=False,
        reasoning_contaminated=detect_reasoning_contamination(response_text),
        category=row.category,
        ground_truth=row.ground_truth_answer,
        retrieved_context=retrieved_context,
        latency_s=time.perf_counter() - started,
    )


async def run_no_guardrail_pipeline(
    api_url: str,
    row: SubsetARow,
    session_id: Optional[str] = None,
) -> PipelineResult:
    """Run the pipeline without guardrails (baseline).

    Calls the chat endpoint with ``?skip_guardrails=true`` to bypass IVM
    (safety + relevance) and RAM (per-sentence assessment). Retrieval still
    runs so the LLM has context.
    """
    return await run_pipeline(api_url, row, session_id, skip_guardrails=True)


def compute_e2e_metrics(
    results: List[PipelineResult],
    exclude_contaminated: bool = False,
) -> Tuple[E2EMetrics, Dict[str, Dict[str, Any]]]:
    """Compute end-to-end metrics from pipeline results.

    NOTE: this does NOT set the primary faithfulness_score/faithfulness_ci
    — those come from compute_posthoc_faithfulness, run uniformly over
    both pipeline conditions by the caller. This function only populates the
    citation-based citation_faithfulness_score as a secondary/diagnostic
    number.

    Args:
        exclude_contaminated: If True, drop rows flagged by
            detect_reasoning_contamination from the BERTScore
            candidate/reference set. contamination_rate is always computed
            regardless, so the scope of contamination is visible either way.

    Returns:
        Tuple of (E2EMetrics, per_row_data) — per_row_data maps question ->
        {"bertscore_f1": float | None, "citation_faithfulness_score":
        float | None} for CSV export; a row's value is None where it
        wasn't computed (e.g. empty response for BERTScore, out-of-domain
        for both).
    """
    metrics = E2EMetrics(total_queries=len(results))
    per_row: Dict[str, Dict[str, Any]] = {
        r.question: {"bertscore_f1": None, "citation_faithfulness_score": None} for r in results
    }

    # Separate in-domain and out-of-domain
    in_domain: List[PipelineResult] = []
    out_of_domain: List[PipelineResult] = []
    for r in results:
        if r.category == "out-of-domain" or r.category == "out_of_domain":
            out_of_domain.append(r)
        else:
            in_domain.append(r)
    metrics.out_of_domain_count = len(out_of_domain)

    # --- Error rate (all rows) ---
    if results:
        metrics.error_rate = sum(1 for r in results if r.errored) / len(results)

    # --- Latency (non-errored rows: a timeout's 120s says nothing about how
    # long the pipeline takes when it works) ---
    latencies = sorted(r.latency_s for r in results if not r.errored and r.latency_s > 0)
    if latencies:
        metrics.latency_mean_s = sum(latencies) / len(latencies)
        metrics.latency_p50_s = latencies[len(latencies) // 2]
        metrics.latency_p95_s = latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))]

    # --- BERTScore F1 (in-domain only, citation markup stripped) ---
    if in_domain:
        in_domain_with_response = [r for r in in_domain if r.response]
        metrics.contamination_rate = (
            sum(1 for r in in_domain_with_response if r.reasoning_contaminated) / len(in_domain_with_response)
            if in_domain_with_response
            else 0.0
        )
        scoring_rows = (
            [r for r in in_domain_with_response if not r.reasoning_contaminated]
            if exclude_contaminated
            else in_domain_with_response
        )
        candidates = [strip_citation_markup(r.response) for r in scoring_rows]
        references = [r.ground_truth for r in scoring_rows]
        if candidates and references:
            # bert_score_f1 already returns per-example scores (f1_per_example)
            # alongside the corpus mean — bootstrap-resample that array
            # directly instead of re-invoking the model per resample (each call
            # would reload IndoBERT and rerun inference over the resampled
            # batch; this is the same near-instant pattern Faithfulness and
            # exp3's Cohen's Kappa use).
            _, _, f1, f1_per_example = bert_score_f1(candidates, references)
            metrics.bertscore_f1 = f1
            metrics.bertscore_f1_ci = bootstrap_ci(f1_per_example, statistic="mean")
            for r, score in zip(scoring_rows, f1_per_example):
                per_row[r.question]["bertscore_f1"] = score

    # --- Citation-based Faithfulness (in-domain only) — SECONDARY/diagnostic,
    # only meaningful for the full-pipeline condition (RAM emits citations
    # there); see the primary post-hoc measure the caller computes instead. ---
    if in_domain:
        faithfulness_scores: List[float] = []
        for r in in_domain:
            if not r.citations:
                # No citations → assume all sentences are "no_source_needed"
                sentences = split_sentences(r.response)
                labels = ["no_source_needed"] * len(sentences)
            else:
                labels = extract_sentence_labels(r.citations)
            score = faithfulness(labels)
            faithfulness_scores.append(score)
            per_row[r.question]["citation_faithfulness_score"] = score
        if faithfulness_scores:
            metrics.citation_faithfulness_score = sum(faithfulness_scores) / len(faithfulness_scores)
            metrics.citation_faithfulness_ci = bootstrap_ci(faithfulness_scores, statistic="mean")

    # --- Abstention Accuracy (out-of-domain only; errored rows are neither
    # a correct nor incorrect abstention — see PipelineResult.errored) ---
    if out_of_domain:
        abstained_flags = [r.abstained for r in out_of_domain]
        metrics.abstention_acc = abstention_accuracy(abstained_flags, len(out_of_domain))
        successes = sum(1 for a in abstained_flags if a)
        metrics.abstention_ci = wilson_interval(successes, len(out_of_domain))

    # --- False Refusal Rate (in-domain only) — counterpart to Abstention
    # Accuracy: a system that refuses everything scores perfectly on
    # out-of-domain abstention without this metric also visibly costing it. ---
    if in_domain:
        false_refusals = sum(1 for r in in_domain if r.abstained)
        metrics.false_refusal_rate = false_refusals / len(in_domain)
        metrics.false_refusal_ci = wilson_interval(false_refusals, len(in_domain))

    return metrics, per_row


async def compute_posthoc_faithfulness(
    results: List[PipelineResult],
    nli_client: EvalNLIClient,
) -> Tuple[float, CI, Dict[str, float]]:
    """Compute the PRIMARY Faithfulness metric via post-hoc NLI checking.

    Called for BOTH pipeline conditions and used as the headline
    faithfulness_score: the citation-based measure only scores the ~17% of
    full-pipeline sentences that carry a citation marker, while this checks
    every sentence against the retrieved context directly — the same method
    for both conditions, over the same denominator, which is what makes the
    two faithfulness numbers an actual controlled comparison instead of two
    different measurement tools. Also benefits from the premise-selection in
    EvalNLIClient (see _shared/clients.py's _select_relevant_chunk, which
    selects a relevant chunk rather than blindly truncating to the first
    ~10% of a long context).

    No new chat/LLM generation calls — just local Infinity NLI
    classification against each result's already-captured (response,
    retrieved_context) pair.

    Returns:
        Tuple of (mean faithfulness score, bootstrap CI, per-question score
        map — for CSV export).
    """
    faithfulness_scores: List[float] = []
    per_question: Dict[str, float] = {}
    for r in results:
        if not r.response or not r.retrieved_context:
            continue
        sentences = split_sentences(r.response)
        if not sentences:
            continue
        labels = []
        for sentence in sentences:
            nli_result = await nli_client.check(premise=r.retrieved_context, hypothesis=sentence)
            labels.append(nli_result.label)
        score = faithfulness(labels)
        faithfulness_scores.append(score)
        per_question[r.question] = score

    if not faithfulness_scores:
        return 0.0, CI(point=0.0, lower=0.0, upper=0.0), per_question

    mean_score = sum(faithfulness_scores) / len(faithfulness_scores)
    ci = bootstrap_ci(faithfulness_scores, statistic="mean")
    return mean_score, ci, per_question


def compute_per_category(
    results: List[PipelineResult],
    per_row: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Break the in-domain scores down by question category.

    Subset A labels each question factual / procedural / multi-hop, so this
    shows whether e.g. multi-hop questions do worse rather than reporting only
    the aggregate. Regroups the already-computed per-row scores rather than
    recomputing anything.

    Args:
        results: Pipeline results for one condition.
        per_row: The per-question score map from compute_e2e_metrics plus
            add_posthoc_faithfulness (bertscore_f1, faithfulness_score).

    Returns:
        Mapping of category -> {n, bertscore_f1, faithfulness_score,
        latency_mean_s}, where a score is None if no row in that category had
        one.
    """
    by_category: Dict[str, List[PipelineResult]] = defaultdict(list)
    for r in results:
        if r.category in ("out-of-domain", "out_of_domain"):
            continue
        by_category[r.category].append(r)

    def _mean(values: List[float]) -> Optional[float]:
        return sum(values) / len(values) if values else None

    out: Dict[str, Dict[str, Any]] = {}
    for category, rows in sorted(by_category.items()):
        bert = [
            v for v in (per_row.get(r.question, {}).get("bertscore_f1") for r in rows)
            if v is not None
        ]
        faith = [
            v for v in (per_row.get(r.question, {}).get("faithfulness_score") for r in rows)
            if v is not None
        ]
        latencies = [r.latency_s for r in rows if not r.errored and r.latency_s > 0]
        out[category] = {
            "n": len(rows),
            "bertscore_f1": _mean(bert),
            "faithfulness_score": _mean(faith),
            "latency_mean_s": _mean(latencies),
        }
    return out


def print_report(
    system_name: str,
    metrics: E2EMetrics,
    per_category: Optional[Dict[str, Dict[str, Any]]] = None,
) -> None:
    """Print a formatted end-to-end evaluation report."""
    print(f"\n{'=' * 70}")
    print(f"  {system_name}")
    print(f"{'=' * 70}")
    print(f"  Total Queries:      {metrics.total_queries}")
    print(f"  In-Domain:          {metrics.total_queries - metrics.out_of_domain_count}")
    print(f"  Out-of-Domain:      {metrics.out_of_domain_count}")
    print()
    print(f"  {'Metric':<25} {'Value':>10} {'95% CI':>25}")
    print(f"  {'-' * 62}")
    print(f"  {'BERTScore F1':<25} {metrics.bertscore_f1:>10.4f} "
          f"[{metrics.bertscore_f1_ci.lower:.4f}, {metrics.bertscore_f1_ci.upper:.4f}]")
    print(f"  {'Faithfulness':<25} {metrics.faithfulness_score:>10.4f} "
          f"[{metrics.faithfulness_ci.lower:.4f}, {metrics.faithfulness_ci.upper:.4f}]"
          f"  (primary, post-hoc NLI)")
    print(f"  {'  Faithfulness (cit.)':<25} {metrics.citation_faithfulness_score:>10.4f} "
          f"[{metrics.citation_faithfulness_ci.lower:.4f}, {metrics.citation_faithfulness_ci.upper:.4f}]"
          f"  (secondary, citation markers only)")
    print(f"  {'Abstention Accuracy':<25} {metrics.abstention_acc:>10.4f} "
          f"[{metrics.abstention_ci.lower:.4f}, {metrics.abstention_ci.upper:.4f}]")
    print(f"  {'False Refusal Rate':<25} {metrics.false_refusal_rate:>10.4f} "
          f"[{metrics.false_refusal_ci.lower:.4f}, {metrics.false_refusal_ci.upper:.4f}]"
          f"  (in-domain, wrongly refused)")
    print()
    print(f"  Latency (mean/p50/p95): {metrics.latency_mean_s:.2f}s / "
          f"{metrics.latency_p50_s:.2f}s / {metrics.latency_p95_s:.2f}s  (non-errored queries)")
    print(f"  Error Rate:          {metrics.error_rate:.4f}  (request failed — timeout/non-200, no verdict)")
    print(f"  Contamination Rate:  {metrics.contamination_rate:.4f}  (responses flagged by "
          f"detect_reasoning_contamination — see --exclude-contaminated)")

    if per_category:
        print()
        print(f"  Per Category (in-domain):")
        print(f"  {'Category':<20} {'N':>5} {'BERTScore':>12} {'Faithfulness':>14} {'Latency':>10}")
        print(f"  {'-' * 64}")
        for category, stats in per_category.items():
            bert = f"{stats['bertscore_f1']:.4f}" if stats["bertscore_f1"] is not None else "—"
            faith = f"{stats['faithfulness_score']:.4f}" if stats["faithfulness_score"] is not None else "—"
            lat = f"{stats['latency_mean_s']:.2f}s" if stats["latency_mean_s"] is not None else "—"
            print(f"  {category:<20} {stats['n']:>5} {bert:>12} {faith:>14} {lat:>10}")
    print()


def stratified_sample(
    dataset: List[SubsetARow], limit: int, seed: int
) -> List[SubsetARow]:
    """Take ~``limit`` rows spread as evenly as possible across categories.

    The limited-budget run (``--limit``) trades the full sweep for a fixed
    budget, but a blind head-N sample would skew toward whichever category
    leads the CSV and could drop a category (e.g. out-of-domain) entirely —
    which would silently disable the abstention metric. This allocates the
    budget round-robin over the category labels so each keeps representation
    in both the quantitative anchor and the per-category breakdown.
    Deterministic in (dataset order, ``seed``): rows are shuffled within each
    category so the
    pick isn't biased by CSV order.

    Args:
        dataset: Rows to sample from.
        limit: Target sample size. ``<= 0`` or ``>= len(dataset)`` returns the
            full dataset unchanged.
        seed: RNG seed for the within-category shuffle.

    Returns:
        The sampled rows, grouped by category (categories in sorted order).
    """
    if limit <= 0 or limit >= len(dataset):
        return list(dataset)

    by_cat: Dict[str, List[SubsetARow]] = defaultdict(list)
    for row in dataset:
        by_cat[row.category].append(row)

    rng = random.Random(seed)
    cats = sorted(by_cat)
    for cat in cats:
        rng.shuffle(by_cat[cat])

    selected: List[SubsetARow] = []
    cursors = {cat: 0 for cat in cats}
    while len(selected) < limit:
        progressed = False
        for cat in cats:
            if len(selected) >= limit:
                break
            idx = cursors[cat]
            if idx < len(by_cat[cat]):
                selected.append(by_cat[cat][idx])
                cursors[cat] = idx + 1
                progressed = True
        if not progressed:
            break  # every category exhausted before reaching the limit

    selected.sort(key=lambda r: cats.index(r.category))
    return selected


async def async_main(args: argparse.Namespace) -> None:
    """Async entry point for Experiment 4."""
    try:
        dataset = load_subset_a(args.dataset)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(dataset)} samples from Subset A")

    if args.limit and args.limit > 0:
        dataset = stratified_sample(dataset, args.limit, args.seed)
        n_cats = len({r.category for r in dataset})
        print(
            f"Stratified sample: {len(dataset)} rows across "
            f"{n_cats} categories (limit={args.limit}, seed={args.seed})"
        )

    csv_rows: List[Dict[str, Any]] = []

    def add_csv_rows(results: List[PipelineResult], condition: str, per_row: Dict[str, Dict[str, Any]]) -> None:
        for r in results:
            row_data = per_row.get(r.question, {})
            csv_rows.append({
                "question": r.question,
                "category": r.category,
                "condition": condition,
                "response": r.response,
                "abstained": r.abstained,
                "errored": r.errored,
                "reasoning_contaminated": r.reasoning_contaminated,
                "latency_s": round(r.latency_s, 3),
                "faithfulness_score": row_data.get("faithfulness_score"),
                "citation_faithfulness_score": row_data.get("citation_faithfulness_score"),
                "bertscore_f1_contribution": row_data.get("bertscore_f1"),
            })

    async def add_posthoc_faithfulness(
        results: List[PipelineResult],
        metrics: E2EMetrics,
        per_row: Dict[str, Dict[str, Any]],
        nli_client: EvalNLIClient,
    ) -> None:
        """Compute the PRIMARY faithfulness metric for one condition.

        Run uniformly for both conditions — the full-pipeline's own
        citation-based faithfulness only covers the ~17% of sentences that
        carry a citation marker, so it isn't a fair like-for-like comparison
        against the baseline's faithfulness on its own (see
        compute_posthoc_faithfulness).
        """
        in_domain = [r for r in results if r.category not in ("out-of-domain", "out_of_domain")]
        rows_to_score = (
            [r for r in in_domain if not r.reasoning_contaminated] if args.exclude_contaminated else in_domain
        )
        score, ci, per_question = await compute_posthoc_faithfulness(rows_to_score, nli_client)
        metrics.faithfulness_score = score
        metrics.faithfulness_ci = ci
        for question, s in per_question.items():
            per_row.setdefault(question, {})["faithfulness_score"] = s

    # Guardrail conditions. "full" and "baseline" keep their original names and
    # meaning so runs stay comparable with the committed results; "ivm_only"
    # and "ram_only" are the two cells that let an effect be attributed to one
    # module instead of to the pair.
    ALL_CONDITIONS: Dict[str, Dict[str, bool]] = {
        "full": {"skip_ivm": False, "skip_ram": False},
        "ivm_only": {"skip_ivm": False, "skip_ram": True},
        "ram_only": {"skip_ivm": True, "skip_ram": False},
        "baseline": {"skip_ivm": True, "skip_ram": True},
    }
    CONDITION_LABELS = {
        "full": "Full Pipeline (IVM + RAM)",
        "ivm_only": "IVM only (RAM disabled)",
        "ram_only": "RAM only (IVM disabled)",
        "baseline": "Baseline (no guardrails)",
    }

    if args.conditions == "all":
        selected = list(ALL_CONDITIONS)
    elif args.conditions == "ablation":
        selected = ["full", "ivm_only", "ram_only", "baseline"]
    else:
        selected = [c.strip() for c in args.conditions.split(",") if c.strip()]

    # Back-compat with the original flags.
    if args.baseline_only:
        selected = ["baseline"]
    elif args.no_baseline:
        selected = [c for c in selected if c != "baseline"]

    unknown = [c for c in selected if c not in ALL_CONDITIONS]
    if unknown:
        print(f"ERROR: unknown condition(s): {', '.join(unknown)}", file=sys.stderr)
        sys.exit(1)

    nli_client = EvalNLIClient(args.infinity_url, args.nli_model)
    try:
        for condition in selected:
            switches = ALL_CONDITIONS[condition]
            label = CONDITION_LABELS[condition]
            print(f"\nRunning condition: {label}...")

            results: List[PipelineResult] = []
            for i, row in enumerate(dataset, 1):
                result = await run_pipeline(args.api_url, row, **switches)
                results.append(result)
                if i % 10 == 0:
                    print(f"  Processed {i}/{len(dataset)} queries...")

            metrics, per_row = compute_e2e_metrics(
                results, exclude_contaminated=args.exclude_contaminated
            )
            print(f"Computing {condition} Faithfulness post-hoc (primary metric)...")
            await add_posthoc_faithfulness(results, metrics, per_row, nli_client)
            per_category = compute_per_category(results, per_row)
            print_report(label, metrics, per_category)
            add_csv_rows(results, condition, per_row)
    finally:
        await nli_client.aclose()

    write_results_csv(args.output_csv, csv_rows)


def main() -> None:
    """Entry point for Experiment 4."""
    parser = argparse.ArgumentParser(
        description="Experiment 4: End-to-End RAG Pipeline Evaluation (with guardrails vs no-guardrail baseline)."
    )
    parser.add_argument("--dataset", required=True, help="Path to Subset A CSV")
    parser.add_argument(
        "--api-url",
        default="http://localhost:8000",
        help="Base URL of the running application",
    )
    parser.add_argument(
        "--infinity-url",
        default="http://localhost:7997",
        help="Infinity server base URL (for the post-hoc Faithfulness NLI check, both conditions)",
    )
    parser.add_argument(
        "--nli-model",
        default="StevenLimcorn/indo-roberta-indonli",
        help="NLI model identifier (must match a model loaded on the Infinity server)",
    )
    parser.add_argument(
        "--exclude-contaminated",
        action="store_true",
        help="Exclude responses flagged by detect_reasoning_contamination (an "
        "English chain-of-thought preamble that should have been Indonesian) from "
        "BERTScore and Faithfulness aggregates. Off by default so the default run "
        "doesn't silently change which rows are scored; contamination_rate is "
        "always reported regardless of this flag.",
    )
    parser.add_argument(
        "--conditions",
        default="full,baseline",
        help="Guardrail conditions to run: 'ablation' (or 'all') for the 4-cell "
        "full/ivm_only/ram_only/baseline ablation, or a comma-separated subset of "
        "those names. Defaults to 'full,baseline' — the original two-arm comparison — "
        "so an unflagged run stays comparable with the committed results.",
    )
    parser.add_argument(
        "--no-baseline",
        action="store_true",
        help="Skip the no-guardrail baseline",
    )
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="Skip the full (with-guardrails) pipeline and only run the no-guardrail baseline — "
        "for re-running just the baseline after a fix that only affects its measurement "
        "(e.g. Faithfulness), without re-paying for the already-valid guardrail pipeline's LLM calls.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="If >0, run on a stratified sample of ~N rows spread evenly across "
        "question categories (a small quantitative anchor rather than the full "
        "sweep). 0 (default) uses the whole dataset.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for the --limit stratified sample (deterministic).",
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
