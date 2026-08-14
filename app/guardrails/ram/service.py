"""RAM Service — Response Assessment Module.

Validates LLM sentences against KB RAG context using NLI.

Usage:
    premise = ram_service.build_premise(contexts)  # once per request
    result = await ram_service.assess_sentence(sentence, premise, contexts)
    if result.label == "contradiction":
        sentence += " *(contradictive)*"
"""
import dataclasses
import difflib
import math
from typing import List, Optional

import structlog

from .interfaces import IRerankerModel, INLIModel, NLIResult, RetrievedContext
from .text_utils import split_sentences, split_table_windows

logger = structlog.get_logger(__name__)

LABEL_NEUTRAL = "neutral"
LABEL_ENTAILMENT = "entailment"
LABEL_CONTRADICTION = "contradiction"

# Contexts included in the premise (bounds NLI input length). Matches
# search_service.RERANK_TOP_K: candidates are already reranked to their top 8
# before sibling/cross-ref hydration, so this covers the full query-relevant
# primary set while excluding the lower-relevance tail.
MAX_PREMISE_CONTEXTS = 8

# Max length of the evidence snippet surfaced in citation tooltips.
EVIDENCE_SNIPPET_MAX_CHARS = 140

# Reranked candidate windows tried per sentence before falling back to the
# first candidate's verdict. Bounded at 2 because assess_sentence runs inline
# in the streaming loop, so each extra NLI call adds per-sentence latency; the
# 2nd window is only spent when the top-1 result is neutral/low-confidence.
NLI_CANDIDATE_WINDOWS = 2

# Minimum dominant-label score to short-circuit the window search. Gates only
# ENTAILMENT — CONTRADICTION must clear the higher NLI_CONTRADICTION_THRESHOLD.
NLI_CONFIDENCE_THRESHOLD = 0.5

# Minimum contradiction_score to surface a "Contradiction" badge. Deliberately
# higher than NLI_CONFIDENCE_THRESHOLD: a false "Contradiction" costs more
# trust than a missed "Supported", and general NLI models flag negation/
# conditional cues as contradictions even for logically consistent statements
# (e.g. complementary "if paid"/"if unpaid" clauses of one article).
NLI_CONTRADICTION_THRESHOLD = 0.7


def _sanitize_snippet(text: str, max_len: int = EVIDENCE_SNIPPET_MAX_CHARS) -> str:
    """Collapse whitespace and strip characters that would break the
    ``*(STATUS: SCORE; ...; Evidence:"...")*`` citation marker grammar,
    then truncate for display.
    """
    cleaned = " ".join(text.split())
    cleaned = cleaned.replace('"', "'")
    cleaned = cleaned.translate(str.maketrans("", "", ";)*"))
    if len(cleaned) > max_len:
        cleaned = cleaned[: max_len - 1].rstrip() + "…"
    return cleaned


class RAMService:
    """Response Assessment Module service.

    Builds a premise from retrieved KB contexts and validates LLM-generated
    sentences against it via NLI, called per-sentence inside ChatService's
    streaming generator. When disabled (``enabled=False``), ``assess_sentence``
    returns a neutral result immediately (zero model calls).

    Depends only on the ``INLIModel``/``IRerankerModel`` Protocols, keeping it
    in the infra-free ``thesis`` research core. Wired in
    ``app/chat/dependency.py::get_ram_service``.
    """

    def __init__(self, nli_model: INLIModel, reranker_model: IRerankerModel, enabled: bool = True):
        """``nli_model`` checks sentence-vs-context entailment; ``reranker_model``
        finds the exact candidate window within the top contexts before NLI.
        ``enabled=False`` short-circuits ``assess_sentence`` to a neutral result.
        """
        self.nli_model = nli_model
        self.reranker_model = reranker_model
        self.enabled = enabled

    def build_premise(self, contexts: List[RetrievedContext]) -> str:
        """Concatenate KB parent chunk texts into a single NLI premise.

        Called once per generate() invocation, not per sentence. The premise
        is then passed to every assess_sentence() call for that request.

        Args:
            contexts (List[RetrievedContext]): Retrieved knowledge-base contexts.

        Returns:
            str: A single string combining the top-N context texts, separated by
                 double newlines. Returns "" if contexts is empty.
        """
        if not contexts:
            return ""

        # Limit to top-N to stay within model token limits
        top_contexts = contexts[:MAX_PREMISE_CONTEXTS]
        premise = "\n\n".join(ctx.text for ctx in top_contexts)

        logger.debug(
            "Built NLI premise from %d contexts (%d chars)",
            len(top_contexts),
            len(premise),
        )
        return premise

    async def assess_sentence(
        self,
        sentence: str,
        premise: str,
        contexts: List[RetrievedContext],
    ) -> NLIResult:
        """Run NLI on one sentence (hypothesis) against the most relevant KB
        chunk. A reranker with sliding windows first locates the exact
        sub-chunk, then NLI runs against it. ``premise`` is a legacy argument
        (not used for NLI). Returns an ``NLIResult`` with canonical label and
        confidence scores.
        """
        if not self.enabled:
            return NLIResult(
                label=LABEL_NEUTRAL,
                entailment_score=1.0,
                contradiction_score=0.0,
            )

        if not contexts or not sentence.strip():
            return NLIResult(
                label=LABEL_NEUTRAL,
                entailment_score=0.5,
                contradiction_score=0.0,
            )

        try:
            top_contexts = contexts[:MAX_PREMISE_CONTEXTS]
            
            # Create sliding windows of ~2-3 sentences for precise reranking
            windows: List[str] = []
            window_to_ctx: List[RetrievedContext] = []
            
            for ctx in top_contexts:
                if ctx.content_type == "table":
                    # Row-group windows repeat the header + separator in every
                    # window; prose splitting would treat each row as a
                    # "sentence" and drop the header after the first window.
                    # Falls through to the prose path only if ctx.text isn't a
                    # parseable Markdown table (e.g. raw HTML from a failed
                    # ingest-time conversion).
                    table_windows = split_table_windows(ctx.text, rows_per_window=3, row_step=2)
                    if table_windows:
                        for window in table_windows:
                            if len(window) > 20:
                                windows.append(window)
                                window_to_ctx.append(ctx)
                        continue

                # Sentence-like units for windowing (split_sentences guards
                # against markdown list markers being read as sentence ends).
                sentences = [
                    s if s.endswith((".", "?", "!")) else s + "."
                    for s in split_sentences(ctx.text)
                ]
                # Windows of 3 sentences with 1-sentence overlap.
                if not sentences:
                    windows.append(ctx.text)
                    window_to_ctx.append(ctx)
                    continue

                window_size = 3
                step = 2
                for i in range(0, max(1, len(sentences)), step):
                    window = " ".join(sentences[i:i + window_size])
                    if len(window) > 20:  # skip tiny fragments
                        windows.append(window)
                        window_to_ctx.append(ctx)
            
            if not windows:
                return NLIResult(label=LABEL_NEUTRAL, entailment_score=0.5, contradiction_score=0.0)

            # Rerank windows against the hypothesis and take the top-N
            # candidates, not just the best match: the reranker's #1 isn't
            # always the window NLI finds entailing (see NLI_CANDIDATE_WINDOWS).
            rerank_results = await self.reranker_model.rerank(
                query=sentence,
                documents=windows,
                top_k=NLI_CANDIDATE_WINDOWS,
            )
            candidate_idxs = [r.index for r in rerank_results if 0 <= r.index < len(windows)]
        except Exception as e:
            logger.warning("Failed to reverse map citation with reranker: %s", str(e), exc_info=True)
            return NLIResult(label=LABEL_NEUTRAL, entailment_score=0.5, contradiction_score=0.0)

        if not candidate_idxs:
            return NLIResult(label=LABEL_NEUTRAL, entailment_score=0.5, contradiction_score=0.0)

        # Try each candidate in rerank order. A confident ENTAILMENT
        # short-circuits (one supporting window is enough). A confident
        # CONTRADICTION is held, not short-circuited, so a spurious
        # contradiction from one window (e.g. an exception clause) can't
        # pre-empt a valid entailment from another (e.g. the rule it qualifies)
        # that reranked lower. If nothing entails and no held contradiction
        # clears the threshold, fall back to the top candidate's raw result.
        fallback: Optional[tuple] = None
        best_contradiction: Optional[tuple] = None
        for idx in candidate_idxs:
            candidate_premise = windows[idx]
            candidate_ctx = window_to_ctx[idx]
            logger.debug(
                "Assessing sentence (%d chars) against candidate premise (%d chars)",
                len(sentence),
                len(candidate_premise),
            )
            try:
                result = await self.nli_model.check(premise=candidate_premise, hypothesis=sentence)
            except Exception as e:
                logger.warning("NLI check failed: %s", str(e), exc_info=True)
                continue

            logger.debug(
                "Candidate NLI result: label=%s entailment=%.3f contradiction=%.3f",
                result.label,
                result.entailment_score,
                result.contradiction_score,
            )

            if fallback is None:
                fallback = (result, candidate_ctx, candidate_premise)

            if result.label == LABEL_ENTAILMENT and result.entailment_score >= NLI_CONFIDENCE_THRESHOLD:
                chosen_result, chosen_ctx, chosen_premise = result, candidate_ctx, candidate_premise
                break

            if (
                best_contradiction is None
                and result.label == LABEL_CONTRADICTION
                and result.contradiction_score >= NLI_CONTRADICTION_THRESHOLD
            ):
                best_contradiction = (result, candidate_ctx, candidate_premise)
        else:
            if best_contradiction is not None:
                chosen_result, chosen_ctx, chosen_premise = best_contradiction
            elif fallback is not None:
                chosen_result, chosen_ctx, chosen_premise = fallback
            else:
                return NLIResult(label=LABEL_NEUTRAL, entailment_score=0.5, contradiction_score=0.0)

        # The fallback can carry a raw CONTRADICTION that never cleared the
        # threshold (only best_contradiction is pre-gated). _format_citation
        # renders whatever label reaches it, so downgrade a sub-threshold
        # contradiction to neutral rather than surfacing a low-confidence badge.
        if (
            chosen_result.label == LABEL_CONTRADICTION
            and chosen_result.contradiction_score < NLI_CONTRADICTION_THRESHOLD
        ):
            chosen_result = dataclasses.replace(chosen_result, label=LABEL_NEUTRAL)

        # NLIResult is frozen; use dataclasses.replace to attach metadata.
        return dataclasses.replace(
            chosen_result,
            source_title=chosen_ctx.source_title,
            page=chosen_ctx.page,
            doc_id=chosen_ctx.doc_id,
            evidence_snippet=_sanitize_snippet(chosen_premise),
        )