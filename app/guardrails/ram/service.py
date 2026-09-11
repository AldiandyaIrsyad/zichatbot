"""RAM Service — Response Assessment Module.

Validates LLM-generated claims against their cited KB chunks using NLI.

Usage:
    result = await ram_service.assess_claim(claim, contexts, citation_ids)
    badge = format_citation(result)
"""
import dataclasses
from typing import List, Optional, Tuple

import structlog

from .interfaces import INLIModel, IRerankerModel, NLIResult, RetrievedContext
from .text_utils import split_sentences, split_table_windows

logger = structlog.get_logger(__name__)

LABEL_NEUTRAL = "neutral"
LABEL_ENTAILMENT = "entailment"
LABEL_CONTRADICTION = "contradiction"

# Max length of the evidence snippet surfaced in citation tooltips.
EVIDENCE_SNIPPET_MAX_CHARS = 140

# Sliding-window geometry for locating the evidence span inside a cited chunk.
# 3 sentences with 1 of overlap: wide enough to keep a rule and its qualifier
# together, narrow enough to stay in the NLI model's training distribution
# (IndoNLI premises are single sentences; a full parent chunk runs to 1200+
# tokens and drives the model to neutral).
WINDOW_SENTENCES = 3
WINDOW_STEP = 2

# Shortest window worth an NLI call — below this it's a fragment, not evidence.
MIN_WINDOW_CHARS = 20

# Windows tried per cited chunk before falling back to the top candidate's
# verdict. Bounded at 2 because assessment runs inline in the streaming loop,
# so each extra NLI call adds per-claim latency; the 2nd is only spent when the
# 1st comes back neutral/low-confidence.
NLI_CANDIDATE_WINDOWS = 2


def _sanitize_snippet(text: str, max_len: int = EVIDENCE_SNIPPET_MAX_CHARS) -> str:
    """Collapse whitespace and strip characters that would break the
    ``*(Supported:...; Neutral:...; Contradicted:...; ...; Evidence:"...")*``
    citation marker grammar, then truncate for display.
    """
    cleaned = " ".join(text.split())
    cleaned = cleaned.replace('"', "'")
    cleaned = cleaned.translate(str.maketrans("", "", ";)*"))
    if len(cleaned) > max_len:
        cleaned = cleaned[: max_len - 1].rstrip() + "…"
    return cleaned


class RAMService:
    """Response Assessment Module service.

    Maps each ``[CIT:N]`` marker to its retrieved chunk, locates the evidence
    span *within that chunk*, and runs NLI with the atomic claim as hypothesis.
    When disabled (``enabled=False``), ``assess_claim`` returns a neutral
    result immediately (zero model calls).

    Citation targeting is the design invariant: only the chunks the LLM
    actually cited are ever considered. The reranker narrows the premise
    *inside* one cited chunk — it never widens the search to uncited context.

    Depends only on the ``INLIModel`` and ``IRerankerModel`` Protocols, keeping
    it in the infra-free research core. Wired in
    ``app/chat/dependency.py::get_ram_service``.
    """

    def __init__(
        self,
        nli_model: INLIModel,
        reranker_model: Optional[IRerankerModel] = None,
        enabled: bool = True,
        entailment_threshold: float = 0.5,
        contradiction_threshold: float = 0.7,
    ):
        """``reranker_model`` is optional: without it the premise falls back to
        the child chunk / leading window, which is still far shorter than the
        full parent text.
        """
        self.nli_model = nli_model
        self.reranker_model = reranker_model
        self.enabled = enabled
        self.entailment_threshold = entailment_threshold
        self.contradiction_threshold = contradiction_threshold

    @staticmethod
    def build_premise(ctx: RetrievedContext) -> str:
        """Build the single best-guess NLI evidence for one cited chunk.

        Prefers ``child_text`` — the sentence (or table header + row group)
        that actually matched retrieval and was reranked — for every content
        type, falling back to the full parent text. The parent is the whole
        page: at 1200+ tokens it exceeds the NLI model's input budget and sits
        far outside its training distribution, so it reads as neutral even when
        the claim is plainly supported.

        ``assess_claim`` uses this as the fallback; when a reranker is wired it
        chooses among :meth:`_premise_windows` instead.
        """
        return ctx.child_text or ctx.text

    @staticmethod
    def _premise_windows(ctx: RetrievedContext) -> List[str]:
        """Candidate evidence spans inside one cited chunk, best-guess first.

        ``child_text`` leads because retrieval already scored it against the
        query. The rest are sliding windows over the parent: row-group windows
        for a Markdown table (so every window keeps the header), sentence
        windows otherwise. Deduplicated, order-preserving.
        """
        candidates: List[str] = []
        if ctx.child_text.strip():
            candidates.append(ctx.child_text)

        windows = split_table_windows(
            ctx.text, rows_per_window=WINDOW_SENTENCES, row_step=WINDOW_STEP
        )
        if not windows:
            sentences = split_sentences(ctx.text)
            windows = [
                " ".join(sentences[i:i + WINDOW_SENTENCES])
                for i in range(0, max(1, len(sentences)), WINDOW_STEP)
            ]
        candidates.extend(w for w in windows if len(w) > MIN_WINDOW_CHARS)

        if not candidates:
            return [ctx.text] if ctx.text.strip() else []

        seen: set[str] = set()
        return [c for c in candidates if not (c in seen or seen.add(c))]

    async def _select_premises(self, claim: str, ctx: RetrievedContext) -> List[str]:
        """Rank this cited chunk's windows against ``claim``, best first.

        Falls back to the leading candidates (``child_text`` first) when no
        reranker is wired, only one candidate exists, or the call fails — a
        reranker outage should cost precision, not the whole assessment.
        """
        candidates = self._premise_windows(ctx)
        if len(candidates) <= 1 or self.reranker_model is None:
            return candidates[:NLI_CANDIDATE_WINDOWS]

        try:
            ranked = await self.reranker_model.rerank(
                query=claim, documents=candidates, top_k=NLI_CANDIDATE_WINDOWS
            )
        except Exception as e:
            logger.warning("ram.window_rerank_failed", error=str(e))
            return candidates[:NLI_CANDIDATE_WINDOWS]

        selected = [
            candidates[r.index] for r in ranked if 0 <= r.index < len(candidates)
        ]
        return selected or candidates[:NLI_CANDIDATE_WINDOWS]

    async def _check_windows(
        self, claim: str, premises: List[str]
    ) -> Optional[Tuple[NLIResult, str]]:
        """Run NLI over ranked windows, returning the chosen (result, premise).

        A confident entailment short-circuits — one supporting window is enough.
        A contradiction is held rather than short-circuited, so a spurious
        contradiction from an exception clause can't pre-empt a valid entailment
        from the rule it qualifies in a lower-ranked window. Returns None when
        every call failed.
        """
        fallback: Optional[Tuple[NLIResult, str]] = None
        best_contradiction: Optional[Tuple[NLIResult, str]] = None

        for premise in premises:
            try:
                result = await self.nli_model.check(premise=premise, hypothesis=claim)
            except Exception as e:
                logger.warning("ram.nli_check_failed", error=str(e), exc_info=True)
                continue

            if fallback is None:
                fallback = (result, premise)

            if (
                result.label == LABEL_ENTAILMENT
                and result.entailment_score >= self.entailment_threshold
            ):
                return result, premise
            if (
                result.label == LABEL_CONTRADICTION
                and result.contradiction_score >= self.contradiction_threshold
                and (
                    best_contradiction is None
                    or result.contradiction_score > best_contradiction[0].contradiction_score
                )
            ):
                best_contradiction = (result, premise)

        return best_contradiction or fallback

    async def assess_claim(
        self,
        claim: str,
        contexts: List[RetrievedContext],
        citation_ids: Tuple[int, ...],
    ) -> NLIResult:
        """Run NLI for one atomic claim against each of its cited chunks and
        return the best (most supporting) result.

        ``citation_ids`` are 1-based indices into ``contexts`` (the same list
        shown to the LLM as ``Sumber N``); nothing outside them is consulted.
        Within each cited chunk the premise is the reranker-selected window
        rather than the whole parent text — see :meth:`_select_premises`.

        Multiple citations are aggregated deterministically: the highest-scoring
        entailment wins; failing that, the highest-scoring contradiction;
        failing that, the first (neutral) result. The returned ``NLIResult``
        carries the chosen chunk's source metadata and sanitized evidence
        snippet (the window that was actually checked), and all three per-class
        scores (entailment/neutral/contradiction) for display.
        """
        if not self.enabled:
            return NLIResult(
                label=LABEL_NEUTRAL,
                entailment_score=1.0,
                contradiction_score=0.0,
            )

        if not contexts or not claim.strip() or not citation_ids:
            return NLIResult(
                label=LABEL_NEUTRAL,
                entailment_score=0.5,
                contradiction_score=0.0,
            )

        results: List[Tuple[NLIResult, RetrievedContext, str]] = []
        for cid in citation_ids:
            idx = cid - 1
            if idx < 0 or idx >= len(contexts):
                logger.debug("ram.citation_index_out_of_range", index=cid, contexts=len(contexts))
                continue
            ctx = contexts[idx]
            premises = await self._select_premises(claim, ctx)
            if not premises:
                continue
            chosen = await self._check_windows(claim, premises)
            if chosen is None:
                continue
            result, premise = chosen
            results.append((result, ctx, premise))

        if not results:
            return NLIResult(label=LABEL_NEUTRAL, entailment_score=0.5, contradiction_score=0.0)

        chosen = self._pick_best(results)

        # NLIResult is frozen; use dataclasses.replace to attach metadata.
        return dataclasses.replace(
            chosen[0],
            source_title=chosen[1].source_title,
            page=chosen[1].page,
            doc_id=chosen[1].doc_id,
            evidence_snippet=_sanitize_snippet(chosen[2]),
        )

    def _pick_best(
        self,
        results: List[Tuple[NLIResult, RetrievedContext, str]],
    ) -> Tuple[NLIResult, RetrievedContext, str]:
        """Prefer the highest entailment; else the highest contradiction; else first.

        The winning label is gated by the configured confidence thresholds: a
        model-entailment below ``entailment_threshold`` or a model-contradiction
        below ``contradiction_threshold`` is downgraded to neutral. Per-class
        scores are preserved for the citation badge — only the *label* is gated.
        """
        def effective_label(item: Tuple[NLIResult, RetrievedContext, str]) -> str:
            result = item[0]
            if (
                result.label == LABEL_ENTAILMENT
                and result.entailment_score < self.entailment_threshold
            ):
                return LABEL_NEUTRAL
            if (
                result.label == LABEL_CONTRADICTION
                and result.contradiction_score < self.contradiction_threshold
            ):
                return LABEL_NEUTRAL
            return result.label

        best = results[0]
        for item in results[1:]:
            item_label = effective_label(item)
            best_label = effective_label(best)
            if item_label == LABEL_ENTAILMENT and (
                best_label != LABEL_ENTAILMENT
                or item[0].entailment_score > best[0].entailment_score
            ):
                best = item
            elif best_label != LABEL_ENTAILMENT and item_label == LABEL_CONTRADICTION and (
                best_label != LABEL_CONTRADICTION
                or item[0].contradiction_score > best[0].contradiction_score
            ):
                best = item

        # Relabel the winner to neutral if the gate downgraded it (scores kept).
        winner = best[0]
        gated_label = effective_label(best)
        if gated_label != winner.label:
            winner = dataclasses.replace(winner, label=gated_label)
        return (winner, best[1], best[2])