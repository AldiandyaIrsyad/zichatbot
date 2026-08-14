"""Concrete IRelevanceChecker implementations.

- ``LLMJudgeRelevanceChecker``: wraps an LLM-as-judge (IJudge), one call per
  check, fails closed on error.
- ``SimilarityThresholdRelevanceChecker``: thresholds retrieval scores only
  (kNN-OOD framing), no model call.
- ``NliEntailmentRelevanceChecker``: thresholds the NLI entailment_score
  between query and joined context.

Fulfills ``app/thesis/ivm/interfaces.py::IRelevanceChecker``; wired in
``app/chat/dependency.py::get_relevance_checker``, which selects one at runtime
via ``ChatConfig.ood_method`` ("llm_judge" / "similarity_threshold" /
"nli_entailment"). Part of the pure ``thesis`` core (no infra imports);
``NliEntailmentRelevanceChecker`` depends only on the ``INLIModel`` Protocol.
"""
from typing import List

import structlog

from .interfaces import IJudge, IRelevanceChecker
from app.guardrails.ram.interfaces import INLIModel

logger = structlog.get_logger(__name__)


class LLMJudgeRelevanceChecker(IRelevanceChecker):
    """Relevance backend using an LLM-as-judge (the default method).

    Delegates the decision to an injected :class:`IJudge`
    (``app/thesis/ivm/judge.py::LLMJudge``) — one LLM call per check. Fails
    closed: any judge exception propagates to ``RelevanceService.check_relevance``
    rather than being swallowed here. Selected when
    ``ChatConfig.ood_method == "llm_judge"``.
    """

    def __init__(self, judge: IJudge):
        """``judge`` is the LLM-as-judge backend relevance calls delegate to."""
        self.judge = judge

    async def check_query(
        self, query: str, context_chunks: List[str], context_scores: List[float]
    ) -> bool:
        """Ask the judge whether ``query`` is on-topic for the joined context.
        ``context_scores`` is accepted for interface compatibility but unused —
        this backend reasons only over the context text. True if in-domain.
        """
        combined_context = "\n".join(context_chunks)
        return await self.judge.evaluate_relevance(query, combined_context)


class SimilarityThresholdRelevanceChecker(IRelevanceChecker):
    """Relevance backend using only the retrieval scores already computed
    upstream — no LLM call, no re-embedding, no new dependencies.

    Treats the top retrieval score as a nearest-neighbor OOD signal (kNN-OOD
    framing): similarity to the nearest in-distribution neighbor is used
    directly as the OOD score.

    IMPORTANT: ``context_scores`` are Qdrant RRF-fusion scores (see
    ``app/kb/infra/qdrant_store.py::hybrid_search``), NOT a bounded cosine
    similarity in [0, 1]. RRF scores are a sum of 1/(rank + k) terms, so their
    scale is small and depends on how many ranked lists a candidate appears in.
    ``threshold`` must be calibrated against this KB's actual score
    distribution. Selected when ``ChatConfig.ood_method == "similarity_threshold"``.
    """

    def __init__(self, threshold: float):
        """``threshold`` is the minimum top retrieval score for a query to count
        as in-domain; calibrate against this KB's own RRF score distribution.
        """
        self.threshold = threshold

    async def check_query(
        self, query: str, context_chunks: List[str], context_scores: List[float]
    ) -> bool:
        """Decide relevance from the max retrieval score alone. ``context_chunks``
        is unused; True if the top score meets ``self.threshold``.
        """
        if not context_scores:
            logger.warning("similarity_threshold_checker.no_scores", query=query[:100])
            return False

        top_score = max(context_scores)
        is_relevant = top_score >= self.threshold
        logger.info(
            "similarity_threshold_checker.result",
            query=query[:100],
            top_score=top_score,
            threshold=self.threshold,
            is_relevant=is_relevant,
        )
        return is_relevant


class NliEntailmentRelevanceChecker(IRelevanceChecker):
    """Relevance backend using an NLI model's entailment score between the
    joined KB context (premise) and the user query (hypothesis) — the
    entailment probability from a general-purpose NLI model used as an
    off-the-shelf relevance score.

    Reuses the Indonesian NLI model already wired for RAM
    (StevenLimcorn/indo-roberta-indonli via Infinity) through the ``INLIModel``
    Protocol; the concrete instance is injected at the composition root.

    Thresholds the continuous ``entailment_score`` rather than checking
    ``label == "entailment"``: off-topic queries often yield low-confidence
    "neutral" rather than a clean unrelated signal, so the continuous score is
    more robust (mirrors RAMService's NLI_CONFIDENCE_THRESHOLD approach).

    KNOWN LIMITATION: ``NLIClient.check()`` fails open on infra errors — it
    returns a neutral ``NLIResult(entailment_score=0.5)`` rather than raising.
    At the default threshold of 0.5 an Infinity outage is treated as relevant,
    unlike ``LLMJudgeRelevanceChecker``'s fail-closed behavior. For fail-closed
    semantics, set ``ood_nli_entailment_threshold`` above 0.5 (e.g. 0.55).
    Selected when ``ChatConfig.ood_method == "nli_entailment"``.
    """

    def __init__(self, nli_model: INLIModel, threshold: float):
        """``nli_model`` scores entailment between context and query;
        ``threshold`` is the minimum entailment_score to count as in-domain
        (see the fail-open note above re: the 0.5 default).
        """
        self.nli_model = nli_model
        self.threshold = threshold

    async def check_query(
        self, query: str, context_chunks: List[str], context_scores: List[float]
    ) -> bool:
        """Run NLI with the joined context as premise and the query as
        hypothesis. ``context_scores`` is unused. True if
        ``entailment_score >= self.threshold``.
        """
        if not context_chunks:
            logger.warning("nli_entailment_checker.no_context", query=query[:100])
            return False

        premise = "\n".join(context_chunks)
        result = await self.nli_model.check(premise=premise, hypothesis=query)
        is_relevant = result.entailment_score >= self.threshold
        logger.info(
            "nli_entailment_checker.result",
            query=query[:100],
            label=result.label,
            entailment_score=result.entailment_score,
            threshold=self.threshold,
            is_relevant=is_relevant,
        )
        return is_relevant
