"""Ports (Protocol interfaces) for the Input Validation Module (IVM).

Part of the pure ``thesis`` core, the IVM defines its own ports so it never
imports chat/kb infra directly. Adapters live in ``app/thesis/ivm/`` and
``app/chat/infra/``, wired in ``app/chat/dependency.py``.

Ports → adapters: :class:`ISafetyModel` → ``prompt_guard_client.PromptGuardClient``;
:class:`ILLMJudgeConnection` → ``llm_connection.LLMConnection`` (a narrower view
of the chat ``ILLMConnection``); :class:`IJudge` → ``judge.LLMJudge``;
:class:`IRelevanceChecker` → ``checkers.py`` (``LLMJudgeRelevanceChecker``
(default), ``SimilarityThresholdRelevanceChecker``, ``NliEntailmentRelevanceChecker``).
"""

from dataclasses import dataclass
from typing import List, Protocol, AsyncIterator, Any, Dict


@dataclass(frozen=True)
class SafetyResult:
    """Outcome of a prompt injection classification: whether the input passed
    (``is_safe``) plus a human-readable reason / raw classifier output.
    """

    is_safe: bool
    message: str


class ISafetyModel(Protocol):
    """Port for prompt injection detection adapters. Implemented by
    ``app/chat/infra/prompt_guard_client.py::PromptGuardClient``.
    """

    async def check_prompt(self, text: str) -> SafetyResult:
        """Classify a user input for prompt injection / jailbreak, returning a
        ``SafetyResult`` with the verdict and reason.
        """
        ...


class ILLMJudgeConnection(Protocol):
    """Port for the LLM connection used by the relevance judge.

    A deliberately narrower view than the chat ``ILLMConnection`` — it exposes
    only ``stream_chat`` — so the judge depends on the minimal surface it needs.
    Implemented by ``app/chat/infra/llm_connection.py::LLMConnection`` (the same
    adapter fulfills both ports).
    """

    def stream_chat(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        max_tokens: int = 100,
    ) -> AsyncIterator[str]:
        """Stream a chat completion, yielding incremental text fragments."""
        ...


class IJudge(Protocol):
    """Port for an LLM-based relevance judge. Implemented by
    ``app/thesis/ivm/judge.py::LLMJudge``.
    """

    async def evaluate_relevance(self, query: str, context: str) -> bool:
        """Return True if ``query`` is relevant to ``context``."""
        ...


class IRelevanceChecker(Protocol):
    """Port for an OOD/relevance decision backend, used by ``RelevanceService``
    to decide whether a query is in-domain. Implementations in
    ``app/thesis/ivm/checkers.py``: ``LLMJudgeRelevanceChecker`` (default,
    wraps :class:`IJudge`), ``SimilarityThresholdRelevanceChecker`` (kNN-OOD,
    no model call), ``NliEntailmentRelevanceChecker`` (NLI entailment threshold).
    """

    async def check_query(
        self, query: str, context_chunks: List[str], context_scores: List[float]
    ) -> bool:
        """Decide whether a user query is in-domain.

        Args:
            query: The user's raw query text.
            context_chunks: Text of the top retrieved KB contexts for this query.
            context_scores: Similarity scores (same order) already computed
                by retrieval, so implementations that only need scores can
                skip re-embedding/re-searching.

        Returns:
            bool: True if the query is relevant/in-domain.
        """
        ...