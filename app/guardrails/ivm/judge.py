"""LLM-based judge implementation for relevance checking.

Fulfills ``app/thesis/ivm/interfaces.py::IJudge``; wired in
``app/chat/dependency.py::get_relevance_checker`` (constructs the ``LLMJudge``
and wraps it in ``LLMJudgeRelevanceChecker`` when
``ChatConfig.ood_method == "llm_judge"``). Depends only on the narrow
``ILLMJudgeConnection`` Protocol, keeping it in the infra-free ``thesis`` core.
"""
from typing import Optional

import structlog
from app.guardrails.ivm.interfaces import IJudge, ILLMJudgeConnection

logger = structlog.get_logger(__name__)

DEFAULT_RELEVANCE_JUDGE_PROMPT = (
    "You are a relevance judge for a retrieval-augmented QA system. Your "
    "task is to determine if a given query is on the same topic or domain "
    "as the provided context, even if the context does not fully or "
    "directly answer it. "
    "Reply with exactly 'YES' if the query relates to the same subject "
    "matter as the context. "
    "Reply with exactly 'NO' only if the query is about a clearly "
    "unrelated topic, or is malicious/nonsensical."
)

DEFAULT_RELEVANCE_JUDGE_USER_TEMPLATE = (
    "Context:\n{context}\n\nQuery: {query}\n\nIs this relevant?"
)


class LLMJudge(IJudge):
    """Judge that uses an LLM to evaluate relevance."""

    def __init__(
        self,
        llm_connection: ILLMJudgeConnection,
        model: str = "llama3-70b-8192",
        system_prompt: Optional[str] = None,
        user_template: Optional[str] = None,
    ) -> None:
        """``llm_connection`` is the narrow ``stream_chat``-only connection used
        to run the judge prompt. ``system_prompt``/``user_template`` override
        the defaults; ``user_template`` must contain ``{context}`` and
        ``{query}`` placeholders.
        """
        self.llm_connection = llm_connection
        self.model = model
        self.system_prompt = system_prompt or DEFAULT_RELEVANCE_JUDGE_PROMPT
        self.user_template = user_template or DEFAULT_RELEVANCE_JUDGE_USER_TEMPLATE

    async def evaluate_relevance(self, query: str, context: str) -> bool:
        """Ask the LLM whether ``query`` is on-topic for ``context``.

        Streams the judge's response (see the ``max_tokens`` comment for why
        400, not a short cap) and fail-closed-parses the trailing YES/NO
        verdict: True only if the last word starts with "YES". Re-raises any
        LLM connection failure — the caller treats that as irrelevant (fail
        closed).
        """
        user_content = self.user_template.replace("{context}", context).replace("{query}", query)
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content}
        ]

        logger.info("llm_judge.evaluating", query=query[:100])
        try:
            response_chunks = []
            async for chunk in self.llm_connection.stream_chat(
                model=self.model,
                messages=messages,
                # Reasoning models (e.g. Qwen3-32B) don't reliably honor the
                # reasoning-disable request over streaming: stream_chat's
                # fallback yields the reasoning delta whenever content is empty,
                # so a ~240-token reasoning preamble arrives as regular chunks
                # before the one-word answer. 400 tokens gives headroom past a
                # typical preamble so the judge actually sees the answer.
                max_tokens=400,
            ):
                response_chunks.append(chunk)

            response_text = "".join(response_chunks).strip().upper()
            logger.info("llm_judge.result", query=query[:100], response=response_text, chunk_count=len(response_chunks))

            # Fail closed: no clear YES means irrelevant. Check the *last* word
            # rather than the first — reasoning models conclude with the answer
            # after their preamble (non-reasoning models output "YES"/"NO"
            # directly, where first- and last-word checks are equivalent).
            last_word = response_text.split()[-1].rstrip(".,!?'\"") if response_text.split() else ""
            return last_word.startswith("YES")
        except Exception as e:
            logger.error("llm_judge.error", error=str(e), exc_info=True)
            # Fail closed on exception
            raise
