"""Tests for LLMJudge (app/thesis/ivm/judge.py)."""
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.guardrails.ivm.judge import DEFAULT_RELEVANCE_JUDGE_USER_TEMPLATE, LLMJudge


async def _fake_stream(chunks: list[str]):
    for chunk in chunks:
        yield chunk


def _make_llm_connection(response_chunks: list[str]) -> MagicMock:
    conn = MagicMock()
    conn.stream_chat = MagicMock(return_value=_fake_stream(response_chunks))
    return conn


class TestLLMJudgeUserTemplate:
    @pytest.mark.asyncio
    async def test_default_user_template_used_when_not_provided(self) -> None:
        conn = _make_llm_connection(["YES"])
        judge = LLMJudge(llm_connection=conn)

        await judge.evaluate_relevance(query="apa itu cuti?", context="SOP cuti pegawai")

        messages = conn.stream_chat.call_args.kwargs["messages"]
        assert messages[1]["content"] == DEFAULT_RELEVANCE_JUDGE_USER_TEMPLATE.replace(
            "{context}", "SOP cuti pegawai"
        ).replace("{query}", "apa itu cuti?")

    @pytest.mark.asyncio
    async def test_custom_user_template_is_used(self) -> None:
        conn = _make_llm_connection(["NO"])
        judge = LLMJudge(
            llm_connection=conn,
            user_template="CTX={context} | Q={query}",
        )

        result = await judge.evaluate_relevance(query="sepeda listrik", context="SOP cuti")

        messages = conn.stream_chat.call_args.kwargs["messages"]
        assert messages[1]["content"] == "CTX=SOP cuti | Q=sepeda listrik"
        assert result is False

    @pytest.mark.asyncio
    async def test_template_substitution_survives_braces_in_context(self) -> None:
        conn = _make_llm_connection(["YES"])
        judge = LLMJudge(llm_connection=conn)

        await judge.evaluate_relevance(
            query="q", context="contains {literal braces} in text"
        )

        messages = conn.stream_chat.call_args.kwargs["messages"]
        assert "contains {literal braces} in text" in messages[1]["content"]


class TestLLMJudgeReasoningModelPreamble:
    """Regression coverage for reasoning models (e.g. Qwen3-32B) whose
    streamed response is a long free-text preamble concluding with the
    actual YES/NO, rather than leading with it. LLMConnection.stream_chat's
    fallback yields the reasoning delta as regular content chunks whenever
    the real content delta is empty (so callers that just concatenate
    chunks don't silently lose everything) — confirmed live against
    OpenRouter that this preamble can run ~240 tokens before the model's
    real answer. A prefix check on such a response always fails; the
    (correct) fail-closed logic then blocks every query regardless of
    true relevance, silently disabling the whole relevance gate."""

    @pytest.mark.asyncio
    async def test_reasoning_preamble_then_yes_is_relevant(self) -> None:
        conn = _make_llm_connection([
            "Okay, let's see. The user is asking about leave procedures ",
            "and the context is about employee leave SOPs. These match. ",
            "The answer is YES",
        ])
        judge = LLMJudge(llm_connection=conn)

        result = await judge.evaluate_relevance(query="apa itu cuti?", context="SOP cuti pegawai")

        assert result is True

    @pytest.mark.asyncio
    async def test_reasoning_preamble_then_no_is_irrelevant(self) -> None:
        conn = _make_llm_connection([
            "Okay, let's see. The query is about baking a chocolate cake, ",
            "and the context is a legal regulation document. Unrelated. ",
            "Therefore the answer should be 'NO'.",
        ])
        judge = LLMJudge(llm_connection=conn)

        result = await judge.evaluate_relevance(query="cara membuat kue coklat?", context="Peraturan Rektor ...")

        assert result is False

    @pytest.mark.asyncio
    async def test_max_tokens_gives_headroom_for_reasoning(self) -> None:
        conn = _make_llm_connection(["YES"])
        judge = LLMJudge(llm_connection=conn)

        await judge.evaluate_relevance(query="q", context="c")

        assert conn.stream_chat.call_args.kwargs["max_tokens"] >= 400
