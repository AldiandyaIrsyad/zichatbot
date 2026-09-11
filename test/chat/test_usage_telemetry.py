"""Tests for LLM cost telemetry.

One chat turn fans out into LLM calls owned by three different objects, so
attribution is by async context rather than by threading a callback through
every signature. These cover the collector's semantics and the adapter's
parsing of the varied usage shapes gateways return.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

from app.chat.domain.usage import (
    LLMUsage,
    UsageCollector,
    collect_usage,
    record_usage,
)
from app.chat.infra.llm_connection import LLMConnection


def _usage(prompt: int = 100, completion: int = 10, cost: float | None = 0.001, **kw) -> LLMUsage:
    return LLMUsage(
        model="m", prompt_tokens=prompt, completion_tokens=completion, cost_usd=cost, **kw
    )


class TestUsageCollector:
    def test_sums_across_calls(self) -> None:
        with collect_usage() as usage:
            record_usage(_usage(265, 150, 0.00005, call="hyde"))
            record_usage(_usage(6509, 350, 0.00058))
        assert usage.call_count == 2
        assert usage.prompt_tokens == 6774
        assert usage.completion_tokens == 500
        assert abs(usage.cost_usd - 0.00063) < 1e-9

    def test_cost_is_none_when_no_call_reported_one(self) -> None:
        # A self-hosted backend returns tokens but no price; reporting 0.0 would
        # read as "free" rather than "unknown".
        with collect_usage() as usage:
            record_usage(_usage(cost=None))
        assert usage.cost_usd is None
        assert usage.prompt_tokens == 100

    def test_recording_outside_a_block_is_a_no_op(self) -> None:
        record_usage(_usage())  # must not raise

    def test_collectors_do_not_leak_between_blocks(self) -> None:
        with collect_usage() as first:
            record_usage(_usage())
        with collect_usage() as second:
            record_usage(_usage())
            record_usage(_usage())
        assert first.call_count == 1
        assert second.call_count == 2

    def test_records_from_child_tasks_are_visible(self) -> None:
        # HyDE fans its passages out with asyncio.gather; contextvars are copied
        # into those tasks, so attribution relies on the collector being mutable.
        async def scenario() -> UsageCollector:
            with collect_usage() as usage:
                async def one() -> None:
                    record_usage(_usage(call="hyde"))

                await asyncio.gather(one(), one(), one())
            return usage

        assert asyncio.run(scenario()).call_count == 3


class TestExtractUsage:
    def test_parses_openrouter_shape(self) -> None:
        response = SimpleNamespace(
            model="deepseek/deepseek-v4-flash-0731",
            provider="DeepInfra",
            id="gen-abc",
            usage=SimpleNamespace(
                prompt_tokens=6509,
                completion_tokens=350,
                cost=0.00058,
                prompt_tokens_details=SimpleNamespace(cached_tokens=1024),
            ),
        )
        usage = LLMConnection._extract_usage(response, "fallback", "generation")
        assert usage.provider == "DeepInfra"
        assert usage.prompt_tokens == 6509
        assert usage.cached_prompt_tokens == 1024
        assert usage.cost_usd == 0.00058
        assert usage.call == "generation"

    def test_parses_plain_openai_shape(self) -> None:
        # No cost, no provider, no cache details — the common self-hosted case.
        response = SimpleNamespace(
            model=None,
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
        )
        usage = LLMConnection._extract_usage(response, "local-model", "hyde")
        assert usage.model == "local-model"
        assert usage.cost_usd is None
        assert usage.provider is None
        assert usage.cached_prompt_tokens == 0

    def test_cache_details_as_a_dict(self) -> None:
        response = SimpleNamespace(
            model="m",
            usage=SimpleNamespace(
                prompt_tokens=10,
                completion_tokens=5,
                prompt_tokens_details={"cached_tokens": 7},
            ),
        )
        assert LLMConnection._extract_usage(response, "m", "x").cached_prompt_tokens == 7

    def test_chunk_without_usage_yields_none(self) -> None:
        # Ordinary content chunks carry no usage; only the trailer does.
        assert LLMConnection._extract_usage(SimpleNamespace(usage=None), "m", "x") is None


def _delta_chunk(content: str, *, finish: str | None = None, usage: object = None):
    """A streaming chunk carrying a content delta (and optionally usage)."""
    return SimpleNamespace(
        model="m",
        provider="DeepInfra",
        id="gen-abc",
        usage=usage,
        choices=[SimpleNamespace(delta=SimpleNamespace(content=content), finish_reason=finish)],
    )


def _usage_payload(prompt: int = 15, completion: int = 4, cost: float = 1.92e-06):
    return SimpleNamespace(
        prompt_tokens=prompt,
        completion_tokens=completion,
        cost=cost,
        prompt_tokens_details=SimpleNamespace(cached_tokens=0),
    )


def _drain(conn: LLMConnection, chunks: list) -> tuple[str, UsageCollector]:
    """Run ``stream_chat`` over a canned chunk sequence, collecting usage."""

    async def fake_create(**_kw):
        async def gen():
            for chunk in chunks:
                yield chunk

        return gen()

    conn._client.chat.completions.create = fake_create  # type: ignore[assignment]

    async def scenario() -> tuple[str, UsageCollector]:
        with collect_usage() as usage:
            text = ""
            async for token in conn.stream_chat("m", [{"role": "user", "content": "hi"}], 16):
                text += token
        return text, usage

    return asyncio.run(scenario())


class TestStreamingUsageCapture:
    """Regression guard: streaming accounting silently recorded nothing.

    A 2h measurement run was invalidated because usage was only read from
    choices-less chunks, while OpenRouter attaches it to the *final content
    chunk* (empty delta, ``finish_reason="stop"``). The answer streamed fine —
    only the cost was missing, so nothing failed loudly.
    """

    def test_usage_on_final_chunk_with_choices_is_captured(self) -> None:
        text, usage = _drain(
            LLMConnection(),
            [
                _delta_chunk("Mer"),
                _delta_chunk("ah."),
                _delta_chunk("", finish="stop", usage=_usage_payload()),
            ],
        )
        assert text == "Merah."
        assert usage.call_count == 1
        assert usage.records[0].call == "generation"
        assert usage.prompt_tokens == 15
        assert usage.cost_usd == 1.92e-06

    def test_usage_on_a_choices_less_trailer_is_captured(self) -> None:
        # The other gateway convention (OpenAI's include_usage trailer).
        _, usage = _drain(
            LLMConnection(),
            [
                _delta_chunk("Merah."),
                SimpleNamespace(model="m", provider=None, id="g", usage=_usage_payload(), choices=[]),
            ],
        )
        assert usage.call_count == 1
        assert usage.prompt_tokens == 15

    def test_usage_is_reported_once_even_if_repeated(self) -> None:
        _, usage = _drain(
            LLMConnection(),
            [
                _delta_chunk("Merah.", usage=_usage_payload()),
                _delta_chunk("", finish="stop", usage=_usage_payload()),
            ],
        )
        assert usage.call_count == 1

    def test_stream_without_any_usage_records_nothing(self) -> None:
        text, usage = _drain(LLMConnection(), [_delta_chunk("Merah.")])
        assert text == "Merah."
        assert usage.call_count == 0


class TestProviderRouting:
    def test_routing_is_sent_when_pinned(self) -> None:
        conn = LLMConnection(
            provider_routing={"order": ["DeepInfra"], "allow_fallbacks": False}
        )
        body = conn._extra_body()
        assert body["provider"]["order"] == ["DeepInfra"]
        assert body["usage"] == {"include": True}

    def test_no_provider_field_when_unpinned(self) -> None:
        # A non-OpenRouter backend should never receive a field it must ignore.
        assert "provider" not in LLMConnection()._extra_body()
