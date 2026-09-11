"""LLM gateway connection adapter.

Wraps the OpenAI Python SDK's async client for any OpenAI-compatible backend
(vLLM, Ollama, OpenRouter, etc.). Satisfies both
``app/chat/domain/interfaces.py::ILLMConnection`` (chat generation) and the
narrower ``app/thesis/ivm/interfaces.py::ILLMJudgeConnection`` (judge LLM).
Wired in ``app/chat/dependency.py::get_llm_connection``.
"""

from typing import Any, AsyncIterator, Optional, List, Dict
import asyncio
import structlog
from openai import APIConnectionError, APIError, AsyncOpenAI
from pydantic import SecretStr

from app.chat.domain.interfaces import ILLMConnection
from app.chat.domain.usage import LLMUsage, UsageCallback, record_usage

logger = structlog.get_logger(__name__)

# Ask OpenRouter to return per-call cost and cache accounting alongside the
# usual token counts. Harmless on backends that ignore unknown body fields
# (vLLM, Ollama), which simply return usage without the extra keys.
_USAGE_ACCOUNTING = {"include": True}


class LLMConnection(ILLMConnection):
    """Async streaming adapter for an OpenAI-compatible LLM backend."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[SecretStr] = None,
        default_headers: Optional[Dict[str, str]] = None,
        max_concurrency: int = 4,
        provider_routing: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Configure the underlying ``AsyncOpenAI`` client. ``base_url=None``
        uses the SDK default; an omitted ``api_key`` falls back to a dummy
        token (fine for backends like Ollama that don't check it).
        ``max_concurrency`` caps in-flight LLM calls across all callers
        (generation, condenser, HyDE, judge) sharing this connection.

        ``provider_routing`` is OpenRouter's provider-selection block. It
        matters for cost, not just preference: a single model is served by
        ~20 providers whose prices differ by 75%, and without pinning the
        gateway may silently route to an expensive one. Ignored by backends
        that don't understand the field.
        """
        resolved_key = api_key.get_secret_value() if api_key else "ollama-dummy-token"
        self._client = AsyncOpenAI(
            base_url=base_url,
            api_key=resolved_key,
            default_headers=default_headers,
        )
        self._semaphore = asyncio.Semaphore(max(1, max_concurrency))
        self._provider_routing = provider_routing or None
        logger.info(
            "chat.llm.initialized",
            base_url=base_url,
            has_api_key=api_key is not None,
            max_concurrency=max_concurrency,
            provider_routing=self._provider_routing,
        )

    def _extra_body(self) -> Dict[str, Any]:
        """Body fields shared by both call paths: reasoning off, usage
        accounting on, and provider pinning when configured."""
        body: Dict[str, Any] = {
            "reasoning": {"enabled": False},
            "usage": _USAGE_ACCOUNTING,
        }
        if self._provider_routing:
            body["provider"] = self._provider_routing
        return body

    @staticmethod
    def _extract_usage(response: Any, model: str, call: str) -> Optional[LLMUsage]:
        """Build an :class:`LLMUsage` from a completion or final stream chunk.

        Shapes vary by gateway: OpenRouter adds ``cost`` and ``provider`` and
        nests the cache hit under ``prompt_tokens_details.cached_tokens``, while
        a bare OpenAI-compatible server returns only the token counts. Anything
        missing is left at its default rather than guessed.
        """
        usage = getattr(response, "usage", None)
        if usage is None:
            return None

        details = getattr(usage, "prompt_tokens_details", None)
        cached = getattr(details, "cached_tokens", None) if details else None
        if cached is None and isinstance(details, dict):
            cached = details.get("cached_tokens")

        return LLMUsage(
            model=getattr(response, "model", model) or model,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            cached_prompt_tokens=cached or 0,
            cost_usd=getattr(usage, "cost", None),
            provider=getattr(response, "provider", None),
            generation_id=getattr(response, "id", None),
            call=call,
        )

    @staticmethod
    def _report_usage(usage: Optional[LLMUsage], on_usage: Optional[UsageCallback]) -> None:
        """Log the call's cost and hand it to ``on_usage`` if one was given."""
        if usage is None:
            return
        record_usage(usage)
        logger.info(
            "chat.llm.usage",
            model=usage.model,
            provider=usage.provider,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            cached_prompt_tokens=usage.cached_prompt_tokens,
            cost_usd=usage.cost_usd,
        )
        if on_usage is not None:
            try:
                on_usage(usage)
            except Exception as exc:  # never let telemetry break generation
                logger.warning("chat.llm.usage_callback_failed", error=str(exc))

    @staticmethod
    def _suppress_thinking(model: str, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Disable Qwen3's thinking mode via the ``/no_think`` soft switch.

        ``reasoning={"enabled": False}`` is ignored for Qwen3 on the served
        providers (``content`` comes back empty, the answer lands in
        ``reasoning``), so the content-empty fallback would stream a long
        English chain-of-thought as the answer. Appending the literal
        ``/no_think`` token is what actually disables it. Scoped to ``qwen``
        models so a control token never reaches a backend that would echo it.
        """
        if "qwen" not in model.lower():
            return messages
        patched = list(messages)
        for i in range(len(patched) - 1, -1, -1):
            if patched[i].get("role") == "user":
                content = patched[i].get("content", "")
                if "/no_think" not in content:
                    patched[i] = {**patched[i], "content": f"{content.rstrip()} /no_think"}
                break
        return patched

    async def stream_chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float = 0.0,
        *,
        on_usage: Optional[UsageCallback] = None,
        call: str = "generation",
    ) -> AsyncIterator[str]:
        """Stream a chat completion, handling reasoning models.

        Reasoning models may return ``content: null`` with the text in a
        ``reasoning`` field; reasoning is disabled via ``extra_body`` where
        possible, with a fallback to the ``reasoning`` delta when ``content``
        is absent. Yields response text chunks.
        """
        logger.debug("chat.llm.stream_start", model=model, message_count=len(messages), max_tokens=max_tokens, temperature=temperature)
        messages = self._suppress_thinking(model, messages)
        async with self._semaphore:
            try:
                stream = await self._client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stream=True,
                    stream_options={"include_usage": True},
                    extra_body=self._extra_body(),
                )
                usage_reported = False
                async for chunk in stream:  # type: ignore[union-attr]
                    # Usage must be checked on *every* chunk, not only on a
                    # choices-less trailer: OpenRouter attaches it to the final
                    # chunk, which still carries a (content-empty,
                    # finish_reason="stop") choice. Gating on `not chunk.choices`
                    # silently dropped every streaming call's accounting.
                    if not usage_reported and getattr(chunk, "usage", None):
                        self._report_usage(self._extract_usage(chunk, model, call), on_usage)
                        usage_reported = True
                    if not chunk.choices:
                        continue  # usage-only trailer, or a keepalive
                    delta = chunk.choices[0].delta
                    # Primary: content field (non-reasoning models or reasoning disabled)
                    content = getattr(delta, "content", None)
                    if content:
                        yield content
                    else:
                        # Fallback: reasoning field (models where reasoning can't be disabled)
                        reasoning = getattr(delta, "reasoning", None)
                        if reasoning:
                            yield reasoning
                if not usage_reported:
                    # Loud, because silence here is what invalidated a 2h
                    # measurement run: the answer streams fine, only the
                    # accounting is missing.
                    logger.warning("chat.llm.usage_missing", model=model, call=call, stream=True)
            except APIConnectionError as exc:
                logger.error("chat.llm.connection_error", model=model, error=str(exc))
                raise
            except APIError as exc:
                logger.error("chat.llm.api_error", model=model, status_code=getattr(exc, "status_code", None), error=str(exc))
                raise
            except Exception as exc:
                logger.error("chat.llm.unexpected_error", model=model, error=str(exc))
                raise

        logger.debug("chat.llm.stream_end", model=model)

    async def generate(
        self,
        model: str,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float = 0.0,
        *,
        on_usage: Optional[UsageCallback] = None,
        call: str = "generation",
    ) -> str:
        """Generate a complete (non-streaming) chat completion and return the
        full text. For tasks needing the whole response before proceeding (e.g.
        HyDE generation); uses ``stream=False`` on the same client.
        """
        logger.debug(
            "chat.llm.generate_start",
            model=model,
            message_count=len(messages),
            max_tokens=max_tokens,
            temperature=temperature,
        )
        messages = self._suppress_thinking(model, messages)
        async with self._semaphore:
            try:
                response = await self._client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stream=False,
                    extra_body=self._extra_body(),
                )
                self._report_usage(self._extract_usage(response, model, call), on_usage)
                content = response.choices[0].message.content  # type: ignore[union-attr]
                if content:
                    return content
                # Fallback: reasoning field (models where reasoning can't be disabled)
                reasoning = getattr(response.choices[0].message, "reasoning", None)  # type: ignore[union-attr]
                return reasoning or ""
            except APIConnectionError as exc:
                logger.error("chat.llm.connection_error", model=model, error=str(exc))
                raise
            except APIError as exc:
                logger.error(
                    "chat.llm.api_error",
                    model=model,
                    status_code=getattr(exc, "status_code", None),
                    error=str(exc),
                )
                raise
            except Exception as exc:
                logger.error("chat.llm.unexpected_error", model=model, error=str(exc))
                raise

    async def close(self) -> None:
        """Release the underlying ``AsyncOpenAI`` HTTP client."""
        await self._client.close()
