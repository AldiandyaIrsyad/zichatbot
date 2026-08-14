"""LLM gateway connection adapter.

Wraps the OpenAI Python SDK's async client for any OpenAI-compatible backend
(vLLM, Ollama, OpenRouter, etc.). Satisfies both
``app/chat/domain/interfaces.py::ILLMConnection`` (chat generation) and the
narrower ``app/thesis/ivm/interfaces.py::ILLMJudgeConnection`` (judge LLM).
Wired in ``app/chat/dependency.py::get_llm_connection``.
"""

from typing import AsyncIterator, Optional, List, Dict
import structlog
from openai import APIConnectionError, APIError, AsyncOpenAI
from pydantic import SecretStr

from app.chat.domain.interfaces import ILLMConnection

logger = structlog.get_logger(__name__)


class LLMConnection(ILLMConnection):
    """Async streaming adapter for an OpenAI-compatible LLM backend."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[SecretStr] = None,
        default_headers: Optional[Dict[str, str]] = None,
    ) -> None:
        """Configure the underlying ``AsyncOpenAI`` client. ``base_url=None``
        uses the SDK default; an omitted ``api_key`` falls back to a dummy
        token (fine for backends like Ollama that don't check it).
        """
        resolved_key = api_key.get_secret_value() if api_key else "ollama-dummy-token"
        self._client = AsyncOpenAI(
            base_url=base_url,
            api_key=resolved_key,
            default_headers=default_headers,
        )
        logger.info(
            "chat.llm.initialized",
            base_url=base_url,
            has_api_key=api_key is not None,
        )

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
    ) -> AsyncIterator[str]:
        """Stream a chat completion, handling reasoning models.

        Reasoning models may return ``content: null`` with the text in a
        ``reasoning`` field; reasoning is disabled via ``extra_body`` where
        possible, with a fallback to the ``reasoning`` delta when ``content``
        is absent. Yields response text chunks.
        """
        logger.debug("chat.llm.stream_start", model=model, message_count=len(messages), max_tokens=max_tokens, temperature=temperature)
        messages = self._suppress_thinking(model, messages)
        try:
            stream = await self._client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
                extra_body={"reasoning": {"enabled": False}},
            )
            async for chunk in stream:  # type: ignore[union-attr]
                if not chunk.choices:
                    continue
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
        try:
            response = await self._client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=False,
                extra_body={"reasoning": {"enabled": False}},
            )
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
