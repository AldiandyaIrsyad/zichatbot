"""HyDE (Hypothetical Document Embeddings) query expander.

Implements :class:`app.kb.domain.interfaces.IQueryExpander` using an
:class:`app.chat.domain.interfaces.ILLMConnection` to generate a hypothetical
answer document for a query, which is then embedded (instead of the raw query)
to improve retrieval recall. Imports only ``chat/domain`` and ``kb/domain``
Protocols, never ``kb/application`` or ``kb/infra``.
"""

import asyncio
import time
from typing import Dict, List, Optional

import structlog

from app.chat.domain.interfaces import ILLMConnection
from app.kb.domain.interfaces import IKBRepository, IQueryExpander

logger = structlog.get_logger(__name__)

# Module-level (not per-instance) TTL cache for the KB grounding context:
# ``get_query_expander`` builds a fresh HyDEExpander per request, so an
# instance-level cache wouldn't survive between requests.
_kb_context_cache: Dict[str, object] = {"text": "", "expires_at": 0.0}
_kb_context_lock = asyncio.Lock()


class HyDEExpander(IQueryExpander):
    """Generates a hypothetical document for HyDE retrieval.

    Uses an LLM to produce a short answer-like paragraph for the query; being
    semantically closer to the KB documents than the question phrasing, its
    dense embedding can improve retrieval recall. Optional KB-title grounding
    is retained as a non-original, explicitly configurable variant.
    """

    def __init__(
        self,
        llm: ILLMConnection,
        model: str,
        prompt_template: str,
        system_prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
        num_passages: int = 3,
        kb_repo: Optional[IKBRepository] = None,
        context_max_docs: int = 20,
        context_refresh_seconds: int = 300,
    ) -> None:
        """Initialize the HyDE expander.

        ``prompt_template`` must contain a ``{query}`` placeholder;
        ``system_prompt`` may contain ``{kb_context}``, filled from ``kb_repo``
        (active document titles/descriptions) when provided, else empty.
        ``context_refresh_seconds`` is the TTL of the module-level grounding
        cache.
        """
        self._llm = llm
        self._model = model
        self._prompt_template = prompt_template
        self._system_prompt = system_prompt
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._num_passages = num_passages
        self._kb_repo = kb_repo
        self._context_max_docs = context_max_docs
        self._context_refresh_seconds = context_refresh_seconds

    async def _get_kb_context(self) -> str:
        """Return a cached, TTL-refreshed grounding block of active KB docs."""
        if self._kb_repo is None:
            return ""

        if time.monotonic() < _kb_context_cache["expires_at"]:
            return _kb_context_cache["text"]

        async with _kb_context_lock:
            # Re-check after acquiring the lock; another request may have
            # refreshed it while we waited.
            if time.monotonic() < _kb_context_cache["expires_at"]:
                return _kb_context_cache["text"]

            docs = await self._kb_repo.get_all_pdfs()
            active_docs = [d for d in docs if d.active][: self._context_max_docs]
            lines = [
                f"- {doc.title}: {(doc.description or '')[:150]}"
                for doc in active_docs
            ]
            text = "\n".join(lines)

            _kb_context_cache["text"] = text
            _kb_context_cache["expires_at"] = time.monotonic() + self._context_refresh_seconds
            return text

    async def _build_messages(self, query: str, variant_instruction: str = "") -> List[Dict[str, str]]:
        """Build one HyDE request while resolving optional KB context."""
        kb_context = await self._get_kb_context()
        system_prompt = self._system_prompt.replace("{kb_context}", kb_context)
        user_prompt = self._prompt_template.replace("{query}", query).replace("{kb_context}", kb_context)
        user_prompt = user_prompt.replace("{variant_instruction}", variant_instruction)
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    async def expand(self, query: str) -> str:
        """Generate a hypothetical answer paragraph for the query, to be
        embedded for retrieval.
        """
        if not query.strip():
            return ""

        messages = await self._build_messages(query)

        logger.debug(
            "chat.hyde.expand_start",
            model=self._model,
            query_len=len(query),
            max_tokens=self._max_tokens,
        )

        try:
            doc = await self._llm.generate(
                model=self._model,
                messages=messages,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
            )
            logger.debug(
                "chat.hyde.expand_done",
                hyde_len=len(doc),
            )
            return doc.strip()
        except Exception as exc:
            logger.error("chat.hyde.expand_error", error=str(exc), exc_info=True)
            raise

    async def expand_many(self, query: str) -> List[str]:
        """Generate a small, diverse HyDE ensemble.

        Failed or empty individual generations are omitted, so a transient
        provider failure does not make the retrieval path unavailable.
        """
        if not query.strip():
            return []

        async def generate_variant(index: int) -> str:
            instruction = (
                f"Write variant {index} of {self._num_passages}. Use a distinct but faithful "
                "wording and emphasize different query terms; preserve only facts explicitly "
                "stated in the question."
            )
            messages = await self._build_messages(query, instruction)
            return await self._llm.generate(
                model=self._model,
                messages=messages,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
            )

        generated = await asyncio.gather(
            *(generate_variant(i) for i in range(1, self._num_passages + 1)),
            return_exceptions=True,
        )
        passages: List[str] = []
        for index, item in enumerate(generated, start=1):
            if isinstance(item, Exception):
                logger.warning("chat.hyde.variant_failed", variant=index, error=str(item))
            elif item and item.strip():
                passages.append(item.strip())
            else:
                logger.warning("chat.hyde.variant_empty", variant=index)
        logger.debug("chat.hyde.expand_many_done", requested=self._num_passages, returned=len(passages))
        return passages

    async def close(self) -> None:
        """No-op — the LLM connection lifecycle is managed by the caller."""
        pass
