"""HTTP adapter for a dedicated pair-input NLI server (``services/nli``).

Unlike Infinity's ``/classify`` (which takes a single pre-joined string), the
dedicated ``services/nli`` server takes premise/hypothesis pairs so its tokenizer
can join them with the model's own separator (mmBERT's Gemma-2 tokenizer uses
``</s>``, not ``[SEP]``). Response parsing is shared with
:class:`SequenceClassifyNLIClient`.
"""

from __future__ import annotations

import asyncio

import httpx
import structlog

from app.guardrails.nli.domain.interfaces import INLIModel
from app.guardrails.nli.domain.models import LABEL_NEUTRAL, NLIResult
from app.guardrails.nli.infra.sequence_classify_client import (
    SequenceClassifyNLIClient,
)

logger = structlog.get_logger(__name__)


class PairClassifyNLIClient(INLIModel):
    """3-way NLI over the dedicated ``services/nli`` ``/classify`` endpoint
    (premise/hypothesis pair input)."""

    def __init__(self, base_url: str, model: str, max_concurrency: int = 8):
        self.model = model
        self._semaphore = asyncio.Semaphore(max(1, max_concurrency))
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=httpx.Timeout(10.0, connect=5.0),
        )
        logger.info("nli.pair.initialized", model=model, base_url=base_url)

    async def check(self, premise: str, hypothesis: str) -> NLIResult:
        """Classify the pair; the server tokenizes + truncates with the model's
        own tokenizer. Fails open to neutral on request error."""
        async with self._semaphore:
            try:
                response = await self._client.post(
                    "/classify",
                    json={
                        "model": self.model,
                        "input": [{"premise": premise, "hypothesis": hypothesis}],
                        "raw_scores": True,
                    },
                )
                response.raise_for_status()
                return SequenceClassifyNLIClient._parse_response(response.json())
            except Exception as e:
                logger.warning("nli.pair.failed", error=str(e))
                return NLIResult(
                    label=LABEL_NEUTRAL,
                    entailment_score=0.5,
                    contradiction_score=0.0,
                    neutral_score=0.5,
                )

    async def close(self) -> None:
        """Release the underlying ``httpx.AsyncClient`` connection pool."""
        await self._client.aclose()
