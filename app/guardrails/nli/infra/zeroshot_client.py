"""HTTP adapter for the binary zero-shot NLI baseline.

``MoritzLaurer/bge-m3-zeroshot-v2.0-c`` is a ``XLMRobertaForSequenceClassification``
with ``id2label = {0: entailment, 1: not_entailment}``. It is a binary NLI
model, so its ``not_entailment`` mass is surfaced as ``neutral_score`` (never
contradiction) — the RAM badge then reports the claim as Supported/Neutral, with
no false "Contradicted" verdicts from a model that was never trained to
distinguish neutral from contradiction.
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional

import httpx
import structlog

from app.guardrails.nli.domain.interfaces import INLIModel
from app.guardrails.nli.domain.models import (
    LABEL_ENTAILMENT,
    LABEL_NEUTRAL,
    NLIResult,
)

logger = structlog.get_logger(__name__)


class ZeroshotNLIClient(INLIModel):
    """Binary entailment inference over an OpenAI-compatible ``/classify``
    endpoint serving ``bge-m3-zeroshot-v2.0-c``."""

    def __init__(self, base_url: str, model: str, max_concurrency: int = 8):
        self.model = model
        self._semaphore = asyncio.Semaphore(max(1, max_concurrency))
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=httpx.Timeout(10.0, connect=5.0),
        )
        logger.info("nli.zeroshot.initialized", model=model, base_url=base_url)

    async def check(self, premise: str, hypothesis: str) -> NLIResult:
        """Classify the pair, mapping entailment/not_entailment to the canonical
        ``NLIResult`` (binary label space: entailment or neutral)."""
        async with self._semaphore:
            try:
                response = await self._client.post(
                    "/classify",
                    json={"model": self.model, "input": [f"{premise} </s></s> {hypothesis}"], "raw_scores": True},
                )
                response.raise_for_status()
                return self._parse_response(response.json())
            except Exception as e:
                logger.warning("nli.zeroshot.failed", error=str(e))
                return NLIResult(label=LABEL_NEUTRAL, entailment_score=0.5, neutral_score=0.5)

    @staticmethod
    def _parse_response(data: dict[str, Any]) -> NLIResult:
        """Parse a ``raw_scores=True`` payload into a binary ``NLIResult``."""
        items = data.get("data", [])
        if not items or not items[0]:
            return NLIResult(label=LABEL_NEUTRAL, entailment_score=0.5, neutral_score=0.5)

        predictions = items[0]
        score_dict: dict[str, float] = {}
        if isinstance(predictions, list):
            for p in predictions:
                score_dict[str(p.get("label", "")).lower()] = float(p.get("score", 0.0))
        elif isinstance(predictions, dict):
            score_field = predictions.get("score")
            if isinstance(score_field, dict):
                score_dict = {str(k).lower(): float(v) for k, v in score_field.items()}
            else:
                score_dict[str(predictions.get("label", "")).lower()] = float(score_field or 0.0)

        entailment = score_dict.get("entailment", 0.0)
        not_entailment = score_dict.get("not_entailment", 0.0) or score_dict.get(
            "not_entailment", 0.0
        )
        # Normalize common aliases in case the server emits label_0/label_1.
        if not entailment and "label_0" in score_dict:
            entailment = score_dict["label_0"]
        if not not_entailment and "label_1" in score_dict:
            not_entailment = score_dict["label_1"]

        label = LABEL_ENTAILMENT if entailment >= not_entailment else LABEL_NEUTRAL
        return NLIResult(
            label=label,
            entailment_score=entailment,
            neutral_score=not_entailment,
            contradiction_score=0.0,
        )

    async def close(self) -> None:
        """Release the underlying ``httpx.AsyncClient`` connection pool."""
        await self._client.aclose()
