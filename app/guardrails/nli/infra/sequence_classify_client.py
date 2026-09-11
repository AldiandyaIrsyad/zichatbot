"""HTTP adapter for a 3-way NLI sequence-classification server.

Calls an OpenAI-compatible ``/classify`` endpoint (Infinity, or the dedicated
``services/nli`` CPU server) to classify the entailment relation between a
premise and a hypothesis. Generalizes the RAM's original ``NLIClient``: the
separator and token budgets are configurable so the same client serves both
``indo-roberta-indonli`` (514-token limit) and the mmBERT fine-tune (8192-token
context).

Fulfills ``app.guardrails.nli.domain.interfaces.INLIModel``; built by
``app.guardrails.nli.application.selector.build_nli_model``.
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional

import httpx
import structlog
from tokenizers import Tokenizer

from app.guardrails.nli.domain.interfaces import INLIModel
from app.guardrails.nli.domain.models import (
    LABEL_CONTRADICTION,
    LABEL_ENTAILMENT,
    LABEL_NEUTRAL,
    NLIResult,
)

logger = structlog.get_logger(__name__)

# Infinity returns either human-readable labels or HF-style "label_0/1/2"
# depending on the model; normalize both to the canonical strings.
_LABEL_MAP: dict[str, str] = {
    "entailment": LABEL_ENTAILMENT,
    "neutral": LABEL_NEUTRAL,
    "contradiction": LABEL_CONTRADICTION,
    "label_0": LABEL_ENTAILMENT,
    "label_1": LABEL_NEUTRAL,
    "label_2": LABEL_CONTRADICTION,
}


def _load_tokenizer(model: str) -> Tokenizer:
    """Load a model's tokenizer, preferring the local Hugging Face cache.

    ``Tokenizer.from_pretrained`` re-fetches ``tokenizer.json`` from the Hub on
    every process start (a ~1.4 MB download, and a hard failure offline).
    ``hf_hub_download`` consults the shared cache first and only revalidates,
    so repeated starts are local. Falls back to the original path if anything
    about the cache lookup fails.
    """
    from huggingface_hub import hf_hub_download

    try:
        # Cache hit: no network at all, not even a revalidation round-trip.
        return Tokenizer.from_file(
            hf_hub_download(model, "tokenizer.json", local_files_only=True)
        )
    except Exception:
        pass
    try:
        # Not cached yet — download once, populating the cache for next start.
        return Tokenizer.from_file(hf_hub_download(model, "tokenizer.json"))
    except Exception as exc:
        logger.debug("nli.tokenizer_hub_download_failed", model=model, error=str(exc))
        return Tokenizer.from_pretrained(model)


class SequenceClassifyNLIClient(INLIModel):
    """3-way NLI inference over an OpenAI-compatible ``/classify`` endpoint."""

    # indo-roberta-indonli has a 514-position embedding table, but Infinity's
    # truncation doesn't reliably clip to it, so an oversized input crashes the
    # batch worker mid-request. These defaults leave margin under 514 for
    # special tokens; mmBERT (8192 ctx) overrides them upward via the spec.
    _DEFAULT_MAX_TOTAL_TOKENS = 500
    _DEFAULT_MAX_HYPOTHESIS_TOKENS = 100

    def __init__(
        self,
        base_url: str,
        model: str,
        max_concurrency: int = 8,
        max_total_tokens: Optional[int] = None,
        max_hypothesis_tokens: Optional[int] = None,
        separator: Optional[str] = None,
    ):
        """Configure the client and load the model's tokenizer (used both to
        pick the separator and for token-accurate truncation).

        ``max_concurrency`` caps in-flight calls so a RAM burst can't overwhelm
        the server. ``max_total_tokens``/``max_hypothesis_tokens`` bound the
        input to the model's position-embedding table; when omitted they fall
        back to the indo-roberta-safe defaults.
        """
        self.model = model
        self._sep = separator or (" </s></s> " if "roberta" in model.lower() else " [SEP] ")
        self._max_total_tokens = max_total_tokens or self._DEFAULT_MAX_TOTAL_TOKENS
        self._max_hypothesis_tokens = (
            max_hypothesis_tokens or self._DEFAULT_MAX_HYPOTHESIS_TOKENS
        )
        self._semaphore = asyncio.Semaphore(max(1, max_concurrency))

        self._tokenizer: Optional[Tokenizer] = None
        try:
            self._tokenizer = _load_tokenizer(model)
            # The model's tokenizer.json ships a truncation config that would
            # override our explicit budgets; disable it so the budgets targeting
            # the real position limit take effect.
            self._tokenizer.no_truncation()
        except Exception as e:
            logger.warning("nli.tokenizer_load_failed", model=model, error=str(e))

        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=httpx.Timeout(10.0, connect=5.0),
        )
        logger.info(
            "nli.initialized",
            model=model,
            base_url=base_url,
            sep=self._sep,
            max_total_tokens=self._max_total_tokens,
        )

    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Clip ``text`` to at most ``max_tokens`` tokens using the loaded
        tokenizer, decoding back to a string (no-op if already short enough
        or if no tokenizer was loaded)."""
        if not text or self._tokenizer is None:
            return text
        ids = self._tokenizer.encode(text).ids
        if len(ids) <= max_tokens:
            return text
        return self._tokenizer.decode(ids[:max_tokens])

    async def check(self, premise: str, hypothesis: str) -> NLIResult:
        """Classify the entailment relation, truncating both inputs to the
        token budget before calling ``/classify``.

        Falls back to a neutral result (rather than raising) on request failure,
        so a transient NLI outage degrades to "no citation" instead of breaking
        the chat stream.
        """
        if self._tokenizer is not None:
            hypothesis = self._truncate_to_tokens(hypothesis, self._max_hypothesis_tokens)
            reserved = (
                len(self._tokenizer.encode(hypothesis).ids)
                + len(self._tokenizer.encode(self._sep).ids)
            )
            premise_budget = max(0, self._max_total_tokens - reserved)
            premise = self._truncate_to_tokens(premise, premise_budget)
        else:
            # No tokenizer (e.g. no network at startup): conservative char cap.
            hypothesis = hypothesis[:400]
            max_premise_chars = max(0, 1000 - len(hypothesis) - len(self._sep))
            premise = premise[:max_premise_chars]

        text = f"{premise}{self._sep}{hypothesis}"
        async with self._semaphore:
            try:
                response = await self._client.post(
                    "/classify",
                    json={"model": self.model, "input": [text], "raw_scores": True},
                )
                response.raise_for_status()
                data = response.json()
                return self._parse_response(data)
            except Exception as e:
                logger.warning("nli.failed", error=str(e))
                return NLIResult(
                    label=LABEL_NEUTRAL,
                    entailment_score=0.5,
                    contradiction_score=0.0,
                    neutral_score=0.5,
                )

    @staticmethod
    def _parse_response(data: dict[str, Any]) -> NLIResult:
        """Parse a ``/classify`` response into an ``NLIResult``.

        The per-input prediction may be a list of ``{"label", "score"}`` dicts
        (one per class — the ``raw_scores=True`` shape), or a single dict with a
        nested ``score`` dict (also raw scores) or a scalar top-1
        ``{"label", "score"}``. Returns neutral for an empty/unrecognized
        payload.
        """
        items = data.get("data", [])
        if not items or not items[0]:
            return NLIResult(
                label=LABEL_NEUTRAL,
                entailment_score=0.5,
                contradiction_score=0.0,
                neutral_score=0.5,
            )

        predictions = items[0]

        if isinstance(predictions, list) and all(isinstance(p, dict) for p in predictions):
            score_dict = {str(p.get("label", "")): float(p.get("score", 0.0)) for p in predictions}
            return SequenceClassifyNLIClient._parse_raw_scores(score_dict)
        if isinstance(predictions, dict):
            score_field = predictions.get("score")
            if isinstance(score_field, dict):
                return SequenceClassifyNLIClient._parse_raw_scores(score_field)
            return SequenceClassifyNLIClient._parse_top1(
                label=str(predictions.get("label", "")),
                score=float(score_field) if score_field is not None else 0.0,
            )

        return NLIResult(
            label=LABEL_NEUTRAL,
            entailment_score=0.5,
            contradiction_score=0.0,
            neutral_score=0.5,
        )

    @staticmethod
    def _parse_raw_scores(score_dict: dict[str, Any]) -> NLIResult:
        """Build an ``NLIResult`` from a per-class score dict, picking the
        highest-scoring class as the label."""
        scores: dict[str, float] = {}
        for raw_label, score in score_dict.items():
            canonical = _LABEL_MAP.get(raw_label.lower(), LABEL_NEUTRAL)
            scores[canonical] = float(score)

        entailment_score = scores.get(LABEL_ENTAILMENT, 0.0)
        neutral_score = scores.get(LABEL_NEUTRAL, 0.0)
        contradiction_score = scores.get(LABEL_CONTRADICTION, 0.0)

        best_label = max(
            [
                (LABEL_ENTAILMENT, entailment_score),
                (LABEL_NEUTRAL, neutral_score),
                (LABEL_CONTRADICTION, contradiction_score),
            ],
            key=lambda x: x[1],
        )[0]

        return NLIResult(
            label=best_label,
            entailment_score=entailment_score,
            contradiction_score=contradiction_score,
            neutral_score=neutral_score,
        )

    @staticmethod
    def _parse_top1(label: str, score: float) -> NLIResult:
        """Build an ``NLIResult`` from a scalar top-1 prediction."""
        canonical = _LABEL_MAP.get(label.lower(), LABEL_NEUTRAL)

        entailment_score = score if canonical == LABEL_ENTAILMENT else 0.0
        contradiction_score = score if canonical == LABEL_CONTRADICTION else 0.0
        neutral_score = score if canonical == LABEL_NEUTRAL else 0.0

        return NLIResult(
            label=canonical,
            entailment_score=entailment_score,
            contradiction_score=contradiction_score,
            neutral_score=neutral_score,
        )

    async def close(self) -> None:
        """Release the underlying ``httpx.AsyncClient`` connection pool."""
        await self._client.aclose()
