"""Natural Language Inference (NLI) infrastructure adapter.

Calls an Infinity-hosted NLI model (indo-roberta-indonli) to classify the
entailment relation between a generated sentence and retrieved KB context,
enabling per-sentence hallucination detection. Fulfills
``app/thesis/ram/interfaces.py::INLIModel``; wired in
``app/chat/dependency.py::get_nli_client``.
"""

import httpx
import structlog
from typing import Any, Optional

from tokenizers import Tokenizer

from app.guardrails.ram.interfaces import INLIModel, NLIResult

logger = structlog.get_logger(__name__)

LABEL_ENTAILMENT = "entailment"
LABEL_NEUTRAL = "neutral"
LABEL_CONTRADICTION = "contradiction"

# Infinity returns either human-readable labels or HF-style "label_0/1/2"
# depending on the model; normalize both to the RAM service's canonical strings.
_LABEL_MAP: dict[str, str] = {
    "entailment": LABEL_ENTAILMENT,
    "neutral": LABEL_NEUTRAL,
    "contradiction": LABEL_CONTRADICTION,
    "label_0": LABEL_ENTAILMENT,
    "label_1": LABEL_NEUTRAL,
    "label_2": LABEL_CONTRADICTION,
}


class NLIClient(INLIModel):
    """NLI inference via the Infinity HTTP server."""

    # indo-roberta-indonli has a 514-position embedding table, but Infinity's
    # truncation doesn't reliably clip to it, so an oversized input crashes the
    # batch worker mid-request (hanging queued requests on the shared server).
    # Char count isn't a safe token proxy for dense Indonesian legal text, so
    # truncate by actual token count, leaving margin under 514 for special
    # tokens.
    _MAX_TOTAL_TOKENS = 500
    _MAX_HYPOTHESIS_TOKENS = 150

    def __init__(self, base_url: str, model: str):
        """Configure the Infinity client and load the model's tokenizer (used
        both to pick the NLI separator style and for token-accurate truncation).
        """
        self.model = model
        self._nli_sep = " </s></s> " if "roberta" in model.lower() else " [SEP] "

        self._tokenizer: Optional[Tokenizer] = None
        try:
            self._tokenizer = Tokenizer.from_pretrained(model)
            # The model's tokenizer.json ships a truncation config (max_length=
            # 128) that would override our explicit budgets; disable it so the
            # budgets targeting the real 514-position limit take effect.
            self._tokenizer.no_truncation()
        except Exception as e:
            logger.warning("chat.nli.tokenizer_load_failed", error=str(e))

        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=httpx.Timeout(10.0, connect=5.0),
        )
        logger.info("chat.nli.initialized", model=model, base_url=base_url, sep=self._nli_sep)

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
        """Classify the entailment relation between ``premise`` (retrieved KB
        context) and ``hypothesis`` (a generated sentence), truncating both to
        the model's token budget before calling Infinity's ``/classify``.

        Falls back to a neutral result (rather than raising) on request failure,
        so a transient NLI outage degrades to "no citation" instead of breaking
        the chat stream.
        """
        if self._tokenizer is not None:
            hypothesis = self._truncate_to_tokens(hypothesis, self._MAX_HYPOTHESIS_TOKENS)
            reserved = len(self._tokenizer.encode(hypothesis).ids) + len(self._tokenizer.encode(self._nli_sep).ids)
            premise_budget = max(0, self._MAX_TOTAL_TOKENS - reserved)
            premise = self._truncate_to_tokens(premise, premise_budget)
        else:
            # No tokenizer (e.g. no network at startup): conservative char cap.
            hypothesis = hypothesis[:400]
            max_premise_chars = max(0, 1000 - len(hypothesis) - len(self._nli_sep))
            premise = premise[:max_premise_chars]

        text = f"{premise}{self._nli_sep}{hypothesis}"
        try:
            response = await self._client.post(
                "/classify",
                json={
                    "model": self.model,
                    "input": [text],
                    "raw_scores": True,
                },
            )
            response.raise_for_status()
            data = response.json()
            return self._parse_response(data)
        except Exception as e:
            logger.warning("chat.nli.failed", error=str(e))
            return NLIResult(
                label=LABEL_NEUTRAL,
                entailment_score=0.5,
                contradiction_score=0.0,
                neutral_score=0.5,
            )

    @staticmethod
    def _parse_response(data: dict[str, Any]) -> NLIResult:
        """Parse an Infinity ``/classify`` response into an ``NLIResult``.

        Infinity's response shape for this model varies by deployment: the
        per-input prediction may be a list of ``{"label", "score"}`` dicts
        (one per class — the ``raw_scores=True`` shape, dispatched to
        ``_parse_raw_scores``), or a single dict with either a nested
        ``score`` dict (also raw scores) or a scalar top-1
        ``{"label", "score"}`` (dispatched to ``_parse_top1``). Returns a
        neutral result for an empty/unrecognized payload.
        """
        items = data.get("data", [])
        if not items or not items[0]:
            return NLIResult(label=LABEL_NEUTRAL, entailment_score=0.5, contradiction_score=0.0, neutral_score=0.5)

        predictions = items[0]
        
        if isinstance(predictions, list) and all(isinstance(p, dict) for p in predictions):
            score_dict = {str(p.get("label", "")): float(p.get("score", 0.0)) for p in predictions}
            return NLIClient._parse_raw_scores(score_dict)
        elif isinstance(predictions, dict):
            score_field = predictions.get("score")
            if isinstance(score_field, dict):
                return NLIClient._parse_raw_scores(score_field)
            return NLIClient._parse_top1(
                label=str(predictions.get("label", "")),
                score=float(score_field) if score_field is not None else 0.0,
            )

        return NLIResult(label=LABEL_NEUTRAL, entailment_score=0.5, contradiction_score=0.0, neutral_score=0.5)

    @staticmethod
    def _parse_raw_scores(score_dict: dict[str, Any]) -> NLIResult:
        """Build an ``NLIResult`` from a per-class score dict (Infinity's
        ``raw_scores=True`` shape: one score per of entailment/neutral/
        contradiction, keyed by either human-readable or ``label_N`` names
        per ``_LABEL_MAP``), picking the highest-scoring class as the label.
        """
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
        """Build an ``NLIResult`` from a scalar top-1 prediction (only the
        winning label's score is known — the other two classes are left at
        0.0, unlike ``_parse_raw_scores`` which has all three).
        """
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
