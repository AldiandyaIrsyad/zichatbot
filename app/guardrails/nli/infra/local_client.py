"""In-process NLI adapter for the benchmark and training evaluation.

Loads a sequence-classification checkpoint with ``transformers`` and runs
inference on the local device (no HTTP server). This lets the IndoNLI benchmark
score all three models without spinning up three serving containers.

Supports both label spaces:
- ``THREE_WAY`` — a 3-way checkpoint (mmBERT fine-tune, indo-roberta-indonli).
- ``BINARY`` — ``bge-m3-zeroshot-v2.0-c``'s entailment/not_entailment head.

The ``transformers`` forward pass is synchronous, so ``check`` offloads it to a
worker thread via ``asyncio.to_thread`` (matches the RAM's async contract without
blocking the event loop).
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any, Optional

import structlog

from app.guardrails.nli.domain.interfaces import INLIModel
from app.guardrails.nli.domain.models import (
    LABEL_CONTRADICTION,
    LABEL_ENTAILMENT,
    LABEL_NEUTRAL,
    LabelSpace,
    NLIResult,
)

logger = structlog.get_logger(__name__)


class LocalNLIClient(INLIModel):
    """In-process ``transformers`` NLI adapter (benchmark / training eval)."""

    def __init__(
        self,
        model_id: str,
        device: str = "cpu",
        label_space: LabelSpace = LabelSpace.THREE_WAY,
        max_total_tokens: int = 500,
        max_hypothesis_tokens: int = 100,
    ):
        self.model_id = model_id
        self.device = device
        self.label_space = label_space
        self.max_total_tokens = max_total_tokens
        self.max_hypothesis_tokens = max_hypothesis_tokens

        # Lazily loaded on first use (transformers/torch are heavy).
        self._model: Any = None
        self._tokenizer: Any = None
        self._id2label: dict[int, str] = {}
        self._load_lock = threading.Lock()

    def _load(self):
        """Load tokenizer + model on first use (never at import)."""
        if self._model is None:
            with self._load_lock:
                if self._model is None:
                    import torch
                    from transformers import (
                        AutoModelForSequenceClassification,
                        AutoTokenizer,
                    )

                    tokenizer = AutoTokenizer.from_pretrained(self.model_id)
                    if tokenizer.pad_token is None:
                        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

                    num_labels = 3 if self.label_space == LabelSpace.THREE_WAY else 2
                    model = AutoModelForSequenceClassification.from_pretrained(
                        self.model_id,
                        num_labels=num_labels,
                        ignore_mismatched_sizes=True,
                    )
                    model.eval().to(self.device)

                    self._tokenizer = tokenizer
                    self._model = model
                    self._id2label = {
                        int(k): str(v) for k, v in (model.config.id2label or {}).items()
                    }
                    logger.info(
                        "nli.local.loaded",
                        model=self.model_id,
                        device=self.device,
                        num_labels=num_labels,
                        id2label=self._id2label,
                    )
        return self._model, self._tokenizer, self._id2label

    async def check(self, premise: str, hypothesis: str) -> NLIResult:
        """Run NLI in-process, offloading the sync forward pass to a thread."""
        model, tokenizer, id2label = self._load()
        return await asyncio.to_thread(
            self._check_sync, model, tokenizer, id2label, premise, hypothesis
        )

    def _check_sync(self, model, tokenizer, id2label, premise: str, hypothesis: str) -> NLIResult:
        import torch

        encoded = tokenizer(
            premise,
            hypothesis,
            truncation=True,
            max_length=self.max_total_tokens,
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            logits = model(**encoded).logits
        probabilities = torch.softmax(logits, dim=-1)[0].cpu().tolist()
        return self._to_result(probabilities, id2label)

    def _to_result(self, probabilities: list[float], id2label: dict[int, str]) -> NLIResult:
        """Map per-class probabilities to a canonical ``NLIResult``.

        For the 3-way space this is a direct entailment/neutral/contradiction
        mapping. For the binary space, ``not_entailment`` mass is surfaced as
        ``neutral_score`` (never contradiction) — the zero-shot baseline was not
        trained to distinguish those two.
        """
        scores: dict[str, float] = {LABEL_ENTAILMENT: 0.0, LABEL_NEUTRAL: 0.0, LABEL_CONTRADICTION: 0.0}
        for idx, prob in enumerate(probabilities):
            raw = id2label.get(idx, f"label_{idx}").lower()
            if raw in ("entailment", "label_0"):
                scores[LABEL_ENTAILMENT] += prob
            elif raw in ("contradiction", "label_2"):
                scores[LABEL_CONTRADICTION] += prob
            else:  # neutral, not_entailment, label_1
                scores[LABEL_NEUTRAL] += prob

        label = max(scores, key=scores.get)
        return NLIResult(
            label=label,
            entailment_score=scores[LABEL_ENTAILMENT],
            contradiction_score=scores[LABEL_CONTRADICTION],
            neutral_score=scores[LABEL_NEUTRAL],
        )

    async def close(self) -> None:
        """Release the model (free VRAM) if it was loaded."""
        if self._model is not None:
            del self._model
            del self._tokenizer
            self._model = None
            self._tokenizer = None
