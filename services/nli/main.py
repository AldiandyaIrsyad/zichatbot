"""Dedicated NLI classification server (3-way sequence classification).

Serves the mmBERT fine-tune (``/models/mmbert_nli_id``) for RAM + IVM citation
verification. Split off from the shared Infinity box because mmBERT is a
ModernBERT architecture that the pinned Infinity image does not serve; a
purpose-built ``transformers`` server is the same pattern already used for the
prompt-guard classifier.

Unlike Infinity, this server accepts *premise/hypothesis pairs* so the model's
own tokenizer joins them with its separator (mmBERT's Gemma-2 tokenizer uses
``</s>``, not ``[SEP]``). The response shape is Infinity-compatible
(``{"data": [[{"label", "score"}, ...], ...]}``) so the app-side parsing is
shared.

API
---
    POST /classify  {"model": "...", "input": [{"premise": "...", "hypothesis": "..."}], "raw_scores": true}
    -> {"object": "classify", "model": "...",
        "data": [[{"label": "entailment", "score": 0.9},
                  {"label": "neutral", "score": 0.05},
                  {"label": "contradiction", "score": 0.05}], ...]}
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Union

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)-8s %(name)s: %(message)s"
)
logger = logging.getLogger("nli")

MODEL_ID = os.getenv("NLI_MODEL", "/models/mmbert_nli_id")
DEVICE = os.getenv("NLI_DEVICE", "cpu")
MAX_LENGTH = int(os.getenv("NLI_MAX_LENGTH", "512"))
BATCH_SIZE = int(os.getenv("NLI_BATCH_SIZE", "8"))
HF_TOKEN = os.getenv("HF_TOKEN") or None

# Pinned 3-way label order, matching indo-roberta-indonli and the RAM service.
# Used only when a checkpoint does not name its own labels.
FALLBACK_ID2LABEL = {0: "entailment", 1: "neutral", 2: "contradiction"}

STATE: Dict[str, Any] = {}


def resolve_labels(config: Any) -> tuple[Dict[int, str], str]:
    """Determine the index → label mapping for the loaded checkpoint."""
    raw = getattr(config, "id2label", None) or {}
    mapping = {int(k): str(v) for k, v in raw.items()}
    generic = all(v.upper().startswith("LABEL_") for v in mapping.values()) if mapping else True
    if not mapping or generic:
        return dict(FALLBACK_ID2LABEL), "fallback (checkpoint declares no labels)"
    return mapping, "checkpoint config"


@asynccontextmanager
async def lifespan(app: FastAPI):
    started = time.perf_counter()
    logger.info("loading model=%s device=%s", MODEL_ID, DEVICE)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, token=HF_TOKEN)
    model.eval().to(DEVICE)

    id2label, provenance = resolve_labels(model.config)
    logger.info("label mapping %s (source: %s)", id2label, provenance)
    if provenance.startswith("fallback"):
        logger.warning("checkpoint did not declare id2label; assuming %s", FALLBACK_ID2LABEL)

    STATE.update(tokenizer=tokenizer, model=model, id2label=id2label, label_source=provenance)
    logger.info("ready in %.1fs", time.perf_counter() - started)
    yield
    STATE.clear()


app = FastAPI(title="NLI Classifier", version="1.0.0", lifespan=lifespan)


class NliPair(BaseModel):
    premise: str
    hypothesis: str


class ClassifyRequest(BaseModel):
    """Accepts both input shapes the app's two NLI clients speak.

    ``PairClassifyNLIClient`` sends structured pairs; ``SequenceClassifyNLIClient``
    (the IndoRoBERTa path, written against Infinity) sends one pre-joined string
    per item, premise and hypothesis already separated by the model's separator
    token. Supporting both here is what lets this service replace Infinity for
    IndoRoBERTa without touching either client or changing the model.
    """

    input: List[Union[NliPair, str]] = Field(
        ..., description="Premise/hypothesis pairs, or pre-joined premise+sep+hypothesis strings"
    )
    model: Optional[str] = Field(default=None, description="Accepted and ignored")
    raw_scores: bool = Field(default=True, description="Return per-class scores")


@app.get("/health")
async def health() -> Dict[str, Any]:
    import transformers

    return {
        "status": "ok" if STATE.get("model") is not None else "loading",
        "model": MODEL_ID,
        "device": DEVICE,
        "id2label": STATE.get("id2label"),
        "label_source": STATE.get("label_source"),
        "transformers": transformers.__version__,
    }


@app.get("/models")
async def models() -> Dict[str, Any]:
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_ID,
                "object": "model",
                "owned_by": "nli",
                "capabilities": ["classify"],
                "backend": "torch",
            }
        ],
    }


@app.post("/classify")
async def classify(request: ClassifyRequest) -> Dict[str, Any]:
    model = STATE.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail="Model still loading")
    if not request.input:
        raise HTTPException(status_code=400, detail="'input' must not be empty")

    tokenizer = STATE["tokenizer"]
    id2label = STATE["id2label"]

    # Two input shapes, two tokenizations — established by matching this server's
    # output against Infinity's for the same checkpoint:
    #
    #  * pairs  -> tokenizer(premise, hypothesis) with special tokens, so the
    #    tokenizer builds the model's own pair encoding. Unchanged mmBERT path.
    #  * joined -> a single segment with add_special_tokens=False. The caller
    #    already embedded the separator, and "</s></s>" maps to real separator
    #    token ids, so adding another set duplicates them. Doing that shifted
    #    scores by up to 0.305 and flipped top-1; suppressing them lands within
    #    ~2e-3 of Infinity.
    is_joined = [isinstance(p, str) for p in request.input]
    if any(is_joined) and not all(is_joined):
        raise HTTPException(
            status_code=400,
            detail="'input' must be all pairs or all strings, not a mix — the two "
                   "shapes require different tokenization.",
        )
    joined_mode = all(is_joined)
    premises = [p if isinstance(p, str) else p.premise for p in request.input]
    hypotheses: Optional[List[str]] = (
        None if joined_mode else [p.hypothesis for p in request.input]  # type: ignore[union-attr]
    )

    all_scores: List[List[Dict[str, Any]]] = []
    with torch.no_grad():
        for start in range(0, len(request.input), BATCH_SIZE):
            batch_premises = premises[start : start + BATCH_SIZE]
            batch_hypotheses = (
                None if hypotheses is None else hypotheses[start : start + BATCH_SIZE]
            )
            enc = tokenizer(
                batch_premises,
                batch_hypotheses,
                truncation=True,
                max_length=MAX_LENGTH,
                padding=True,
                return_tensors="pt",
                add_special_tokens=not joined_mode,
            ).to(model.device)
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu().tolist()
            for row_probs in probs:
                # Preserve the checkpoint's label order (softmax index i ↔ id2label[i]).
                scores = [
                    {"label": id2label.get(i, f"label_{i}"), "score": float(row_probs[i])}
                    for i in range(len(row_probs))
                ]
                scores.sort(key=lambda s: s["score"], reverse=True)
                all_scores.append(scores)

    return {"object": "classify", "model": MODEL_ID, "data": all_scores}
