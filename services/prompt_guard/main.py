"""Prompt-guard classification server.

A single-purpose HTTP wrapper around a sequence-classification checkpoint,
serving the IVM's safety gate.

Why this is not the shared inference server
-------------------------------------------
The guard used to be one model among several on a general-purpose inference
server. Two problems came from that, and both are structural rather than
incidental:

1. **The label mapping had to be patched from outside.** The upstream
   ``Llama-Prompt-Guard-2-86M`` config declares
   ``DebertaV2ForSequenceClassification`` but carries **no ``id2label``**, so a
   generic server falls back to ``LABEL_0``/``LABEL_1`` and every consumer has
   to know which index means what. The workaround was a patched ``config.json``
   bind-mounted over a *pinned snapshot hash* — which by construction stops
   applying the moment the model changes, i.e. exactly when swapping in the
   Indonesian fine-tune. This server resolves labels in code instead, so the
   mapping is explicit, logged, and travels with the checkpoint.

2. **Swapping the guard meant restarting everything.** Changing an 86M
   classifier forced a reload of the reranker and the NLI model that share the
   process.

Deployment note
---------------
Runs on CPU by default. The target machine's GPU is an 8 GB card already
holding the reranker, the NLI model and the in-process embedder; an 86M DeBERTa
is cheap enough on CPU that spending scarce VRAM on it is the wrong trade.

API
---
Compatible with the previous server's shape, so the application-side adapter
needs no change:

    POST /classify  {"model": "...", "input": ["text", ...]}
    -> {"object": "classify", "model": "...",
        "data": [[{"label": "BENIGN", "score": 0.99},
                  {"label": "MALICIOUS", "score": 0.01}], ...]}
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)-8s %(name)s: %(message)s"
)
logger = logging.getLogger("prompt_guard")

MODEL_ID = os.getenv("PROMPT_GUARD_MODEL", "meta-llama/Llama-Prompt-Guard-2-86M")
DEVICE = os.getenv("PROMPT_GUARD_DEVICE", "cpu")
MAX_LENGTH = int(os.getenv("PROMPT_GUARD_MAX_LENGTH", "512"))
BATCH_SIZE = int(os.getenv("PROMPT_GUARD_BATCH_SIZE", "8"))
HF_TOKEN = os.getenv("HF_TOKEN") or None

# Fallback mapping, used only when the checkpoint does not name its own labels.
# The order is the model's documented one: index 0 benign, index 1 malicious.
# It is a *fallback*, not an override — a checkpoint that names its labels wins,
# because a fine-tune could legitimately order them differently and silently
# inverting the guard is the worst failure this service can have.
FALLBACK_ID2LABEL = {0: "BENIGN", 1: "MALICIOUS"}

STATE: Dict[str, Any] = {}


def resolve_labels(config: Any) -> tuple[Dict[int, str], str]:
    """Determine the index → label mapping for the loaded checkpoint.

    Args:
        config: The loaded model config.

    Returns:
        A tuple of (mapping, provenance string describing where it came from).
    """
    raw = getattr(config, "id2label", None) or {}
    mapping = {int(k): str(v) for k, v in raw.items()}

    # transformers synthesises LABEL_0/LABEL_1 when a checkpoint names no
    # labels, so their presence means "unlabelled", not "labelled generically".
    generic = all(v.upper().startswith("LABEL_") for v in mapping.values()) if mapping else True

    if not mapping or generic:
        return dict(FALLBACK_ID2LABEL), "fallback (checkpoint declares no labels)"
    return mapping, "checkpoint config"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the classifier once at startup and keep it resident."""
    started = time.perf_counter()
    logger.info("loading model=%s device=%s", MODEL_ID, DEVICE)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, token=HF_TOKEN)
    model.eval().to(DEVICE)

    id2label, provenance = resolve_labels(model.config)

    # Logged at startup because an inverted mapping produces a guard that passes
    # attacks and blocks legitimate queries while looking perfectly healthy.
    logger.info("label mapping %s (source: %s)", id2label, provenance)
    if provenance.startswith("fallback"):
        logger.warning(
            "checkpoint did not declare id2label; assuming %s — verify with a "
            "known attack and a known benign query before trusting this",
            FALLBACK_ID2LABEL,
        )

    STATE.update(
        tokenizer=tokenizer, model=model, id2label=id2label, label_source=provenance
    )
    logger.info("ready in %.1fs", time.perf_counter() - started)
    yield
    STATE.clear()


app = FastAPI(title="Prompt Guard", version="1.0.0", lifespan=lifespan)


class ClassifyRequest(BaseModel):
    """Request body for ``/classify``."""

    input: List[str] = Field(..., description="Texts to classify")
    model: Optional[str] = Field(default=None, description="Accepted and ignored")


@app.get("/health")
async def health() -> Dict[str, Any]:
    """Report readiness, the active label mapping, and the library version.

    The version is reported because it changes the predictions. Measured on this
    checkpoint: transformers 4.57.1 and 5.9.0 disagree on 8 of Subset B's 160
    rows, moving off-the-shelf detection from 0.4250 to 0.3500 with nothing else
    altered. A container built before the pin was added therefore serves a
    measurably different classifier than the one evaluated, and nothing in its
    behaviour reveals that. Exposing the version lets a caller check rather than
    assume.
    """
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
    """List the served model, mirroring the previous server's discovery shape."""
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_ID,
                "object": "model",
                "owned_by": "prompt-guard",
                "capabilities": ["classify"],
                "backend": "torch",
            }
        ],
    }


@app.post("/classify")
async def classify(request: ClassifyRequest) -> Dict[str, Any]:
    """Classify each input text and return per-label scores.

    Args:
        request: Texts to classify.

    Returns:
        A payload whose ``data`` holds one list of label/score objects per
        input, ordered most-confident first.

    Raises:
        HTTPException: 503 while the model is still loading, 400 on empty input.
    """
    model = STATE.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail="Model still loading")
    if not request.input:
        raise HTTPException(status_code=400, detail="'input' must not be empty")

    tokenizer = STATE["tokenizer"]
    id2label = STATE["id2label"]

    results: List[List[Dict[str, Any]]] = []
    for start in range(0, len(request.input), BATCH_SIZE):
        batch = request.input[start : start + BATCH_SIZE]
        encoded = tokenizer(
            batch,
            truncation=True,
            max_length=MAX_LENGTH,
            padding=True,
            return_tensors="pt",
        ).to(DEVICE)

        with torch.no_grad():
            probabilities = torch.softmax(model(**encoded).logits, dim=-1)

        for row in probabilities.cpu().tolist():
            scored = [
                {"label": id2label.get(index, f"LABEL_{index}"), "score": float(score)}
                for index, score in enumerate(row)
            ]
            scored.sort(key=lambda item: item["score"], reverse=True)
            results.append(scored)

    return {"object": "classify", "model": MODEL_ID, "data": results}
