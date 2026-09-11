"""NLI bounded context.

DDD layout: ``domain/`` (ports + models), ``application/`` (the factory that
builds an ``INLIModel``), ``infra/`` (HTTP + in-process adapters). Shared by RAM
(citation verification) and IVM (``nli_entailment`` relevance).

Usage:
    from app.guardrails.nli import NLIModelKind, NLIModelSpec, build_nli_model
    spec = NLIModelSpec(kind=NLIModelKind.MMBERT, base_url="http://localhost:8002", model_id="/models/mmbert_nli_id")
    nli = build_nli_model(spec)
    result = await nli.check(premise=..., hypothesis=...)
"""

from .application import build_nli_model
from .domain import (
    INLIModel,
    LABEL_CONTRADICTION,
    LABEL_ENTAILMENT,
    LABEL_NEUTRAL,
    NLI_LABEL2ID,
    NLI_LABELS,
    LabelSpace,
    NLIModelKind,
    NLIModelSpec,
    NLIResult,
)

__all__ = [
    "INLIModel",
    "LABEL_CONTRADICTION",
    "LABEL_ENTAILMENT",
    "LABEL_NEUTRAL",
    "NLI_LABEL2ID",
    "NLI_LABELS",
    "LabelSpace",
    "NLIModelKind",
    "NLIModelSpec",
    "NLIResult",
    "build_nli_model",
]
