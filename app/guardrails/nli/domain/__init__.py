"""NLI domain package — ports and models only (no infra imports)."""

from .interfaces import INLIModel
from .models import (
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
]
