"""Domain models for the NLI (Natural Language Inference) subdomain.

NLI is shared by two guardrail consumers — RAM (citation verification) and the
IVM ``nli_entailment`` relevance checker — so its canonical types live here,
in their own bounded context, rather than inside either consumer. ``ram/`` and
``ivm/`` import these; nothing here imports infra.

Label order is pinned to the convention the RAM service and the deployed
``indo-roberta-indonli`` model both use (``label_0`` = entailment, ``label_1`` =
neutral, ``label_2`` = contradiction), so a fine-tuned checkpoint that swaps
labels is caught by assertion before it can silently invert the RAM verdict.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

LABEL_ENTAILMENT = "entailment"
LABEL_NEUTRAL = "neutral"
LABEL_CONTRADICTION = "contradiction"

# Canonical 3-way label order (index → label), matching indo-roberta-indonli's
# ``id2label`` and the RAM ``_pick_best`` gating.
NLI_LABELS: tuple[str, str, str] = (
    LABEL_ENTAILMENT,
    LABEL_NEUTRAL,
    LABEL_CONTRADICTION,
)

NLI_LABEL2ID: dict[str, int] = {label: i for i, label in enumerate(NLI_LABELS)}


@dataclass(frozen=True)
class NLIResult:
    """Outcome of a single NLI call: the canonical label
    (entailment/neutral/contradiction) with per-class confidence scores, plus
    the best-matching source's title/page/doc_id and a sanitized
    ``evidence_snippet`` — the exact cited chunk the check ran against, so a
    user can see which part of the context backed a claim.
    """

    label: str
    entailment_score: float = 0.0
    contradiction_score: float = 0.0
    neutral_score: float = 0.0
    source_title: str = ""
    page: Optional[int] = None
    doc_id: str = ""
    evidence_snippet: str = ""


class NLIModelKind(str, Enum):
    """The selectable NLI backends.

    - ``INDO_ROBERTA`` — ``StevenLimcorn/indo-roberta-indonli`` (3-way, current
      default, served on the shared Infinity box).
    - ``MMBERT`` — ``jhu-clsp/mmBERT-small`` fine-tuned on IndoNLI train (3-way,
      served on the dedicated CPU NLI service).
    - ``ZEROSHOT`` — ``MoritzLaurer/bge-m3-zeroshot-v2.0-c`` (binary entailment
      vs not_entailment; benchmark baseline, not a production candidate).
    """

    INDO_ROBERTA = "indo_roberta"
    MMBERT = "mmbert"
    ZEROSHOT = "zeroshot"


class LabelSpace(str, Enum):
    """The label space a backend emits.

    ``THREE_WAY`` = entailment / neutral / contradiction (RAM's native space).
    ``BINARY`` = entailment / not_entailment (the zero-shot baseline); its
    non-entailment mass is surfaced as neutral, never contradiction.
    """

    THREE_WAY = "three_way"
    BINARY = "binary"


@dataclass(frozen=True)
class NLIModelSpec:
    """The concrete configuration for building one NLI adapter.

    ``base_url`` + ``model_id`` select a server/model for the HTTP adapters;
    ``label_space`` tells the adapter how to map model output to ``NLIResult``;
    ``max_concurrency`` bounds in-flight calls; ``device``/``max_total_tokens``/
    ``max_hypothesis_tokens`` apply to the in-process adapter (benchmark).
    """

    kind: NLIModelKind
    base_url: Optional[str] = None
    model_id: str = ""
    label_space: LabelSpace = LabelSpace.THREE_WAY
    max_concurrency: int = 8
    device: str = "cpu"
    max_total_tokens: int = 500
    max_hypothesis_tokens: int = 100
