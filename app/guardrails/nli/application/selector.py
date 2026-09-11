"""NLI application layer — the factory that builds an ``INLIModel`` from a spec.

Keeps infra (``httpx``, ``transformers``, ``torch``) out of the domain: adapters
are imported lazily inside the builder functions, so importing this module
(and the rest of ``nli/``) never pulls in a heavy runtime dependency.
"""

from __future__ import annotations

from typing import Dict

from ..domain.interfaces import INLIModel
from ..domain.models import NLIModelKind, NLIModelSpec


def _build_sequence_classify(spec: NLIModelSpec) -> INLIModel:
    """HTTP adapter for a 3-way sequence-classification server (IndoRoBERTa
    on Infinity)."""
    from ..infra.sequence_classify_client import SequenceClassifyNLIClient

    return SequenceClassifyNLIClient(
        base_url=spec.base_url or "",
        model=spec.model_id,
        max_concurrency=spec.max_concurrency,
    )


def _build_pair_classify(spec: NLIModelSpec) -> INLIModel:
    """HTTP adapter for the dedicated ``services/nli`` pair-input server
    (the mmBERT fine-tune)."""
    from ..infra.pair_classify_client import PairClassifyNLIClient

    return PairClassifyNLIClient(
        base_url=spec.base_url or "",
        model=spec.model_id,
        max_concurrency=spec.max_concurrency,
    )


def _build_zeroshot(spec: NLIModelSpec) -> INLIModel:
    """HTTP adapter for the binary zero-shot baseline."""
    from ..infra.zeroshot_client import ZeroshotNLIClient

    return ZeroshotNLIClient(
        base_url=spec.base_url or "",
        model=spec.model_id,
        max_concurrency=spec.max_concurrency,
    )


def _build_local(spec: NLIModelSpec) -> INLIModel:
    """In-process ``transformers`` adapter (benchmark / training eval)."""
    from ..infra.local_client import LocalNLIClient

    return LocalNLIClient(
        model_id=spec.model_id,
        device=spec.device,
        label_space=spec.label_space,
        max_total_tokens=spec.max_total_tokens,
        max_hypothesis_tokens=spec.max_hypothesis_tokens,
    )


# kind → HTTP builder. The local (in-process) adapter is selected via
# ``backend="local"`` rather than by kind, since the benchmark loads every model
# locally regardless of how production serves it.
_HTTP_BUILDERS: Dict[NLIModelKind, object] = {
    NLIModelKind.INDO_ROBERTA: _build_sequence_classify,
    NLIModelKind.MMBERT: _build_pair_classify,
    NLIModelKind.ZEROSHOT: _build_zeroshot,
}


def build_nli_model(spec: NLIModelSpec, backend: str = "http") -> INLIModel:
    """Build the ``INLIModel`` described by ``spec``.

    ``backend="http"`` builds the production HTTP adapter (selected by
    ``spec.kind``); ``backend="local"`` builds the in-process transformers
    adapter used by the benchmark and training evaluation.
    """
    if backend == "local":
        return _build_local(spec)
    builder = _HTTP_BUILDERS[spec.kind]
    return builder(spec)
