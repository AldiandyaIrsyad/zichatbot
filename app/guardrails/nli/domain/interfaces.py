"""Ports (Protocol interfaces) for the NLI subdomain.

``INLIModel`` is the single port RAM and IVM both depend on; concrete adapters
live in ``app/guardrails/nli/infra/`` and are built by
``app/guardrails/nli/application/selector.py::build_nli_model``.
"""

from __future__ import annotations

from typing import Protocol

from .models import NLIResult


class INLIModel(Protocol):
    """Port for NLI adapters.

    Implementations classify the entailment relation between a ``premise``
    (retrieved KB evidence) and a ``hypothesis`` (a generated claim / query) and
    return an :class:`NLIResult` with the label and per-class scores.
    """

    async def check(self, premise: str, hypothesis: str) -> NLIResult:
        """Compare a hypothesis against a reference premise, returning an
        ``NLIResult`` with the label and per-class scores.
        """
        ...
