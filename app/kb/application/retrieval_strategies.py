"""Retrieval strategies — how retrieved documents are finally ranked.

A strategy takes the document-level candidates produced by
``SearchService.search_documents`` (already grouped by ``doc_id`` and scored)
and returns them in final output order. This is where future "toggleable"
ranking behaviours live, selected globally via config
(``RetrievalSettings.strategy``), not per request.

Current strategies:

- ``baseline`` — identity: keep the retrieval/rerank ordering (score only).
- ``date_priority`` — same pipeline, but documents are penalised by age using
  an exponential decay, so newer documents surface first. Formula mirrors
  ``evals/probes/probe4_recency.py``: scores are shifted non-negative first
  (cross-encoder logits can be negative), then multiplied by
  ``exp(-lambda * age_years)``. Documents without a release date are left
  unpenalised rather than guessed at.
"""

import math
from dataclasses import dataclass
from typing import List, Protocol, runtime_checkable

from app.kb.domain.models import RetrievedDocument

# Default decay strength. Higher = older documents sink faster.
DEFAULT_DATE_PRIORITY_LAMBDA = 0.1


@runtime_checkable
class RetrievalStrategy(Protocol):
    """Port for a final document-ranking strategy."""

    def rank_documents(self, documents: List[RetrievedDocument]) -> List[RetrievedDocument]:
        """Return ``documents`` reordered by the strategy's ranking rule."""
        ...


class BaselineStrategy:
    """Identity strategy — keeps the retrieval/rerank ordering unchanged."""

    def rank_documents(self, documents: List[RetrievedDocument]) -> List[RetrievedDocument]:
        return documents


@dataclass
class DatePriorityStrategy:
    """Penalise older documents with exponential decay on their release date.

    The reference point is the newest release date among the candidates, so the
    strategy is corpus-independent for a single call: the newest retrieved
    document has age 0 and keeps its score, every older document decays.
    Documents missing ``released_date`` are left unpenalised (age 0).
    """

    lam: float = DEFAULT_DATE_PRIORITY_LAMBDA

    def rank_documents(self, documents: List[RetrievedDocument]) -> List[RetrievedDocument]:
        if not documents:
            return documents

        dated = [d for d in documents if d.released_date is not None]
        if not dated:
            return documents

        ref = max(d.released_date for d in dated)
        # Cross-encoder logits are frequently negative; a decay factor < 1
        # would *raise* a negative score, inverting the intended penalty.
        floor = min(d.score for d in documents)
        shift = -floor + 1e-6 if floor < 0 else 0.0

        def _decayed(doc: RetrievedDocument) -> RetrievedDocument:
            if doc.released_date is not None:
                age_years = (ref - doc.released_date).days / 365.25
                decay = math.exp(-self.lam * age_years)
            else:
                decay = 1.0
            return doc.model_copy(update={"score": (doc.score + shift) * decay})

        return sorted(
            (_decayed(doc) for doc in documents),
            key=lambda d: d.score,
            reverse=True,
        )
