"""Ports (Protocol interfaces) for the Response Assessment Module (RAM).

The RAM validates generated LLM sentences against retrieved KB context using
NLI to detect per-sentence hallucinations. Part of the pure ``thesis`` core,
it defines its own ports so it never imports chat/kb infra directly.

Ports → adapters: :class:`INLIModel` → ``app/chat/infra/nli_client.py::NLIClient``;
:class:`IRerankerModel` → ``app/kb/infra/infinity_reranker.py::InfinityReranker``
(the same adapter fulfills the KB ``IReranker`` port; this is a narrower view
for the RAM's reverse-mapping step).
"""

from dataclasses import dataclass
from typing import List, Optional, Protocol


@dataclass(frozen=True)
class NLIResult:
    """Outcome of a single NLI call: the canonical label
    (entailment/neutral/contradiction) with per-class confidence scores, plus
    the best-matching source's title/page/doc_id and a sanitized
    ``evidence_snippet`` — the exact reranker-matched sub-passage the check ran
    against, so a user can see which part of the context backed a sentence.
    """
    label: str
    entailment_score: float = 0.0
    contradiction_score: float = 0.0
    neutral_score: float = 0.0
    source_title: str = ""
    page: Optional[int] = None
    doc_id: str = ""
    evidence_snippet: str = ""


@dataclass(frozen=True)
class RetrievedContext:
    """A retrieved context block from the knowledge base: the full parent chunk
    text with its source title, page, hierarchical breadcrumbs, structural type,
    matching child chunk id, ltree path, and source doc id.
    """
    text: str
    source_title: str
    page: Optional[int] = None
    breadcrumbs: List[str] = ()  # type: ignore[assignment]
    content_type: str = "text"
    chunk_id: str = ""
    path: str = ""
    doc_id: str = ""


class INLIModel(Protocol):
    """Port for NLI adapters in the research core. Implemented by
    ``app/chat/infra/nli_client.py::NLIClient``.
    """

    async def check(self, premise: str, hypothesis: str) -> NLIResult:
        """Compare a hypothesis against a reference premise (e.g. a KB parent
        chunk), returning an ``NLIResult`` with the label and per-class scores.
        """
        ...


@dataclass(frozen=True)
class RerankResult:
    """A reranking result: the document's original index and its relevance
    score (higher = more relevant).
    """

    index: int
    score: float


class IRerankerModel(Protocol):
    """Port for reranker adapters in the research core. Implemented by
    ``app/kb/infra/infinity_reranker.py::InfinityReranker`` (the same adapter
    fulfills the KB ``IReranker`` port).
    """

    async def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None,
    ) -> List[RerankResult]:
        """Rerank documents by relevance to the query.

        Args:
            query: The search query (e.g., hypothesis).
            documents: List of document texts (e.g., context windows).
            top_k: Optional limit on results.

        Returns:
            List of RerankResult ordered by descending relevance.
        """
        ...