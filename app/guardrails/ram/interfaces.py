"""Ports (Protocol interfaces) for the Response Assessment Module (RAM).

The RAM validates generated LLM sentences against retrieved KB context using
NLI to detect per-sentence hallucinations. Part of the pure ``thesis`` core,
it defines its own ports so it never imports chat/kb infra directly.

``INLIModel`` and ``NLIResult`` live in the shared NLI bounded context
(``app.guardrails.nli``) and are re-exported here for backward compatibility —
RAM is one of two consumers (the IVM ``nli_entailment`` relevance checker is the
other).
"""

from dataclasses import dataclass
from typing import List, Optional, Protocol, Tuple

from app.guardrails.nli.domain.interfaces import INLIModel
from app.guardrails.nli.domain.models import NLIResult

# Re-exported so existing ``from app.guardrails.ram.interfaces import INLIModel,
# NLIResult`` import sites keep working unchanged.
__all__ = [
    "ClaimUnit",
    "INLIModel",
    "IRerankerModel",
    "NLIResult",
    "RerankResult",
    "RetrievedContext",
]


@dataclass(frozen=True)
class RerankResult:
    """A reranking result: the document's original index and its relevance
    score (higher = more relevant).
    """

    index: int
    score: float


class IRerankerModel(Protocol):
    """Port for reranker adapters in the research core.

    Used to locate the sub-passage of a *cited* chunk that an NLI premise
    should be built from. Structurally identical to the KB's ``IReranker``, so
    ``app/kb/infra/infinity_reranker.py::InfinityReranker`` satisfies both;
    declared here so ``ram/`` never imports ``kb/``.
    """

    async def rerank(
        self, query: str, documents: List[str], top_k: Optional[int] = None
    ) -> List[RerankResult]:
        """Rerank ``documents`` against ``query``, best first, capped to
        ``top_k``."""
        ...


@dataclass(frozen=True)
class RetrievedContext:
    """A retrieved context block from the knowledge base: the full parent chunk
    text with its source title, page, hierarchical breadcrumbs, structural type,
    matching child chunk id, ltree path, and source doc id. ``released_date``
    carries the document's release date (ISO string) when known, for
    document-level prompts/citations.
    """
    text: str
    source_title: str
    page: Optional[int] = None
    breadcrumbs: List[str] = ()  # type: ignore[assignment]
    content_type: str = "text"
    chunk_id: str = ""
    path: str = ""
    doc_id: str = ""
    released_date: Optional[str] = None
    # Sentence/row-group child text (small-to-big). For table chunks this is
    # the header + row group that matched retrieval; for text chunks it is the
    # matching sentence. Citation-local evidence prefers it when present.
    child_text: str = ""
    parent_chunk_id: str = ""


@dataclass(frozen=True)
class ClaimUnit:
    """One claim extracted from the generated Markdown, ready for verification.

    ``text`` is the marker-stripped claim; ``citation_ids`` are 1-based
    indices into the retrieved-context list shown to the LLM as ``Sumber N``.
    ``kind`` is ``prose``/``list_item``/``table_row``; ``table_row`` units are
    passthrough (accumulated by the caller until the table block completes and
    then cell-parsed), while prose/list_item units are assessed directly.
    ``separator`` is the original trailing whitespace to re-emit after the
    claim so formatting survives streaming.

    ``is_sentence_end`` is False for every clause of a multi-clause sentence
    except the last. Each clause is still verified on its own (that is the
    point of clause splitting), but only the sentence-final unit closes the
    sentence with a full stop and carries the badge, so a verdict badge never
    lands mid-sentence.
    """
    text: str
    citation_ids: Tuple[int, ...] = ()
    kind: str = "prose"
    separator: str = ""
    is_sentence_end: bool = True