"""Domain ports (Protocol interfaces) for the Knowledge Base bounded context.

The KB application/domain layers import only these Protocols; concrete adapters
live in ``app/kb/infra/`` and are injected at the composition root
(``app/kb/dependency.py``).

Ports → adapters: :class:`IDocumentParser` → ``unstructured_client.UnstructuredClient``;
:class:`ITextEmbedder` → ``bge_m3_embeddings.BGEM3Embeddings`` (or
``infinity_embeddings.InfinityEmbeddings``); :class:`IQueryExpander` →
``app/chat/infra/hyde_expander.HyDEExpander`` (implemented in the chat context,
injected across the boundary); :class:`IReranker` → ``infinity_reranker.InfinityReranker``;
:class:`IVectorStore` → ``qdrant_store.QdrantStore``; :class:`IKBRepository` →
``postgres_repo.PostgresKBRepository``.
"""

from typing import Protocol, List, Optional, Any, Dict
from dataclasses import dataclass
from datetime import datetime
from app.kb.domain.models import PDFDocument, ParentChunk, IngestionTask, ChildChunk
from app.rag.chunking.models import ParsedElement


class IDocumentParser(Protocol):
    """Port for parsing unstructured PDFs into semantic elements. Implemented
    by ``app/kb/infra/unstructured_client.py::UnstructuredClient``.
    """

    async def parse_pdf(self, file_path: str) -> List[ParsedElement]:
        """Parse a PDF into an ordered list of :class:`ParsedElement` (text,
        tables, figures).
        """
        ...

    async def close(self) -> None:
        """Release the underlying HTTP client / connection pool."""
        ...


@dataclass
class EmbeddingResult:
    """Result of an embedding operation: a dense float vector (BGE-M3,
    1024-dim) plus BM25-style sparse term indices and aligned weights.
    """

    dense: List[float]
    sparse_indices: List[int]
    sparse_values: List[float]


class ITextEmbedder(Protocol):
    """Port for generating text embeddings (dense + sparse). Implemented by
    ``app/kb/infra/bge_m3_embeddings.py::BGEM3Embeddings`` (in-process); an
    HTTP-backed alternative is ``infinity_embeddings.py::InfinityEmbeddings``.
    """

    async def embed_texts(
        self, texts: List[str], is_query: bool = False
    ) -> List[EmbeddingResult]:
        """Embed a batch of texts, returning one :class:`EmbeddingResult` per
        input in order.

        ``is_query`` marks the batch as search queries rather than corpus
        documents. Symmetric models (BGE-M3) ignore it; asymmetric ones
        (Qwen3-Embedding) prepend their instruction prefix to queries only,
        and BM25 sparse encoders skip document-side IDF weighting for queries.
        """
        ...

    async def close(self) -> None:
        """Release the model / HTTP client."""
        ...


class IQueryExpander(Protocol):
    """Port for query expansion (e.g. HyDE). Implementations generate a
    hypothetical document from the raw query. SearchService uses it only for
    the dense query representation and retains raw-query sparse weights.
    Implemented by
    ``app/chat/infra/hyde_expander.py::HyDEExpander`` (injected across the
    chat→kb boundary into ``SearchService``).
    """

    async def expand(self, query: str) -> str:
        """Generate a hypothetical answer document for the query, to be
        embedded for retrieval.
        """
        ...

    async def expand_many(self, query: str) -> List[str]:
        """Generate several hypothetical documents for a retrieval ensemble.

        Implementations may return fewer passages when individual generations
        fail. Callers must fall back safely to raw-query retrieval when the
        returned list is empty.
        """
        ...

    async def close(self) -> None:
        """Release any underlying LLM connection."""
        ...


@dataclass
class RerankResult:
    """A single reranking result: the document's original index and its
    relevance score (higher = more relevant).
    """

    index: int
    score: float


class IReranker(Protocol):
    """Port for reranking retrieved documents by relevance to a query.
    Implemented by ``app/kb/infra/infinity_reranker.py::InfinityReranker``.
    """

    async def rerank(self, query: str, documents: List[str], top_k: Optional[int] = None) -> List[RerankResult]:
        """Rerank ``documents`` (in original retrieval order) against ``query``,
        returning ``RerankResult``s in descending score, capped to ``top_k``.
        """
        ...

    async def close(self) -> None:
        """Release the underlying HTTP client."""
        ...


@dataclass
class ChunkVector:
    """A vectorized child chunk for the vector database: IDs, dense + sparse
    vectors, hierarchical breadcrumbs, structural type, optional session scope,
    and the chunk text (stored as payload for retrieval display).
    """

    chunk_id: str
    parent_chunk_id: str
    doc_id: str
    dense_vector: List[float]
    sparse_indices: List[int]
    sparse_values: List[float]
    breadcrumbs: List[str]
    content_type: str = "text"
    session_id: Optional[str] = None
    text: str = ""


@dataclass
class SearchResult:
    """A raw vector-store match: the matched child chunk, its parent chunk to
    hydrate (Small-to-Big), the source document, and the hybrid similarity score.
    """

    chunk_id: str
    parent_chunk_id: str
    doc_id: str
    score: float


class IVectorStore(Protocol):
    """Port for vector database operations (hybrid dense + sparse search).
    Implemented by ``app/kb/infra/qdrant_store.py::QdrantStore``.
    """

    async def ensure_collection(self) -> None:
        """Create the Qdrant collection if it doesn't exist (idempotent)."""
        ...

    async def upsert_chunks(self, chunks: List[ChunkVector]) -> None:
        """Upsert a batch of vectorized child chunks into the collection."""
        ...

    async def hybrid_search(
        self,
        dense_vector: List[float],
        sparse_indices: List[int],
        sparse_values: List[float],
        top_k: int = 15,
        session_id: Optional[str] = None,
        mode: str = "hybrid",
    ) -> List[SearchResult]:
        """Run a hybrid (dense + sparse) search against the collection.
        ``session_id`` restricts to a per-session collection; ``mode`` selects
        the fusion ("hybrid"/RRF, "dense", or "sparse"). Returns
        :class:`SearchResult`s ranked by hybrid score.
        """
        ...

    async def update_payload(self, doc_id: str, payload: Dict[str, Any]) -> None:
        """Update payload fields for all points of a given doc_id."""
        ...

    async def update_payloads(self, doc_ids: List[str], payload: Dict[str, Any]) -> None:
        """Update payload fields for all points of the given doc_ids (bulk)."""
        ...

    async def delete_by_doc_id(self, doc_id: str) -> None:
        """Delete all points belonging to a document (cascade on doc removal)."""
        ...

    async def delete_by_doc_ids(self, doc_ids: List[str]) -> None:
        """Delete all points belonging to the given doc_ids (bulk)."""
        ...

    async def close(self) -> None:
        """Release the underlying Qdrant client."""
        ...


class IKBRepository(Protocol):
    """Port for KB relational database operations (Postgres). Implemented by
    ``app/kb/infra/postgres_repo.py::PostgresKBRepository``.
    """

    async def get_all_pdfs(self) -> List[PDFDocument]:
        """Return all KB documents."""
        ...

    async def query_pdfs(
        self,
        search: Optional[str] = None,
        active: Optional[bool] = None,
        released_from: Optional[datetime] = None,
        released_to: Optional[datetime] = None,
        offset: int = 0,
        limit: int = 50,
    ) -> tuple[List[PDFDocument], int]:
        """Paginated + filtered document listing. Returns (items, total)."""
        ...

    async def get_pdf_by_id(self, pdf_id: str) -> Optional[PDFDocument]:
        """Fetch a single document by ID."""
        ...

    async def get_pdfs_by_ids(self, pdf_ids: List[str]) -> List[PDFDocument]:
        """Fetch multiple documents by ID (preserving order is not guaranteed)."""
        ...

    async def search_titles_naive(self, query: str) -> List[PDFDocument]:
        """Naive ILIKE title search (used to ground HyDE with active doc titles)."""
        ...

    async def create_pdf(self, title: str, description: str, pdf_path: str) -> PDFDocument:
        """Create a new KB document record."""
        ...

    async def update_pdf_active_status(self, pdf_id: str, active: bool) -> Optional[PDFDocument]:
        """Toggle a document's active flag (inactive docs are excluded from retrieval)."""
        ...

    async def bulk_update_active_status(self, pdf_ids: List[str], active: bool) -> List[str]:
        """Toggle ``active`` on many documents in a single UPDATE; returns the ids that existed."""
        ...

    async def delete_pdf(self, pdf_id: str) -> bool:
        """Delete a document and its chunks. Returns True if deleted."""
        ...

    async def bulk_delete_pdfs(self, pdf_ids: List[str]) -> List[str]:
        """Delete many documents (chunks cascade via FK) in a single DELETE; returns the ids that existed."""
        ...

    async def save_parent_chunks(self, chunks: List[ParentChunk]) -> List[ParentChunk]:
        """Persist parent chunks (full-text sections for LLM context)."""
        ...

    async def get_parent_chunks_by_ids(self, chunk_ids: List[str]) -> List[ParentChunk]:
        """Fetch parent chunks by ID (Small-to-Big hydration step)."""
        ...

    async def save_child_chunks(self, chunks: List[ChildChunk]) -> List[ChildChunk]:
        """Persist child chunks (sentence-level, indexed in Qdrant)."""
        ...

    async def get_child_chunks_by_ids(self, chunk_ids: List[str]) -> List[ChildChunk]:
        """Fetch child chunks by ID."""
        ...

    async def get_sibling_chunks(self, parent_id: str) -> List[ParentChunk]:
        """Fetch sibling parent chunks sharing the same parent (cross-ref hydration)."""
        ...

    async def get_chunks_by_path_prefix(self, path_prefix: str) -> List[ParentChunk]:
        """Fetch chunks whose ltree path starts with the given prefix."""
        ...

    async def create_ingestion_task(self, doc_id: str) -> IngestionTask:
        """Create an ingestion task record for tracking pipeline progress."""
        ...

    async def update_ingestion_task(self, task_id: str, status: str, error_message: Optional[str] = None) -> Optional[IngestionTask]:
        """Update an ingestion task's status (and optional error message)."""
        ...

    async def get_ingestion_task_by_doc_id(self, doc_id: str) -> Optional[IngestionTask]:
        """Fetch the ingestion task for a document, if any."""
        ...

    async def commit(self) -> None:
        """Commit the current transaction."""
        ...

    async def rollback(self) -> None:
        """Roll back the current transaction."""
        ...
