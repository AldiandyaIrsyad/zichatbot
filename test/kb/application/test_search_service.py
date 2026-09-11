"""Tests for the SearchService 6-step retrieval pipeline.

Tests cover:
    - Cross-reference regex extraction (Pasal, BAB, Ayat patterns)
    - Merge + deduplication logic
    - Full 6-step pipeline with mocked infra (embed → search → fetch → rerank → hydrate → merge)
    - Sibling hydration
    - Cross-reference detection and fetching
    - Fallback path when child chunks are not persisted
    - HyDE query expansion integration
    - Empty query handling
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

from app.kb.application.search_service import SearchService
from app.kb.application.retrieval_strategies import DatePriorityStrategy
from app.kb.domain.interfaces import (
    EmbeddingResult,
    SearchResult,
    RerankResult,
)
from app.kb.domain.models import (
    ChildChunk,
    ParentChunk,
    PDFDocument,
    RetrievedContext,
    RetrievedDocument,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_embedder() -> MagicMock:
    """Create a mock ITextEmbedder."""
    emb = MagicMock()
    emb.embed_texts = AsyncMock(
        return_value=[
            EmbeddingResult(
                dense=[0.1, 0.2, 0.3],
                sparse_indices=[1, 2, 3],
                sparse_values=[0.5, 0.4, 0.3],
            )
        ]
    )
    emb.close = AsyncMock()
    return emb


def _make_vector_store(results: list[SearchResult]) -> MagicMock:
    """Create a mock IVectorStore returning the given search results."""
    vs = MagicMock()
    vs.hybrid_search = AsyncMock(return_value=results)
    vs.close = AsyncMock()
    return vs


def _make_parent(
    pid: str = "p1",
    doc_id: str = "doc1",
    text: str = "Parent text",
    parent_id: str | None = None,
    path: str = "root.p1",
    depth: int = 0,
    ordinal: int = 0,
    breadcrumbs: list[str] | None = None,
    content_type: str = "text",
    page: int | None = 1,
) -> MagicMock:
    """Create a mock ParentChunk with the given attributes."""
    pc = MagicMock(spec=ParentChunk)
    pc.id = pid
    pc.doc_id = doc_id
    pc.text = text
    pc.chunk_index = 0
    pc.page = page
    pc.breadcrumbs = breadcrumbs or []
    pc.content_type = content_type
    pc.element_metadata = {}
    pc.parent_id = parent_id
    pc.ordinal = ordinal
    pc.path = path
    pc.depth = depth
    return pc


def _make_child(
    cid: str = "c1",
    parent_id: str = "p1",
    doc_id: str = "doc1",
    text: str = "Child text",
    ordinal: int = 0,
    path: str = "root.p1.c0",
    content_type: str = "text",
) -> MagicMock:
    """Create a mock ChildChunk with the given attributes."""
    cc = MagicMock(spec=ChildChunk)
    cc.id = cid
    cc.parent_chunk_id = parent_id
    cc.doc_id = doc_id
    cc.text = text
    cc.ordinal = ordinal
    cc.path = path
    cc.page = 1
    cc.content_type = content_type
    return cc


def _make_pdf(
    doc_id: str = "doc1",
    title: str = "Test Doc",
    released_date: datetime | None = None,
) -> MagicMock:
    """Create a mock PDFDocument with the given attributes."""
    pdf = MagicMock(spec=PDFDocument)
    pdf.id = doc_id
    pdf.title = title
    pdf.description = ""
    pdf.pdf_path = "/fake/path.pdf"
    pdf.is_active = True
    pdf.released_date = released_date
    return pdf


def _make_kb_repo(
    child_chunks: list[ChildChunk] | None = None,
    parent_chunks: list[ParentChunk] | None = None,
    pdf_docs: list[PDFDocument] | None = None,
    siblings: list[ParentChunk] | None = None,
    cross_refs: list[ParentChunk] | None = None,
) -> MagicMock:
    """Create a mock IKBRepository."""
    repo = MagicMock()
    repo.get_child_chunks_by_ids = AsyncMock(return_value=child_chunks or [])
    repo.get_parent_chunks_by_ids = AsyncMock(return_value=parent_chunks or [])
    repo.get_pdfs_by_ids = AsyncMock(return_value=pdf_docs or [])
    repo.get_sibling_chunks = AsyncMock(return_value=siblings or [])
    repo.get_chunks_by_path_prefix = AsyncMock(return_value=cross_refs or [])
    return repo


def _make_reranker(results: list[RerankResult] | None = None) -> MagicMock:
    """Create a mock IReranker."""
    rr = MagicMock()
    rr.rerank = AsyncMock(return_value=results or [])
    rr.close = AsyncMock()
    return rr


# ---------------------------------------------------------------------------
# Cross-reference extraction
# ---------------------------------------------------------------------------

class TestExtractCrossReferences:
    """Tests for SearchService._extract_cross_references()."""

    def test_pasal_reference(self) -> None:
        svc = SearchService(
            text_embedder=_make_embedder(),
            vector_store=_make_vector_store([]),
            kb_repo=_make_kb_repo(),
        )
        prefixes = svc._extract_cross_references("Lihat Pasal 5 untuk detail.")
        assert "pasal_5" in prefixes

    def test_bab_roman_reference(self) -> None:
        svc = SearchService(
            text_embedder=_make_embedder(),
            vector_store=_make_vector_store([]),
            kb_repo=_make_kb_repo(),
        )
        prefixes = svc._extract_cross_references("Sesuai BAB II tentang kewajiban.")
        assert "bab_ii" in prefixes

    def test_bab_arabic_reference(self) -> None:
        svc = SearchService(
            text_embedder=_make_embedder(),
            vector_store=_make_vector_store([]),
            kb_repo=_make_kb_repo(),
        )
        prefixes = svc._extract_cross_references("Dalam BAB 3 dijelaskan.")
        assert "bab_3" in prefixes

    def test_ayat_reference(self) -> None:
        svc = SearchService(
            text_embedder=_make_embedder(),
            vector_store=_make_vector_store([]),
            kb_repo=_make_kb_repo(),
        )
        prefixes = svc._extract_cross_references("Pada Ayat (3) disebutkan.")
        assert "ayat_3" in prefixes

    def test_multiple_references(self) -> None:
        svc = SearchService(
            text_embedder=_make_embedder(),
            vector_store=_make_vector_store([]),
            kb_repo=_make_kb_repo(),
        )
        text = "Pasal 5 dan BAB II serta Ayat 3 saling terkait."
        prefixes = svc._extract_cross_references(text)
        assert "pasal_5" in prefixes
        assert "bab_ii" in prefixes
        assert "ayat_3" in prefixes

    def test_no_references(self) -> None:
        svc = SearchService(
            text_embedder=_make_embedder(),
            vector_store=_make_vector_store([]),
            kb_repo=_make_kb_repo(),
        )
        prefixes = svc._extract_cross_references("Tidak ada referensi di sini.")
        assert prefixes == []

    def test_empty_text(self) -> None:
        svc = SearchService(
            text_embedder=_make_embedder(),
            vector_store=_make_vector_store([]),
            kb_repo=_make_kb_repo(),
        )
        assert svc._extract_cross_references("") == []

    def test_case_insensitive_pasal(self) -> None:
        svc = SearchService(
            text_embedder=_make_embedder(),
            vector_store=_make_vector_store([]),
            kb_repo=_make_kb_repo(),
        )
        prefixes = svc._extract_cross_references("PASAL 12 jelas.")
        assert "pasal_12" in prefixes


# ---------------------------------------------------------------------------
# Merge + dedupe
# ---------------------------------------------------------------------------

class TestMergeAndDedupe:
    """Tests for SearchService._merge_and_dedupe()."""

    def test_primary_first_then_siblings_then_crossrefs(self) -> None:
        svc = SearchService(
            text_embedder=_make_embedder(),
            vector_store=_make_vector_store([]),
            kb_repo=_make_kb_repo(),
        )
        primary = [
            RetrievedContext(chunk_id="c1", parent_chunk_id="p1", doc_id="d1", text="A", score=0.9),
        ]
        siblings = [
            RetrievedContext(chunk_id="s1", parent_chunk_id="p2", doc_id="d1", text="B", score=0.0),
        ]
        cross_refs = [
            RetrievedContext(chunk_id="x1", parent_chunk_id="p3", doc_id="d1", text="C", score=0.0),
        ]
        result = svc._merge_and_dedupe(primary, siblings, cross_refs)
        assert len(result) == 3
        assert result[0].text == "A"
        assert result[1].text == "B"
        assert result[2].text == "C"

    def test_dedup_by_parent_chunk_id(self) -> None:
        svc = SearchService(
            text_embedder=_make_embedder(),
            vector_store=_make_vector_store([]),
            kb_repo=_make_kb_repo(),
        )
        primary = [
            RetrievedContext(chunk_id="c1", parent_chunk_id="p1", doc_id="d1", text="A", score=0.9),
        ]
        siblings = [
            RetrievedContext(chunk_id="s1", parent_chunk_id="p1", doc_id="d1", text="B", score=0.0),
        ]
        result = svc._merge_and_dedupe(primary, siblings, [])
        assert len(result) == 1
        assert result[0].text == "A"  # Primary wins

    def test_empty_inputs(self) -> None:
        svc = SearchService(
            text_embedder=_make_embedder(),
            vector_store=_make_vector_store([]),
            kb_repo=_make_kb_repo(),
        )
        assert svc._merge_and_dedupe([], [], []) == []


# ---------------------------------------------------------------------------
# Full 6-step pipeline
# ---------------------------------------------------------------------------

class TestSearchPipeline:
    """Tests for the full 6-step SearchService.search() pipeline."""

    @pytest.mark.asyncio
    async def test_empty_query_returns_empty(self) -> None:
        svc = SearchService(
            text_embedder=_make_embedder(),
            vector_store=_make_vector_store([]),
            kb_repo=_make_kb_repo(),
        )
        result = await svc.search("")
        assert result == []

    @pytest.mark.asyncio
    async def test_whitespace_query_returns_empty(self) -> None:
        svc = SearchService(
            text_embedder=_make_embedder(),
            vector_store=_make_vector_store([]),
            kb_repo=_make_kb_repo(),
        )
        result = await svc.search("   ")
        assert result == []

    @pytest.mark.asyncio
    async def test_no_search_results_returns_empty(self) -> None:
        svc = SearchService(
            text_embedder=_make_embedder(),
            vector_store=_make_vector_store([]),
            kb_repo=_make_kb_repo(),
        )
        result = await svc.search("query")
        assert result == []

    @pytest.mark.asyncio
    async def test_full_pipeline_with_reranker(self) -> None:
        """Test the full 6-step pipeline with reranker enabled."""
        search_results = [
            SearchResult(chunk_id="c1", parent_chunk_id="p1", doc_id="doc1", score=0.8),
            SearchResult(chunk_id="c2", parent_chunk_id="p1", doc_id="doc1", score=0.6),
        ]
        children = [
            _make_child(cid="c1", parent_id="p1", text="Child 1 text"),
            _make_child(cid="c2", parent_id="p1", text="Child 2 text"),
        ]
        parents = [_make_parent(pid="p1", text="Full parent text", path="root.p1")]
        pdfs = [_make_pdf(doc_id="doc1", title="Test PDF")]

        reranker = _make_reranker([
            RerankResult(index=1, score=0.95),  # Reorder: child 2 first
            RerankResult(index=0, score=0.80),
        ])

        svc = SearchService(
            text_embedder=_make_embedder(),
            vector_store=_make_vector_store(search_results),
            kb_repo=_make_kb_repo(
                child_chunks=children,
                parent_chunks=parents,
                pdf_docs=pdfs,
            ),
            reranker=reranker,
        )

        result = await svc.search("query", top_k=5)

        assert len(result) >= 1
        assert result[0].text == "Full parent text"
        assert result[0].source_title == "Test PDF"
        assert result[0].child_text == "Child 2 text"  # Reranked to first
        assert result[0].path == "root.p1"
        assert result[0].breadcrumbs == []

    @pytest.mark.asyncio
    async def test_reranker_success_path_truncates_to_rerank_top_k(self) -> None:
        """Even when the reranker returns more results than RERANK_TOP_K (the
        deployed Infinity server has been observed to ignore top_k and return
        every candidate reranked), the success path must still cap the pool
        at RERANK_TOP_K before sibling/cross-ref hydration."""
        from app.kb.application.search_service import RERANK_TOP_K

        n = RERANK_TOP_K + 5
        search_results = [
            SearchResult(chunk_id=f"c{i}", parent_chunk_id=f"p{i}", doc_id="doc1", score=0.9 - i * 0.01)
            for i in range(n)
        ]
        children = [
            _make_child(cid=f"c{i}", parent_id=f"p{i}", text=f"Child {i}")
            for i in range(n)
        ]
        parents = [_make_parent(pid=f"p{i}", text=f"Parent {i}") for i in range(n)]
        pdfs = [_make_pdf()]

        # Reranker "ignores" top_k and returns all n results reranked.
        reranker = _make_reranker([
            RerankResult(index=i, score=1.0 - i * 0.01) for i in range(n)
        ])

        svc = SearchService(
            text_embedder=_make_embedder(),
            vector_store=_make_vector_store(search_results),
            kb_repo=_make_kb_repo(
                child_chunks=children,
                parent_chunks=parents,
                pdf_docs=pdfs,
            ),
            reranker=reranker,
        )

        result = await svc.search("query", top_k=n)
        # Primary contexts must be capped at RERANK_TOP_K, not the full candidate pool.
        assert len(result) <= RERANK_TOP_K

    @pytest.mark.asyncio
    async def test_pipeline_without_reranker(self) -> None:
        """Pipeline works without reranker — truncates to RERANK_TOP_K."""
        search_results = [
            SearchResult(chunk_id=f"c{i}", parent_chunk_id="p1", doc_id="doc1", score=0.8 - i * 0.1)
            for i in range(3)
        ]
        children = [
            _make_child(cid=f"c{i}", parent_id="p1", text=f"Child {i}")
            for i in range(3)
        ]
        parents = [_make_parent(pid="p1", text="Parent")]
        pdfs = [_make_pdf()]

        svc = SearchService(
            text_embedder=_make_embedder(),
            vector_store=_make_vector_store(search_results),
            kb_repo=_make_kb_repo(
                child_chunks=children,
                parent_chunks=parents,
                pdf_docs=pdfs,
            ),
        )

        result = await svc.search("query", top_k=5)
        assert len(result) >= 1
        assert result[0].text == "Parent"

    @pytest.mark.asyncio
    async def test_fallback_when_no_child_chunks(self) -> None:
        """When child chunks are not persisted, fallback to parent text directly."""
        search_results = [
            SearchResult(chunk_id="c1", parent_chunk_id="p1", doc_id="doc1", score=0.9),
        ]
        # No child chunks returned
        parents = [_make_parent(pid="p1", text="Fallback parent text")]
        pdfs = [_make_pdf()]

        svc = SearchService(
            text_embedder=_make_embedder(),
            vector_store=_make_vector_store(search_results),
            kb_repo=_make_kb_repo(
                child_chunks=[],
                parent_chunks=parents,
                pdf_docs=pdfs,
            ),
        )

        result = await svc.search("query", top_k=5)
        assert len(result) == 1
        assert result[0].text == "Fallback parent text"

    @pytest.mark.asyncio
    async def test_sibling_hydration(self) -> None:
        """Siblings are fetched and included in results."""
        search_results = [
            SearchResult(chunk_id="c1", parent_chunk_id="p1", doc_id="doc1", score=0.9),
        ]
        children = [_make_child(cid="c1", parent_id="p1", text="Child 1")]
        parent1 = _make_parent(pid="p1", text="Parent 1", parent_id="root", path="root.p1")
        parent2 = _make_parent(
            pid="p2",
            # Long enough to clear MIN_HYDRATED_CHARS — a heading-only chunk
            # carries no evidence and is filtered out of hydration.
            text="Sibling text. " + "Ketentuan lanjutan pada bagian ini. " * 8,
            parent_id="root",
            path="root.p2",
        )
        parents = [parent1]
        siblings = [parent2]
        pdfs = [_make_pdf()]

        svc = SearchService(
            text_embedder=_make_embedder(),
            vector_store=_make_vector_store(search_results),
            kb_repo=_make_kb_repo(
                child_chunks=children,
                parent_chunks=parents,
                pdf_docs=pdfs,
                siblings=siblings,
            ),
        )

        result = await svc.search("query", top_k=10)
        # Should include both primary and sibling
        texts = [r.text for r in result]
        assert "Parent 1" in texts
        assert any(t.startswith("Sibling text") for t in texts)

    @pytest.mark.asyncio
    async def test_cross_ref_detection(self) -> None:
        """Cross-references in child text are detected and fetched."""
        search_results = [
            SearchResult(chunk_id="c1", parent_chunk_id="p1", doc_id="doc1", score=0.9),
        ]
        # Child text contains a cross-reference
        children = [
            _make_child(cid="c1", parent_id="p1", text="Lihat Pasal 5 untuk detail."),
        ]
        parent1 = _make_parent(pid="p1", text="Parent 1 text", path="root.p1")
        parents = [parent1]
        # Cross-ref result
        cross_ref_parent = _make_parent(
            pid="p5",
            # Long enough to clear MIN_HYDRATED_CHARS (see sibling test).
            text="Pasal 5 content. " + "Isi ketentuan Pasal 5 selengkapnya. " * 8,
            path="pasal_5",
        )
        pdfs = [_make_pdf()]

        svc = SearchService(
            text_embedder=_make_embedder(),
            vector_store=_make_vector_store(search_results),
            kb_repo=_make_kb_repo(
                child_chunks=children,
                parent_chunks=parents,
                pdf_docs=pdfs,
                cross_refs=[cross_ref_parent],
            ),
        )

        result = await svc.search("query", top_k=10)
        texts = [r.text for r in result]
        assert "Parent 1 text" in texts
        assert any(t.startswith("Pasal 5 content") for t in texts)

    @pytest.mark.asyncio
    async def test_reranker_failure_truncates_gracefully(self) -> None:
        """When reranker fails, pipeline truncates to RERANK_TOP_K by original order."""
        search_results = [
            SearchResult(chunk_id=f"c{i}", parent_chunk_id="p1", doc_id="doc1", score=0.9 - i * 0.1)
            for i in range(3)
        ]
        children = [
            _make_child(cid=f"c{i}", parent_id="p1", text=f"Child {i}")
            for i in range(3)
        ]
        parents = [_make_parent(pid="p1", text="Parent")]
        pdfs = [_make_pdf()]

        reranker = MagicMock()
        reranker.rerank = AsyncMock(side_effect=Exception("Reranker down"))

        svc = SearchService(
            text_embedder=_make_embedder(),
            vector_store=_make_vector_store(search_results),
            kb_repo=_make_kb_repo(
                child_chunks=children,
                parent_chunks=parents,
                pdf_docs=pdfs,
            ),
            reranker=reranker,
        )

        result = await svc.search("query", top_k=5)
        # Should still return results (truncated to RERANK_TOP_K=8, but only 3 candidates)
        assert len(result) >= 1


# ---------------------------------------------------------------------------
# HyDE integration
# ---------------------------------------------------------------------------

class TestHyDEIntegration:
    """Tests for HyDE query expansion in the search pipeline."""

    @pytest.mark.asyncio
    async def test_hyde_expands_query(self) -> None:
        """HyDE changes dense embedding but keeps raw-query sparse features."""
        expander = MagicMock()
        expander.expand_many = AsyncMock(return_value=[
            "Hypothetical answer document one.",
            "Hypothetical answer document two.",
            "Hypothetical answer document three.",
        ])
        expander.close = AsyncMock()

        emb = _make_embedder()
        search_results = [
            SearchResult(chunk_id="c1", parent_chunk_id="p1", doc_id="doc1", score=0.9),
        ]
        children = [_make_child(cid="c1", parent_id="p1", text="Child")]
        parents = [_make_parent(pid="p1", text="Parent")]
        pdfs = [_make_pdf()]

        svc = SearchService(
            text_embedder=emb,
            vector_store=_make_vector_store(search_results),
            kb_repo=_make_kb_repo(
                child_chunks=children,
                parent_chunks=parents,
                pdf_docs=pdfs,
            ),
            query_expander=expander,
        )

        await svc.search("original query", top_k=5)

        expander.expand_many.assert_called_once_with("original query")
        # Raw-query retrieval is retained, then fused with dense-only mean HyDE retrieval.
        assert emb.embed_texts.await_args_list[0].args == (["original query"],)
        assert emb.embed_texts.await_args_list[1].args == ([
            "Hypothetical answer document one.",
            "Hypothetical answer document two.",
            "Hypothetical answer document three.",
        ],)
        vector_store = svc.vector_store
        assert vector_store.hybrid_search.await_count == 2
        raw_kwargs = vector_store.hybrid_search.await_args_list[0].kwargs
        hyde_kwargs = vector_store.hybrid_search.await_args_list[1].kwargs
        assert raw_kwargs["dense_vector"] == [0.1, 0.2, 0.3]
        assert raw_kwargs["sparse_indices"] == [1, 2, 3]
        assert hyde_kwargs["mode"] == "dense"
        assert hyde_kwargs["dense_vector"] == pytest.approx([0.26726124, 0.53452248, 0.80178373])

    def test_rrf_fusion_rewards_agreement_without_dropping_unique_candidates(self) -> None:
        first = [
            SearchResult(chunk_id="a", parent_chunk_id="pa", doc_id="da", score=1.0),
            SearchResult(chunk_id="b", parent_chunk_id="pb", doc_id="db", score=0.9),
        ]
        second = [
            SearchResult(chunk_id="b", parent_chunk_id="pb", doc_id="db", score=1.0),
            SearchResult(chunk_id="c", parent_chunk_id="pc", doc_id="dc", score=0.9),
        ]
        fused = SearchService._rrf_fuse([first, second])
        assert [item.chunk_id for item in fused] == ["b", "a", "c"]

    @pytest.mark.asyncio
    async def test_hyde_empty_falls_back_to_raw_query(self) -> None:
        """When HyDE returns empty, raw query is used for embedding."""
        expander = MagicMock()
        expander.expand_many = AsyncMock(return_value=[])
        expander.close = AsyncMock()

        emb = _make_embedder()
        search_results = [
            SearchResult(chunk_id="c1", parent_chunk_id="p1", doc_id="doc1", score=0.9),
        ]
        children = [_make_child(cid="c1", parent_id="p1", text="Child")]
        parents = [_make_parent(pid="p1", text="Parent")]
        pdfs = [_make_pdf()]

        svc = SearchService(
            text_embedder=emb,
            vector_store=_make_vector_store(search_results),
            kb_repo=_make_kb_repo(
                child_chunks=children,
                parent_chunks=parents,
                pdf_docs=pdfs,
            ),
            query_expander=expander,
        )

        await svc.search("raw query", top_k=5)

        # Should fall back to raw query
        emb.embed_texts.assert_called_once_with(["raw query"], is_query=True)

    @pytest.mark.asyncio
    async def test_hyde_failure_falls_back_to_raw_query(self) -> None:
        """When HyDE raises, raw query is used for embedding."""
        expander = MagicMock()
        expander.expand_many = AsyncMock(side_effect=Exception("HyDE failed"))
        expander.close = AsyncMock()

        emb = _make_embedder()
        search_results = [
            SearchResult(chunk_id="c1", parent_chunk_id="p1", doc_id="doc1", score=0.9),
        ]
        children = [_make_child(cid="c1", parent_id="p1", text="Child")]
        parents = [_make_parent(pid="p1", text="Parent")]
        pdfs = [_make_pdf()]

        svc = SearchService(
            text_embedder=emb,
            vector_store=_make_vector_store(search_results),
            kb_repo=_make_kb_repo(
                child_chunks=children,
                parent_chunks=parents,
                pdf_docs=pdfs,
            ),
            query_expander=expander,
        )

        await svc.search("raw query", top_k=5)

        emb.embed_texts.assert_called_once_with(["raw query"], is_query=True)


class TestAggregateDocuments:
    @pytest.mark.asyncio
    async def test_groups_by_document_and_dedupes_parents(self) -> None:
        d1 = datetime(2026, 1, 1, tzinfo=timezone.utc)
        pdfs = [
            _make_pdf(doc_id="doc1", title="Doc One", released_date=d1),
            _make_pdf(doc_id="doc2", title="Doc Two", released_date=None),
        ]
        contexts = [
            RetrievedContext(chunk_id="c1", parent_chunk_id="p1", doc_id="doc1", text="A", score=0.9),
            # Same parent surfaced twice (sibling/cross-ref) — must be deduped.
            RetrievedContext(chunk_id="c2", parent_chunk_id="p1", doc_id="doc1", text="A", score=0.7),
            RetrievedContext(chunk_id="c3", parent_chunk_id="p2", doc_id="doc1", text="B", score=0.5),
            RetrievedContext(chunk_id="c4", parent_chunk_id="p3", doc_id="doc2", text="C", score=0.8),
        ]
        svc = SearchService(
            text_embedder=_make_embedder(),
            vector_store=_make_vector_store([]),
            kb_repo=_make_kb_repo(pdf_docs=pdfs),
        )

        docs = await svc.aggregate_documents(contexts)

        assert [d.doc_id for d in docs] == ["doc1", "doc2"]
        assert docs[0].title == "Doc One"
        assert docs[0].released_date == d1
        assert docs[0].content == "A\n\nB"  # duplicate parent removed
        assert docs[0].score == 0.9
        assert docs[1].released_date is None


class TestDatePriorityStrategy:
    def test_newer_documents_rank_first(self) -> None:
        older = datetime(2020, 1, 1, tzinfo=timezone.utc)
        newer = datetime(2024, 1, 1, tzinfo=timezone.utc)
        docs = [
            RetrievedDocument(doc_id="old", title="Old", released_date=older, content="x", score=1.0),
            RetrievedDocument(doc_id="new", title="New", released_date=newer, content="x", score=1.0),
        ]

        ranked = DatePriorityStrategy(lam=0.5).rank_documents(docs)

        assert [d.doc_id for d in ranked] == ["new", "old"]

    def test_missing_date_is_not_penalized(self) -> None:
        ref = datetime(2024, 1, 1, tzinfo=timezone.utc)
        old = datetime(2020, 1, 1, tzinfo=timezone.utc)
        docs = [
            RetrievedDocument(doc_id="new", title="New", released_date=ref, content="x", score=1.0),
            RetrievedDocument(doc_id="old", title="Old", released_date=old, content="x", score=1.0),
            RetrievedDocument(doc_id="undated", title="Undated", released_date=None, content="x", score=1.0),
        ]

        ranked = DatePriorityStrategy(lam=0.5).rank_documents(docs)

        assert [d.doc_id for d in ranked] == ["new", "undated", "old"]


class TestQueryExpansionOptOut:
    """HyDE costs one LLM round-trip per passage and is the dominant cost of a
    chat request, so callers that only need a coarse signal can opt out."""

    @staticmethod
    def _expander() -> AsyncMock:
        expander = AsyncMock()
        expander.expand_many = AsyncMock(return_value=["hypothetical passage"])
        return expander

    @pytest.mark.asyncio
    async def test_expansion_runs_by_default(self) -> None:
        expander = self._expander()
        svc = SearchService(
            text_embedder=_make_embedder(),
            vector_store=_make_vector_store([]),
            kb_repo=_make_kb_repo(),
            query_expander=expander,
        )
        await svc.search("query")
        expander.expand_many.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_use_expansion_false_spends_no_llm_call(self) -> None:
        expander = self._expander()
        svc = SearchService(
            text_embedder=_make_embedder(),
            vector_store=_make_vector_store([]),
            kb_repo=_make_kb_repo(),
            query_expander=expander,
        )
        await svc.search("query", use_expansion=False)
        expander.expand_many.assert_not_awaited()


class TestHydrationBudget:
    """Siblings and cross-refs carry score 0.0 — they were never scored against
    the query. Uncapped they crowded out the reranked results and made the
    result set nondeterministic."""

    @pytest.mark.asyncio
    async def test_cross_refs_are_capped(self) -> None:
        from app.kb.application.search_service import MAX_CROSS_REF_CONTEXTS

        primary = [
            RetrievedContext(
                chunk_id="c1", parent_chunk_id="p1", doc_id="d1",
                text="Sesuai Pasal 1, Pasal 2, Pasal 3, Pasal 4, Pasal 5, Pasal 6 dan BAB II.",
                score=0.9, source_title="Doc",
            )
        ]
        repo = _make_kb_repo()
        repo.get_chunks_by_path_prefix = AsyncMock(
            return_value=[_make_parent(pid=f"x{i}", text=f"Isi {i}") for i in range(20)]
        )
        svc = SearchService(
            text_embedder=_make_embedder(),
            vector_store=_make_vector_store([]),
            kb_repo=repo,
        )
        refs = await svc._detect_and_fetch_cross_refs(primary, {}, {"d1": "Doc"})
        assert len(refs) <= MAX_CROSS_REF_CONTEXTS

    @pytest.mark.asyncio
    async def test_cross_ref_prefix_order_is_deterministic(self) -> None:
        # A set would iterate in hash order, so which refs survived varied
        # between processes for the very same query.
        svc = SearchService(
            text_embedder=_make_embedder(),
            vector_store=_make_vector_store([]),
            kb_repo=_make_kb_repo(),
        )
        text = "Mengacu pada Pasal 9, kemudian Pasal 3, lalu Pasal 7."
        assert svc._extract_cross_references(text) == ["pasal_9", "pasal_3", "pasal_7"]

    @pytest.mark.asyncio
    async def test_hydrate_false_skips_the_db_fan_out(self) -> None:
        repo = _make_kb_repo()
        repo.get_sibling_chunks = AsyncMock(return_value=[])
        repo.get_chunks_by_path_prefix = AsyncMock(return_value=[])
        svc = SearchService(
            text_embedder=_make_embedder(),
            vector_store=_make_vector_store([]),
            kb_repo=repo,
        )
        await svc.search("query", hydrate=False)
        repo.get_sibling_chunks.assert_not_awaited()
        repo.get_chunks_by_path_prefix.assert_not_awaited()


class TestRerankProbe:
    """What the cross-encoder scores against. Against the raw question a chunk
    that merely echoes its wording can outrank the one that answers it; the
    generated passage is written in the register of the passage that matches."""

    @staticmethod
    def _svc(probe: str, reranker, expander=None):
        return SearchService(
            text_embedder=_make_embedder(),
            vector_store=_make_vector_store([]),
            kb_repo=_make_kb_repo(),
            reranker=reranker,
            query_expander=expander,
            rerank_probe=probe,
        )

    @staticmethod
    def _expander() -> AsyncMock:
        expander = AsyncMock()
        expander.expand_many = AsyncMock(return_value=["hypothetical answer passage"])
        return expander

    @pytest.mark.asyncio
    async def test_hyde_passage_is_the_probe_when_expansion_ran(self) -> None:
        search_results = [SearchResult(chunk_id="c1", parent_chunk_id="p1", doc_id="d1", score=0.9)]
        reranker = AsyncMock()
        reranker.rerank = AsyncMock(return_value=[RerankResult(index=0, score=0.9)])
        svc = SearchService(
            text_embedder=_make_embedder(),
            vector_store=_make_vector_store(search_results),
            kb_repo=_make_kb_repo(
                child_chunks=[_make_child(cid="c1", parent_id="p1", text="Isi")],
                parent_chunks=[_make_parent(pid="p1", text="Isi parent")],
            ),
            reranker=reranker,
            query_expander=self._expander(),
            rerank_probe="hyde",
        )
        await svc.search("berapa UKT 2026")
        assert reranker.rerank.await_args.kwargs["query"] == "hypothetical answer passage"

    @pytest.mark.asyncio
    async def test_raw_query_is_the_probe_when_configured(self) -> None:
        search_results = [SearchResult(chunk_id="c1", parent_chunk_id="p1", doc_id="d1", score=0.9)]
        reranker = AsyncMock()
        reranker.rerank = AsyncMock(return_value=[RerankResult(index=0, score=0.9)])
        svc = SearchService(
            text_embedder=_make_embedder(),
            vector_store=_make_vector_store(search_results),
            kb_repo=_make_kb_repo(
                child_chunks=[_make_child(cid="c1", parent_id="p1", text="Isi")],
                parent_chunks=[_make_parent(pid="p1", text="Isi parent")],
            ),
            reranker=reranker,
            query_expander=self._expander(),
            rerank_probe="query",
        )
        await svc.search("berapa UKT 2026")
        assert reranker.rerank.await_args.kwargs["query"] == "berapa UKT 2026"

    @pytest.mark.asyncio
    async def test_falls_back_to_raw_query_without_expansion(self) -> None:
        search_results = [SearchResult(chunk_id="c1", parent_chunk_id="p1", doc_id="d1", score=0.9)]
        reranker = AsyncMock()
        reranker.rerank = AsyncMock(return_value=[RerankResult(index=0, score=0.9)])
        svc = SearchService(
            text_embedder=_make_embedder(),
            vector_store=_make_vector_store(search_results),
            kb_repo=_make_kb_repo(
                child_chunks=[_make_child(cid="c1", parent_id="p1", text="Isi")],
                parent_chunks=[_make_parent(pid="p1", text="Isi parent")],
            ),
            reranker=reranker,
            query_expander=None,
            rerank_probe="hyde",
        )
        await svc.search("berapa UKT 2026")
        assert reranker.rerank.await_args.kwargs["query"] == "berapa UKT 2026"
