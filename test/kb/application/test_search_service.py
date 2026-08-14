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
from unittest.mock import AsyncMock, MagicMock

from app.kb.application.search_service import SearchService
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


def _make_pdf(doc_id: str = "doc1", title: str = "Test Doc") -> MagicMock:
    """Create a mock PDFDocument with the given attributes."""
    pdf = MagicMock(spec=PDFDocument)
    pdf.id = doc_id
    pdf.title = title
    pdf.description = ""
    pdf.pdf_path = "/fake/path.pdf"
    pdf.is_active = True
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
        parent2 = _make_parent(pid="p2", text="Sibling text", parent_id="root", path="root.p2")
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
        assert "Sibling text" in texts

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
            pid="p5", text="Pasal 5 content", path="pasal_5",
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
        assert "Pasal 5 content" in texts

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
        emb.embed_texts.assert_called_once_with(["raw query"])

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

        emb.embed_texts.assert_called_once_with(["raw query"])
