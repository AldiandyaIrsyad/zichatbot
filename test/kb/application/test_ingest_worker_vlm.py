"""Integration tests for the IngestWorker with VLM enrichment.

Tests the VLM enrichment step in the ingestion pipeline using mocked
adapters (no real HTTP calls, no real database).
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.kb.application.ingest_worker import IngestWorker
from app.rag.chunking.models import ParsedElement, ContentType
from app.rag.chunking.page_classifier import (
    DEFAULT_GARBAGE_RATIO_THRESHOLD,
    DEFAULT_IMAGE_RATIO_THRESHOLD,
    VLM_PAGE_EXTRACTION_PROMPT,
)


@pytest.fixture
def mock_db() -> MagicMock:
    """Mock AsyncSession."""
    db = MagicMock()
    db.commit = AsyncMock()
    db.rollback = AsyncMock()
    return db


@pytest.fixture
def mock_parser() -> AsyncMock:
    """Mock IDocumentParser."""
    parser = AsyncMock()
    return parser


@pytest.fixture
def mock_embedder() -> AsyncMock:
    """Mock ITextEmbedder."""
    embedder = AsyncMock()
    embedder.embed_texts = AsyncMock(return_value=[])
    return embedder


@pytest.fixture
def mock_vector_store() -> AsyncMock:
    """Mock IVectorStore."""
    return AsyncMock()


@pytest.fixture
def mock_kb_repo() -> AsyncMock:
    """Mock IKBRepository."""
    repo = AsyncMock()
    repo.create_ingestion_task = AsyncMock(return_value=MagicMock(id="task-1"))
    repo.update_ingestion_task = AsyncMock()
    repo.get_pdf_by_id = AsyncMock(return_value=MagicMock(
        id="doc-1",
        pdf_path="/fake/path.pdf",
        ingestion_status="pending",
    ))
    repo.save_parent_chunks = AsyncMock()
    return repo


@pytest.fixture
def mock_vlm_enricher() -> AsyncMock:
    """Mock IVLMEnricher."""
    enricher = AsyncMock()
    enricher.describe_image = AsyncMock(return_value="A flowchart showing 5 steps.")
    enricher.set_pdf_path = MagicMock()
    return enricher


@pytest.fixture
def worker(
    mock_db: MagicMock,
    mock_parser: AsyncMock,
    mock_embedder: AsyncMock,
    mock_vector_store: AsyncMock,
    mock_kb_repo: AsyncMock,
    mock_vlm_enricher: AsyncMock,
) -> IngestWorker:
    """IngestWorker with mocked dependencies and VLM enricher."""
    return IngestWorker(
        db=mock_db,
        document_parser=mock_parser,
        text_embedder=mock_embedder,
        vector_store=mock_vector_store,
        kb_repo=mock_kb_repo,
        vlm_enricher=mock_vlm_enricher,
        image_dir="/tmp/test_images",
    )


class TestVLMEnrichmentStep:
    """Tests for the _route_and_enrich_elements method (figure enrichment)."""

    @pytest.mark.asyncio
    async def test_enrich_figures_calls_vlm_for_empty_text_figures(
        self,
        worker: IngestWorker,
        mock_vlm_enricher: AsyncMock,
    ) -> None:
        """FIGURE elements with empty text should be enriched by the VLM.

        Two NarrativeText elements share the image's page so the page's
        image_ratio (1/3) stays below the VISUAL threshold and this
        exercises the per-element routing path, not page-level VISUAL
        extraction.
        """
        elements = [
            ParsedElement(
                element_type="Image",
                text="",
                metadata={"image_path": "/tmp/page1.png", "page_number": 1},
            ),
            ParsedElement(
                element_type="NarrativeText",
                text="Some narrative text.",
                metadata={"page_number": 1},
            ),
            ParsedElement(
                element_type="NarrativeText",
                text="More narrative text.",
                metadata={"page_number": 1},
            ),
        ]

        result = await worker._route_and_enrich_elements(elements, "/fake/path.pdf", "doc-1")

        # VLM should have been called for the figure with the per-image
        # description prompt (per-element path, not the page-extraction prompt)
        mock_vlm_enricher.describe_image.assert_called_once_with(
            "/tmp/page1.png", prompt=worker.image_description_prompt
        )
        # The figure's text should now be the VLM description
        assert result[0].text == "A flowchart showing 5 steps."
        # The narrative text should be unchanged
        assert result[1].text == "Some narrative text."
        assert result[2].text == "More narrative text."

    @pytest.mark.asyncio
    async def test_enrich_figures_skips_figures_with_existing_text(
        self,
        worker: IngestWorker,
        mock_vlm_enricher: AsyncMock,
    ) -> None:
        """FIGURE elements that already have text should not be re-enriched."""
        elements = [
            ParsedElement(
                element_type="Image",
                text="Already has a description.",
                metadata={"image_path": "/tmp/page1.png"},
            ),
        ]

        result = await worker._route_and_enrich_elements(elements, "/fake/path.pdf", "doc-1")

        mock_vlm_enricher.describe_image.assert_not_called()
        assert result[0].text == "Already has a description."

    @pytest.mark.asyncio
    async def test_enrich_figures_no_vlm_filters_empty_figures(
        self,
        mock_db: MagicMock,
        mock_parser: AsyncMock,
        mock_embedder: AsyncMock,
        mock_vector_store: AsyncMock,
        mock_kb_repo: AsyncMock,
    ) -> None:
        """Without a VLM enricher, empty-text figures should be filtered out.

        The lone Image element on page 1 classifies as a VISUAL page, so
        this now exercises `_process_visual_page`'s "no vlm_enricher"
        short-circuit rather than the old per-element "no VLM" branch —
        the externally observable result is identical either way.
        """
        worker = IngestWorker(
            db=mock_db,
            document_parser=mock_parser,
            text_embedder=mock_embedder,
            vector_store=mock_vector_store,
            kb_repo=mock_kb_repo,
            vlm_enricher=None,
        )

        elements = [
            ParsedElement(
                element_type="Image",
                text="",
                metadata={"page_number": 1},
            ),
            ParsedElement(
                element_type="NarrativeText",
                text="Keep this text.",
                metadata={},
            ),
        ]

        result = await worker._route_and_enrich_elements(elements, "/fake/path.pdf", "doc-1")

        # The empty-text figure should be filtered out
        assert len(result) == 1
        assert result[0].text == "Keep this text."

    @pytest.mark.asyncio
    async def test_enrich_figures_vlm_failure_keeps_element(
        self,
        mock_db: MagicMock,
        mock_parser: AsyncMock,
        mock_embedder: AsyncMock,
        mock_vector_store: AsyncMock,
        mock_kb_repo: AsyncMock,
    ) -> None:
        """A VLM failure must abort the document under the default strict mode.

        Ingestion is write-once, so a swallowed VLM error bakes un-transcribed
        OCR into the index — that is how 21 documents completed with pages
        missing to 413 errors. With ``vlm_strict=False`` the old lenient
        behaviour still applies, and that half is asserted below.

        The lone Image element on page 1 classifies as a VISUAL page, so this
        exercises `_process_visual_page`'s VLM-exception path.
        `_extract_page_image` is stubbed so the failure genuinely comes from the
        VLM call, not from PyMuPDF failing on a fake path.
        """
        mock_vlm = AsyncMock()
        mock_vlm.describe_image = AsyncMock(side_effect=Exception("VLM API error"))
        mock_vlm.set_pdf_path = MagicMock()

        worker = IngestWorker(
            db=mock_db,
            document_parser=mock_parser,
            text_embedder=mock_embedder,
            vector_store=mock_vector_store,
            kb_repo=mock_kb_repo,
            vlm_enricher=mock_vlm,
        )

        elements = [
            ParsedElement(
                element_type="Image",
                text="",
                metadata={"image_path": "/tmp/page1.png", "page_number": 1},
            ),
            ParsedElement(
                element_type="NarrativeText",
                text="Narrative.",
                metadata={},
            ),
        ]

        # Strict (the default): the failure propagates and the document fails.
        with patch.object(worker, "_extract_page_image", return_value="/tmp/page1.png"):
            with pytest.raises(Exception, match="VLM API error"):
                await worker._route_and_enrich_elements(elements, "/fake/path.pdf", "doc-1")

        mock_vlm.describe_image.assert_called_once_with(
            "/tmp/page1.png", prompt=VLM_PAGE_EXTRACTION_PROMPT
        )

        # Lenient: opt out explicitly and the page's figure content is dropped
        # while the page-less narrative (page_key=None) survives untouched.
        worker.vlm_strict = False
        mock_vlm.describe_image.reset_mock()
        with patch.object(worker, "_extract_page_image", return_value="/tmp/page1.png"):
            result = await worker._route_and_enrich_elements(elements, "/fake/path.pdf", "doc-1")
        assert len(result) == 1
        assert result[0].text == "Narrative."

    @pytest.mark.asyncio
    async def test_enrich_figures_no_figure_elements(
        self,
        worker: IngestWorker,
        mock_vlm_enricher: AsyncMock,
    ) -> None:
        """If there are no FIGURE elements, VLM should not be called."""
        elements = [
            ParsedElement(element_type="NarrativeText", text="Text 1."),
            ParsedElement(element_type="Title", text="Heading"),
        ]

        result = await worker._route_and_enrich_elements(elements, "/fake/path.pdf", "doc-1")

        mock_vlm_enricher.describe_image.assert_not_called()
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_enrich_figures_empty_list(
        self,
        worker: IngestWorker,
        mock_vlm_enricher: AsyncMock,
    ) -> None:
        """Empty elements list should return empty without calling VLM."""
        result = await worker._route_and_enrich_elements([], "/fake/path.pdf", "doc-1")
        assert result == []
        mock_vlm_enricher.describe_image.assert_not_called()

    @pytest.mark.asyncio
    async def test_enrich_figures_sets_pdf_path_for_fallback(
        self,
        worker: IngestWorker,
        mock_vlm_enricher: AsyncMock,
    ) -> None:
        """The VLM enricher should receive the PDF path (for fallback mode)."""
        elements = [
            ParsedElement(
                element_type="Image",
                text="",
                metadata={"image_path": "/tmp/page1.png", "page_number": 1},
            ),
        ]

        await worker._route_and_enrich_elements(elements, "/custom/path.pdf", "doc-1")

        mock_vlm_enricher.set_pdf_path.assert_called_once_with("/custom/path.pdf")


class TestVisualPageRouting:
    """Tests for page-level VISUAL classification and consolidated VLM extraction."""

    @pytest.mark.asyncio
    async def test_visual_page_multiple_garbage_figures_single_vlm_call(
        self,
        worker: IngestWorker,
        mock_vlm_enricher: AsyncMock,
    ) -> None:
        """A page mis-split into several garbage-OCR figures should trigger
        exactly one page image extraction and one VLM call, not one per
        element."""
        elements = [
            ParsedElement(element_type="Image", text=t, metadata={"page_number": 2})
            for t in ["", "L", "qp", "", "6"]
        ]

        with patch.object(
            worker, "_extract_page_image", return_value="/tmp/page2.png"
        ) as mock_extract:
            result = await worker._route_and_enrich_elements(elements, "/fake/path.pdf", "doc-1")

        mock_extract.assert_called_once_with("/fake/path.pdf", 2, "doc-1")
        mock_vlm_enricher.describe_image.assert_called_once_with(
            "/tmp/page2.png", prompt=VLM_PAGE_EXTRACTION_PROMPT
        )
        assert len(result) == 1
        assert result[0].content_type == ContentType.FIGURE
        assert result[0].text == "A flowchart showing 5 steps."
        assert result[0].metadata["page_classification"] == "visual"

    @pytest.mark.asyncio
    async def test_visual_page_preserves_title_element(
        self,
        worker: IngestWorker,
        mock_vlm_enricher: AsyncMock,
    ) -> None:
        """A genuine Title on a VISUAL page must survive so downstream
        heading breadcrumbs aren't lost."""
        elements = [
            ParsedElement(
                element_type="Title", text="BAB II - Alur Proses", metadata={"page_number": 3}
            ),
            *[
                ParsedElement(element_type="Image", text=t, metadata={"page_number": 3})
                for t in ["", "", "L", ""]
            ],
        ]

        with patch.object(worker, "_extract_page_image", return_value="/tmp/page3.png"):
            result = await worker._route_and_enrich_elements(elements, "/fake/path.pdf", "doc-1")

        assert result[0].element_type == "Title"
        assert result[0].text == "BAB II - Alur Proses"
        assert result[1].content_type == ContentType.FIGURE
        mock_vlm_enricher.describe_image.assert_called_once()

    @pytest.mark.asyncio
    async def test_visual_page_vlm_failure_preserves_title_no_figure(
        self,
        mock_db: MagicMock,
        mock_parser: AsyncMock,
        mock_embedder: AsyncMock,
        mock_vector_store: AsyncMock,
        mock_kb_repo: AsyncMock,
    ) -> None:
        """A VISUAL-page VLM failure aborts under strict; under lenient the
        Title survives and no figure chunk is produced.

        Strict is the default because a VISUAL page has no usable native text —
        dropping its figure means the page contributes nothing to the index."""
        mock_vlm = AsyncMock()
        mock_vlm.describe_image = AsyncMock(side_effect=Exception("VLM API error"))
        mock_vlm.set_pdf_path = MagicMock()

        worker = IngestWorker(
            db=mock_db,
            document_parser=mock_parser,
            text_embedder=mock_embedder,
            vector_store=mock_vector_store,
            kb_repo=mock_kb_repo,
            vlm_enricher=mock_vlm,
        )

        elements = [
            ParsedElement(
                element_type="Title", text="BAB II - Alur Proses", metadata={"page_number": 3}
            ),
            *[
                ParsedElement(element_type="Image", text=t, metadata={"page_number": 3})
                for t in ["", "", "L", ""]
            ],
        ]

        with patch.object(worker, "_extract_page_image", return_value="/tmp/page3.png"):
            with pytest.raises(Exception, match="VLM API error"):
                await worker._route_and_enrich_elements(elements, "/fake/path.pdf", "doc-1")

        worker.vlm_strict = False
        with patch.object(worker, "_extract_page_image", return_value="/tmp/page3.png"):
            result = await worker._route_and_enrich_elements(elements, "/fake/path.pdf", "doc-1")

        assert len(result) == 1
        assert result[0].element_type == "Title"

    @pytest.mark.asyncio
    async def test_visual_page_image_extraction_failure_drops_figure(
        self,
        worker: IngestWorker,
        mock_vlm_enricher: AsyncMock,
    ) -> None:
        """If page image extraction fails, the VLM is never called and the
        figure content is dropped."""
        elements = [
            ParsedElement(element_type="Image", text=t, metadata={"page_number": 4})
            for t in ["", "", "L", ""]
        ]

        with patch.object(worker, "_extract_page_image", return_value=None):
            result = await worker._route_and_enrich_elements(elements, "/fake/path.pdf", "doc-1")

        mock_vlm_enricher.describe_image.assert_not_called()
        assert result == []

    @pytest.mark.asyncio
    async def test_table_rich_page_unaffected_by_page_classification(
        self,
        worker: IngestWorker,
        mock_vlm_enricher: AsyncMock,
    ) -> None:
        """A page with a table and narrative text (no images) never
        classifies as VISUAL and keeps the existing table conversion
        behavior."""
        elements = [
            ParsedElement(
                element_type="Table",
                text="<table><tr><td>A</td></tr></table>",
                metadata={"text_as_html": "<table><tr><td>A</td></tr></table>", "page_number": 5},
            ),
            ParsedElement(
                element_type="NarrativeText",
                text="Some text.",
                metadata={"page_number": 5},
            ),
        ]

        result = await worker._route_and_enrich_elements(elements, "/fake/path.pdf", "doc-1")

        mock_vlm_enricher.describe_image.assert_not_called()
        assert len(result) == 2
        assert result[0].content_type == ContentType.TABLE
        assert "|" in result[0].text  # converted to Markdown

    @pytest.mark.asyncio
    async def test_page_without_page_number_uses_legacy_routing(
        self,
        worker: IngestWorker,
        mock_vlm_enricher: AsyncMock,
    ) -> None:
        """Elements with no page_number metadata are grouped under
        page_key=None and always use per-element routing, never page-level
        VISUAL classification, even if the group would numerically satisfy
        VISUAL thresholds."""
        elements = [
            ParsedElement(element_type="Image", text="", metadata={})
            for _ in range(4)
        ]

        with patch.object(
            worker, "_extract_page_image", return_value="/tmp/page1.png"
        ) as mock_extract:
            result = await worker._route_and_enrich_elements(elements, "/fake/path.pdf", "doc-1")

        # Per-element routing: one extraction + one VLM call per element
        assert mock_extract.call_count == 4
        assert mock_vlm_enricher.describe_image.call_count == 4
        assert len(result) == 4

    @pytest.mark.asyncio
    async def test_page_classification_thresholds_are_configurable(
        self,
        mock_db: MagicMock,
        mock_parser: AsyncMock,
        mock_embedder: AsyncMock,
        mock_vector_store: AsyncMock,
        mock_kb_repo: AsyncMock,
        mock_vlm_enricher: AsyncMock,
    ) -> None:
        """Stricter custom thresholds should keep a page that would be
        VISUAL under defaults on the per-element path instead."""
        worker = IngestWorker(
            db=mock_db,
            document_parser=mock_parser,
            text_embedder=mock_embedder,
            vector_store=mock_vector_store,
            kb_repo=mock_kb_repo,
            vlm_enricher=mock_vlm_enricher,
            page_image_ratio_threshold=0.95,
            page_garbage_ratio_threshold=0.95,
        )

        elements = [
            ParsedElement(element_type="Image", text="", metadata={"page_number": 6}),
            ParsedElement(
                element_type="NarrativeText", text="Some body text.", metadata={"page_number": 6}
            ),
        ]

        with patch.object(
            worker, "_extract_page_image", return_value="/tmp/page6.png"
        ):
            await worker._route_and_enrich_elements(elements, "/fake/path.pdf", "doc-1")

        # Per-element path: description prompt passed, not the page-extraction prompt
        mock_vlm_enricher.describe_image.assert_called_once_with(
            "/tmp/page6.png", prompt=worker.image_description_prompt
        )


class TestIngestWorkerConstruction:
    """Tests for IngestWorker constructor and configuration."""

    def test_worker_accepts_vlm_enricher(
        self,
        mock_db: MagicMock,
        mock_parser: AsyncMock,
        mock_embedder: AsyncMock,
        mock_vector_store: AsyncMock,
        mock_kb_repo: AsyncMock,
        mock_vlm_enricher: AsyncMock,
    ) -> None:
        """IngestWorker should accept an optional vlm_enricher parameter."""
        worker = IngestWorker(
            db=mock_db,
            document_parser=mock_parser,
            text_embedder=mock_embedder,
            vector_store=mock_vector_store,
            kb_repo=mock_kb_repo,
            vlm_enricher=mock_vlm_enricher,
            image_dir="/custom/images",
        )

        assert worker.vlm_enricher is mock_vlm_enricher
        assert worker.image_dir == "/custom/images"

    def test_worker_works_without_vlm_enricher(
        self,
        mock_db: MagicMock,
        mock_parser: AsyncMock,
        mock_embedder: AsyncMock,
        mock_vector_store: AsyncMock,
        mock_kb_repo: AsyncMock,
    ) -> None:
        """IngestWorker should work without a VLM enricher (None default)."""
        worker = IngestWorker(
            db=mock_db,
            document_parser=mock_parser,
            text_embedder=mock_embedder,
            vector_store=mock_vector_store,
            kb_repo=mock_kb_repo,
        )

        assert worker.vlm_enricher is None

    def test_worker_defaults_page_classification_thresholds(
        self,
        mock_db: MagicMock,
        mock_parser: AsyncMock,
        mock_embedder: AsyncMock,
        mock_vector_store: AsyncMock,
        mock_kb_repo: AsyncMock,
    ) -> None:
        """Without explicit thresholds, the worker uses page_classifier's defaults."""
        worker = IngestWorker(
            db=mock_db,
            document_parser=mock_parser,
            text_embedder=mock_embedder,
            vector_store=mock_vector_store,
            kb_repo=mock_kb_repo,
        )

        assert worker.page_image_ratio_threshold == DEFAULT_IMAGE_RATIO_THRESHOLD
        assert worker.page_garbage_ratio_threshold == DEFAULT_GARBAGE_RATIO_THRESHOLD
