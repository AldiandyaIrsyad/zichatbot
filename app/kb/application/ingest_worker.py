"""
Asynchronous document ingestion workflow for the KB domain.
"""

import asyncio
import os
import structlog
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Callable, Optional

from app.kb.domain.interfaces import (
    IDocumentParser,
    ITextEmbedder,
    IVectorStore,
    IKBRepository,
    ChunkVector,
)
from app.kb.domain.models import ParentChunk, ChildChunk
from app.rag.chunking.config import ChunkingConfig, ITokenizer
from app.rag.chunking.logic import infer_acronyms
from app.rag.chunking.models import ChildChunkData, ContentType, ParsedElement
from app.rag.chunking.strategy import chunk_children, chunk_parents
from app.rag.chunking.page_classifier import (
    DEFAULT_GARBAGE_RATIO_THRESHOLD,
    DEFAULT_IMAGE_RATIO_THRESHOLD,
    PageType,
    VLM_PAGE_EXTRACTION_PROMPT,
    classify_page,
    group_elements_by_page,
    parsed_effectively,
)
from app.rag.chunking.router import classify_element
from app.rag.vlm.cleanup import clean_vlm_output
from app.rag.vlm.client import DEFAULT_VLM_PROMPT
from app.rag.vlm.interfaces import IVLMEnricher
from app.shared.table_converter import html_table_to_markdown

logger = structlog.get_logger(__name__)


class IngestWorker:
    """Orchestrates the ingestion pipeline for a document into the Knowledge Base.

    Pipeline stages:
    1. Parse PDF → ParsedElement[] (Unstructured API)
    2. Route & enrich elements (single linear pass):
       - FIGURE (Image) → VLM enrichment (generate text description)
       - TABLE → HTML → Markdown conversion
       - TEXT → keep as-is
    3. Create parent chunks (table-aware, figure-aware, breadcrumbs)
    4. Save parent chunks to Postgres
    5. Split into children (content-type-aware, min-length gibberish filter)
    6. Embed children (in-process BGE-M3 dense + sparse)
    7. Upsert to Qdrant
    8. Complete
    """

    def __init__(
        self,
        db: AsyncSession,
        document_parser: IDocumentParser,
        text_embedder: ITextEmbedder,
        vector_store: IVectorStore,
        kb_repo: IKBRepository,
        vlm_enricher: Optional[IVLMEnricher] = None,
        image_dir: str = "./uploads/knowledge_base/images",
        page_image_ratio_threshold: float = DEFAULT_IMAGE_RATIO_THRESHOLD,
        page_garbage_ratio_threshold: float = DEFAULT_GARBAGE_RATIO_THRESHOLD,
        image_description_prompt: str = DEFAULT_VLM_PROMPT,
        page_extraction_prompt: str = VLM_PAGE_EXTRACTION_PROMPT,
        chunking_config: Optional[ChunkingConfig] = None,
        tokenizer: Optional[ITokenizer] = None,
        on_stage: Optional[Callable[[str, dict], None]] = None,
        vlm_strict: bool = True,
    ):
        """Wire the pipeline's collaborators (all injected ports/adapters).

        ``db`` is committed at each stage boundary so partial progress survives
        a later failure. ``vlm_enricher=None`` disables figure enrichment
        (figures are dropped, or heuristically described in fallback mode).
        ``chunking_config=None`` uses the hierarchical defaults; ``tokenizer``
        is only required when its strategy is ``"fixed"``.
        ``on_stage`` is an optional research/visualization hook (see below).
        """
        self.db = db
        self.document_parser = document_parser
        self.text_embedder = text_embedder
        self.vector_store = vector_store
        self.kb_repo = kb_repo
        self.vlm_enricher = vlm_enricher
        self.image_dir = image_dir
        # When true (default), a VLM failure fails the document instead of
        # silently storing un-enriched OCR. See VLMSettings.strict.
        self.vlm_strict = vlm_strict
        self.page_image_ratio_threshold = page_image_ratio_threshold
        self.page_garbage_ratio_threshold = page_garbage_ratio_threshold
        self.image_description_prompt = image_description_prompt
        self.page_extraction_prompt = page_extraction_prompt
        self.chunking_config = chunking_config or ChunkingConfig()
        self.tokenizer = tokenizer
        # Cap concurrent VLM ``describe_image`` calls. Pages are processed
        # concurrently (asyncio.gather); VLM providers rate-limit hard, so the
        # shared enricher's httpx client is bounded rather than flooded.
        self._vlm_semaphore = asyncio.Semaphore(4)
        # Optional research/visualization hook invoked at each pipeline stage
        # with the data on hand. Defaults to None so production ingestion is
        # unaffected; set by tools.visualize.production_ingestion_viz to capture
        # real routing decisions for documentation.
        self.on_stage = on_stage

    def _emit(self, stage: str, data: dict) -> None:
        """Invoke ``self.on_stage`` (if configured) with a pipeline stage's
        intermediate data; a no-op for normal (production) ingestion runs."""
        if self.on_stage is not None:
            self.on_stage(stage, data)

    async def ingest_document(self, doc_id: str) -> None:
        """Run the full ingestion pipeline for a document."""
        task = await self.kb_repo.create_ingestion_task(doc_id)
        await self.db.commit()

        try:
            await self.kb_repo.update_ingestion_task(task.id, "processing")
            pdf_doc = await self.kb_repo.get_pdf_by_id(doc_id)
            if not pdf_doc:
                raise ValueError(f"PDFDocument not found: {doc_id}")
            pdf_doc.ingestion_status = "processing"  # type: ignore
            await self.db.commit()

            logger.info("kb.ingest.started", doc_id=doc_id, stage="parsing")

            # 1. Parse PDF
            elements = await self.document_parser.parse_pdf(str(pdf_doc.pdf_path))
            self._emit("parsed", {"elements": elements})
            if not elements:
                await self.kb_repo.update_ingestion_task(task.id, "completed")
                pdf_doc.ingestion_status = "completed"  # type: ignore
                await self.db.commit()
                return

            # 2. Route & enrich elements (single linear pass):
            #    - FIGURE (Image) → VLM enrichment (generate text description)
            #    - TABLE → HTML → Markdown conversion
            #    - TEXT → keep as-is
            elements = await self._route_and_enrich_elements(
                elements, str(pdf_doc.pdf_path), doc_id
            )
            self._emit("routed", {"elements": elements})

            # Stamp the document title onto Table elements so the table-summary
            # child can name the subject a reader would ("Kelompok Tarif Uang
            # Kuliah Tunggal") rather than only its column headers. The title is
            # already loaded above; chunking has no access to the repository.
            doc_title = (getattr(pdf_doc, "title", "") or "").strip()
            if doc_title:
                for element in elements:
                    if element.element_type == "Table":
                        element.metadata["doc_title"] = doc_title
            # Stamp the abbreviations this document defines ("Uang Kuliah
            # Tunggal yang selanjutnya disingkat UKT") so the table summary can
            # name its subject the way a question does. The definitions live in
            # Pasal 1, i.e. in a different element from the table, so they have
            # to be gathered document-wide before chunking splits them apart.
            acronyms = infer_acronyms(
                doc_title, "\n".join(el.text for el in elements if el.text)
            )
            if acronyms:
                for element in elements:
                    if element.element_type == "Table":
                        element.metadata["acronyms"] = acronyms

            # Carry the configured table mode on the element too: the chunking
            # layer reads os.getenv, which never sees .env (pydantic-settings
            # parses that file without exporting), so config must travel here.
            for element in elements:
                if element.element_type == "Table":
                    element.metadata["table_child_mode"] = self.chunking_config.table_child_mode

            # 3. Pure Chunking Algorithm (Thesis) — hierarchical or fixed
            parent_chunk_data = chunk_parents(
                elements, doc_id, self.chunking_config, self.tokenizer
            )
            self._emit("parent_chunks", {"parent_chunk_data": parent_chunk_data})

            # 4. Save Parent Chunks to Postgres
            parent_chunk_models = [
                ParentChunk(
                    id=pc.id,
                    doc_id=pc.doc_id,
                    text=pc.text,
                    chunk_index=pc.chunk_index,
                    page=pc.page,
                    breadcrumbs=pc.breadcrumbs,
                    content_type=pc.content_type.value,
                    element_metadata=pc.element_metadata,
                    parent_id=pc.parent_id,
                    ordinal=pc.ordinal,
                    path=pc.path,
                    depth=pc.depth,
                )
                for pc in parent_chunk_data
            ]
            await self.kb_repo.save_parent_chunks(parent_chunk_models)
            await self.db.commit()

            # 5. Split into children
            all_children: list[ChildChunkData] = []
            for parent in parent_chunk_data:
                all_children.extend(
                    chunk_children(parent, self.chunking_config, self.tokenizer)
                )

            if not all_children:
                await self.kb_repo.update_ingestion_task(task.id, "completed")
                doc = await self.kb_repo.get_pdf_by_id(doc_id)
                if doc:
                    doc.ingestion_status = "completed"  # type: ignore
                await self.db.commit()
                return

            logger.info("kb.ingest.chunking_complete", doc_id=doc_id, strategy=self.chunking_config.strategy, parent_chunks=len(parent_chunk_data), child_chunks=len(all_children))
            self._emit("children", {"all_children": all_children})

            # 5b. Save Child Chunks to Postgres (for chunk-level reranking + sibling lookups)
            child_chunk_models = [
                ChildChunk(
                    id=child.id,
                    parent_chunk_id=child.parent_chunk_id,
                    doc_id=child.doc_id,
                    text=child.text,
                    ordinal=child.ordinal,
                    path=child.path,
                    page=child.page,
                    content_type=child.content_type.value,
                )
                for child in all_children
            ]
            await self.kb_repo.save_child_chunks(child_chunk_models)
            await self.db.commit()

            # 6. Embed children
            child_texts = [c.text for c in all_children]
            embeddings = await self.text_embedder.embed_texts(child_texts)
            self._emit("embeddings", {"all_children": all_children, "embeddings": embeddings})

            # 7. Upsert to Qdrant
            chunk_vectors = [
                ChunkVector(
                    chunk_id=child.id,
                    parent_chunk_id=child.parent_chunk_id,
                    doc_id=child.doc_id,
                    dense_vector=emb.dense,
                    sparse_indices=emb.sparse_indices,
                    sparse_values=emb.sparse_values,
                    breadcrumbs=child.breadcrumbs,
                    content_type=child.content_type.value,
                    text=child.text,
                )
                for child, emb in zip(all_children, embeddings)
            ]
            
            # Check if document was deleted by user during long processing
            current_doc = await self.kb_repo.get_pdf_by_id(doc_id)
            if not current_doc:
                logger.warning("kb.ingest.aborted_document_deleted", doc_id=doc_id)
                return
                
            await self.vector_store.upsert_chunks(chunk_vectors)
            self._emit("upserted", {"chunk_vectors": chunk_vectors})

            # 8. Complete
            await self.kb_repo.update_ingestion_task(task.id, "completed")
            doc = await self.kb_repo.get_pdf_by_id(doc_id)
            if doc:
                doc.ingestion_status = "completed"  # type: ignore
            await self.db.commit()

            logger.info("kb.ingest.completed", doc_id=doc_id)

        except Exception as e:
            await self.db.rollback()
            logger.error("kb.ingest.failed", doc_id=doc_id, error=str(e), exc_info=True)
            try:
                await self.kb_repo.update_ingestion_task(task.id, "failed", error_message=str(e))
                pdf_doc = await self.kb_repo.get_pdf_by_id(doc_id)
                if pdf_doc:
                    pdf_doc.ingestion_status = "failed"  # type: ignore
                await self.db.commit()
            except Exception as rollback_err:
                logger.error("kb.ingest.status_update_failed", error=str(rollback_err))
            raise

    async def _route_and_enrich_elements(
        self,
        elements: list[ParsedElement],
        pdf_path: str,
        doc_id: str,
    ) -> list[ParsedElement]:
        """Route each parsed element to the best processing strategy.

        Groups elements by page and classifies each page via
        :func:`classify_page`:

        - **VISUAL** pages (mostly images with garbage OCR — flowcharts,
          diagrams): the noisy per-element Unstructured output is discarded
          and replaced by a single full-page VLM extraction
          (:meth:`_process_visual_page`), avoiding redundant per-element VLM
          calls when OCR mis-splits one diagram into several "figure"
          elements. Genuine ``Title`` elements are preserved so downstream
          heading breadcrumbs aren't lost.
        - **TEXT_RICH / TABLE_RICH / MIXED** pages (and elements with no
          page number) keep the original per-element routing
          (:meth:`_route_default_page_elements`): FIGURE elements get
          per-element VLM enrichment, TABLE elements get HTML→Markdown
          conversion, TEXT elements are kept as-is.

        Elements with empty text after processing are filtered out. Breadcrumbs
        are preserved — built later by :func:`create_parent_chunks` from the
        heading hierarchy. Returns the processed elements with figures
        enriched, tables converted, and empty-text elements removed.
        """
        if not elements:
            return elements

        # Classify all elements by content type
        for el in elements:
            el.content_type = classify_element(el)

        figure_count = sum(1 for el in elements if el.content_type == ContentType.FIGURE)
        table_count = sum(1 for el in elements if el.content_type == ContentType.TABLE)
        logger.info(
            "kb.ingest.element_routing_started",
            doc_id=doc_id,
            total_elements=len(elements),
            figures=figure_count,
            tables=table_count,
        )

        # Set PDF path for VLM fallback mode (heuristic description generator)
        if self.vlm_enricher is not None and hasattr(self.vlm_enricher, "set_pdf_path"):
            self.vlm_enricher.set_pdf_path(pdf_path)  # type: ignore[attr-defined]

        os.makedirs(self.image_dir, exist_ok=True)

        page_groups = group_elements_by_page(elements)
        page_items = list(page_groups.items())

        result_elements: list[ParsedElement] = []
        enriched_count = 0
        tables_converted = 0
        pages_classified = 0
        visual_pages_count = 0
        visual_pages_enriched = 0
        visual_pages_failed = 0

        # 1. Classify pages sequentially (CPU-light) so emit order and logs stay
        #    deterministic; VISUAL pages are counted here.
        classifications = {}
        for page_key, page_elements in page_items:
            classification = None
            if page_key is not None:
                classification = classify_page(
                    page_elements,
                    page_number=page_key,
                    image_ratio_threshold=self.page_image_ratio_threshold,
                    garbage_ratio_threshold=self.page_garbage_ratio_threshold,
                    native_text_len=self._get_native_text_length(pdf_path, page_key),
                )
                pages_classified += 1
                logger.debug(
                    "kb.ingest.page_classified",
                    doc_id=doc_id,
                    page=page_key,
                    page_type=classification.page_type.value,
                    image_ratio=classification.image_ratio,
                    garbage_ratio=classification.garbage_ratio,
                    element_count=classification.element_count,
                )
                self._emit("page_classified", {"classification": classification})
            classifications[page_key] = classification
            if classification is not None and classification.page_type == PageType.VISUAL:
                visual_pages_count += 1

        # 2. Process pages concurrently. VLM describe_image calls dominate, so
        #    gathering overlaps the network round-trips across pages; the
        #    per-VLM semaphore bounds provider load. Returns
        #    (elements, enriched_delta, tables_delta, visual_enriched, visual_failed).
        visual_pages_parser_kept = 0

        async def _process_page(page_key, page_elements, classification):
            nonlocal visual_pages_parser_kept
            if (
                classification is not None
                and classification.page_type == PageType.VISUAL
                and parsed_effectively(page_elements)
            ):
                # The page looks visual, but the parser read it anyway — MinerU
                # OCRs scanned tables that Unstructured could only garble, and
                # this classification predates that. Replacing a real table with
                # a VLM transcription of the same page loses cells (measured:
                # half of those transcriptions ended mid-row), so the parser's
                # output wins and only genuine images go to the VLM, via the
                # per-element figure path below.
                visual_pages_parser_kept += 1
                logger.info(
                    "kb.ingest.visual_page_parser_kept",
                    doc_id=doc_id,
                    page=page_key,
                    elements=len(page_elements),
                    tables=sum(1 for el in page_elements if el.element_type == "Table"),
                )
                processed, enriched_delta, tables_delta = await self._route_default_page_elements(
                    page_elements, pdf_path, doc_id,
                )
                return processed, enriched_delta, tables_delta, 0, 0

            if classification is not None and classification.page_type == PageType.VISUAL:
                # Preserve genuine section-boundary Titles so downstream
                # heading breadcrumbs (create_parent_chunks) aren't broken.
                titles = [
                    el for el in page_elements
                    if el.element_type == "Title" and len(el.text.strip()) > 3
                ]
                out: list[ParsedElement] = list(titles)
                figure_el = await self._process_visual_page(page_key, pdf_path, doc_id)
                if figure_el is not None:
                    figure_el.metadata.update({
                        "source_element_count": classification.element_count,
                        "source_image_ratio": round(classification.image_ratio, 3),
                        "source_garbage_ratio": round(classification.garbage_ratio, 3),
                    })
                    out.append(figure_el)
                    return out, 1, 0, 1, 0
                logger.warning(
                    "kb.ingest.visual_page_dropped",
                    doc_id=doc_id,
                    page=page_key,
                )
                return out, 0, 0, 0, 1
            processed, enriched_delta, tables_delta = await self._route_default_page_elements(
                page_elements, pdf_path, doc_id,
            )
            return processed, enriched_delta, tables_delta, 0, 0

        page_results = await asyncio.gather(
            *(
                _process_page(page_key, page_elements, classifications[page_key])
                for page_key, page_elements in page_items
            ),
            return_exceptions=True,
        )

        # 3. Reassemble in original page order.
        for (page_key, _page_elements), res in zip(page_items, page_results):
            if isinstance(res, Exception):
                logger.error(
                    "kb.ingest.page_failed",
                    doc_id=doc_id,
                    page=page_key,
                    error=str(res),
                )
                # gather(return_exceptions=True) turns a raised VLM failure back
                # into a value, which silently undid the strict check inside
                # _process_page: the first full reingest completed 21 documents
                # that were each missing pages to 413 errors. Re-raise here so
                # strict mode actually fails the document.
                if self.vlm_strict:
                    raise res
                continue
            elems, enriched_delta, tables_delta, visual_enriched, visual_failed = res
            result_elements.extend(elems)
            enriched_count += enriched_delta
            tables_converted += tables_delta
            visual_pages_enriched += visual_enriched
            visual_pages_failed += visual_failed

        # Final filter: remove any elements with empty text (can't embed them)
        # This catches figures where VLM failed and any other empty artifacts.
        final_elements = [el for el in result_elements if el.text.strip()]
        dropped = len(result_elements) - len(final_elements)

        logger.info(
            "kb.ingest.element_routing_complete",
            doc_id=doc_id,
            input_elements=len(elements),
            output_elements=len(final_elements),
            figures_enriched=enriched_count,
            tables_converted=tables_converted,
            dropped_empty=dropped,
            pages_classified=pages_classified,
            visual_pages_count=visual_pages_count,
            visual_pages_enriched=visual_pages_enriched,
            visual_pages_failed=visual_pages_failed,
            visual_pages_parser_kept=visual_pages_parser_kept,
        )
        return final_elements

    async def _route_default_page_elements(
        self,
        page_elements: list[ParsedElement],
        pdf_path: str,
        doc_id: str,
    ) -> tuple[list[ParsedElement], int, int]:
        """Route one page's elements individually (non-VISUAL pages).

        - **FIGURE** (Image/Figure): If the element has no text, extract the
          page image and call the VLM enricher to generate a text description.
          If no VLM is configured or enrichment fails, the element is dropped
          (it cannot be embedded without text).
        - **TABLE**: Convert HTML table markup to Markdown in-place using
          :func:`html_table_to_markdown`, preserving the original HTML in
          metadata. This makes the table embeddable and splittable.
        - **TEXT** (NarrativeText, Title, etc.): Keep as-is.

        Returns:
            (processed elements, figures_enriched delta, tables_converted delta).
        """
        result_elements: list[ParsedElement] = []
        enriched_count = 0
        tables_converted = 0

        for el in page_elements:
            # --- FIGURE routing: VLM enrichment for empty-text images ---
            if el.content_type == ContentType.FIGURE:
                # If already has text (e.g. from a previous run), keep it
                if el.text.strip():
                    result_elements.append(el)
                    enriched_count += 1
                    continue

                if self.vlm_enricher is None:
                    logger.debug(
                        "kb.ingest.figure_skipped",
                        doc_id=doc_id,
                        page=el.metadata.get("page_number"),
                        reason="no_vlm_enricher",
                    )
                    continue  # Drop — can't embed without text

                # Get image path from metadata, or extract the page
                image_path = el.metadata.get("image_path")
                page_number = el.metadata.get("page_number", 1)

                if not image_path:
                    image_path = self._extract_page_image(pdf_path, page_number, doc_id)
                    if image_path:
                        el.metadata["image_path"] = image_path

                if not image_path:
                    logger.warning(
                        "kb.ingest.figure_no_image",
                        doc_id=doc_id,
                        page=page_number,
                        reason="image_extraction_failed",
                    )
                    continue

                try:
                    async with self._vlm_semaphore:
                        description = await self.vlm_enricher.describe_image(
                            image_path, prompt=self.image_description_prompt
                        )
                    if description and description.strip():
                        el.text = clean_vlm_output(description).strip()
                        enriched_count += 1
                        result_elements.append(el)
                        logger.info(
                            "kb.ingest.figure_enriched",
                            doc_id=doc_id,
                            page=page_number,
                            desc_len=len(description),
                        )
                    else:
                        logger.warning(
                            "kb.ingest.vlm_empty_description",
                            doc_id=doc_id,
                            page=page_number,
                        )
                except Exception as exc:
                    logger.error(
                        "kb.ingest.vlm_enrichment_failed",
                        doc_id=doc_id,
                        page=page_number,
                        error=str(exc),
                    )
                    # Ingestion is write-once: whatever lands here is what every
                    # later stage reads. Swallowing a VLM failure stores raw OCR
                    # that looks fine until a table turns out to be unusable
                    # months later. Fail the document so it can be retried.
                    if self.vlm_strict:
                        raise
                continue

            # --- TABLE routing: HTML → Markdown conversion ---
            if el.content_type == ContentType.TABLE:
                html_text = el.metadata.get("text_as_html") or el.text
                if html_text and html_text.strip():
                    markdown = html_table_to_markdown(html_text)
                    el = ParsedElement(
                        element_type=el.element_type,
                        text=markdown,
                        metadata={**el.metadata, "text_as_html": html_text},
                        content_type=el.content_type,
                    )
                    tables_converted += 1
                    logger.debug(
                        "kb.ingest.table_converted",
                        doc_id=doc_id,
                        page=el.metadata.get("page_number"),
                        original_len=len(html_text),
                        markdown_len=len(markdown),
                    )
                result_elements.append(el)
                continue

            # --- TEXT routing: keep as-is ---
            result_elements.append(el)

        return result_elements, enriched_count, tables_converted

    async def _process_visual_page(
        self,
        page_number: int,
        pdf_path: str,
        doc_id: str,
    ) -> Optional[ParsedElement]:
        """Run a single full-page VLM extraction for a page classified VISUAL.

        Renders the page fresh via :meth:`_extract_page_image` rather than
        reusing an element's ``image_path`` — on VISUAL pages that metadata is
        typically a small, unreliable crop, while the full-page prompt needs the
        whole page. Fail-closed: returns ``None`` (visual content dropped) if no
        VLM is configured, image extraction fails, the VLM call raises, or the
        VLM returns empty text.
        """
        if self.vlm_enricher is None:
            logger.debug(
                "kb.ingest.visual_page_skipped",
                doc_id=doc_id,
                page=page_number,
                reason="no_vlm_enricher",
            )
            return None

        image_path = self._extract_page_image(pdf_path, page_number, doc_id)
        if not image_path:
            logger.warning(
                "kb.ingest.visual_page_no_image",
                doc_id=doc_id,
                page=page_number,
                reason="image_extraction_failed",
            )
            return None

        try:
            async with self._vlm_semaphore:
                description = await self.vlm_enricher.describe_image(
                    image_path, prompt=self.page_extraction_prompt
                )
        except Exception as exc:
            logger.error(
                "kb.ingest.visual_page_vlm_failed",
                doc_id=doc_id,
                page=page_number,
                error=str(exc),
            )
            # A VISUAL page is one the classifier decided has no usable native
            # text, so returning None here means the page contributes nothing
            # (or garbage) to the index. That is precisely the failure that
            # produced the mangled UKT tariff table.
            if self.vlm_strict:
                raise
            return None

        if not description or not description.strip():
            logger.warning(
                "kb.ingest.visual_page_vlm_empty",
                doc_id=doc_id,
                page=page_number,
            )
            return None

        # The VLM narrates the page as an image before transcribing it, and
        # decorates its headings with emoji. The framing is near-identical
        # across every scanned decree, so it retrieves for almost any question
        # about an institutional document.
        description = clean_vlm_output(description)

        logger.info(
            "kb.ingest.visual_page_extracted",
            doc_id=doc_id,
            page=page_number,
            desc_len=len(description),
        )

        return ParsedElement(
            element_type="Image",
            text=description.strip(),
            metadata={
                "page_number": page_number,
                "image_path": image_path,
                "page_classification": "visual",
            },
            content_type=ContentType.FIGURE,
        )

    def _get_native_text_length(
        self,
        pdf_path: str,
        page_number: int,
    ) -> Optional[int]:
        """Length of a PDF page's native (embedded) text layer via PyMuPDF.

        Feeds ``classify_page``'s scan-only detection: near-zero native text
        plus at least one image element is a more reliable scan-only signal than
        Unstructured's per-element garbage-OCR ratio, which misses pages whose
        OCR emits mis-read-but-real-word text as separate text elements.
        Returns ``None`` if the PDF can't be opened or the page is out of range
        (treated as "signal unavailable", not "zero text").
        """
        try:
            import fitz  # PyMuPDF

            doc = fitz.open(pdf_path)
            page_idx = max(0, page_number - 1)
            if page_idx >= len(doc):
                doc.close()
                return None

            text_len = len(doc[page_idx].get_text())
            doc.close()
            return text_len
        except Exception as exc:
            logger.warning(
                "kb.ingest.native_text_length_failed",
                page=page_number,
                error=str(exc),
            )
            return None

    def _extract_page_image(
        self,
        pdf_path: str,
        page_number: int,
        doc_id: str,
    ) -> Optional[str]:
        """Extract a PDF page as a PNG via PyMuPDF. Returns the PNG path, or
        None if extraction failed.
        """
        try:
            import fitz  # PyMuPDF

            doc = fitz.open(pdf_path)
            page_idx = max(0, page_number - 1)
            if page_idx >= len(doc):
                doc.close()
                return None

            page = doc[page_idx]
            mat = fitz.Matrix(150 / 72, 150 / 72)  # 150 DPI
            pix = page.get_pixmap(matrix=mat)

            output_path = os.path.join(
                self.image_dir,
                f"{doc_id}_page_{page_number}.png",
            )
            pix.save(output_path)
            doc.close()
            return output_path
        except Exception as exc:
            logger.error(
                "kb.ingest.image_extraction_failed",
                page=page_number,
                error=str(exc),
            )
            return None
