"""Fast, synchronous PDF text extraction for chat attachments.

Uses PyMuPDF (``fitz``) to pull native embedded text per page. Deliberately
doesn't reuse the KB ingestion parser (``UnstructuredClient``), whose OCR/VLM
enrichment can take ~15 minutes — incompatible with a single chat request.
This trades scanned/image-only PDF support for speed, an acceptable fit for
born-digital legal/regulatory documents. Fulfills
``app/chat/domain/interfaces.py::IAttachmentExtractor``; wired in
``app/chat/dependency.py::get_pdf_text_extractor``.
"""
import fitz  # PyMuPDF
import structlog

from app.chat.domain.interfaces import ExtractedPdfText

logger = structlog.get_logger(__name__)


class PdfExtractionError(Exception):
    """Base exception for PDF attachment extraction failures. The three
    subclasses below correspond to the ``Raises:`` documented on
    ``IAttachmentExtractor.extract``.
    """


class PdfCorruptError(PdfExtractionError):
    """The file could not be opened/parsed as a PDF."""


class PdfTooManyPagesError(PdfExtractionError):
    """The PDF has more pages than the configured cap."""

    def __init__(self, page_count: int, max_pages: int):
        self.page_count = page_count
        self.max_pages = max_pages
        super().__init__(f"PDF has {page_count} pages, exceeding the cap of {max_pages}.")


class PdfNoTextError(PdfExtractionError):
    """No extractable text was found (e.g. a scanned/image-only PDF)."""


class PdfTextExtractor:
    """Extracts native embedded text from a PDF using PyMuPDF."""

    def __init__(self, max_pages: int, max_chars: int):
        """Set the extraction limits: ``max_pages`` raises
        ``PdfTooManyPagesError`` above the cap; ``max_chars`` truncates the
        extracted text (flagged via ``ExtractedPdfText.truncated``).
        """
        self.max_pages = max_pages
        self.max_chars = max_chars

    def extract(self, file_bytes: bytes) -> ExtractedPdfText:
        """Extract native text from a PDF's bytes.

        Raises:
            PdfCorruptError: The file can't be opened as a PDF.
            PdfTooManyPagesError: Page count exceeds ``max_pages``.
            PdfNoTextError: No extractable text on any page.
        """
        try:
            doc = fitz.open(stream=file_bytes, filetype="pdf")
        except Exception as e:
            logger.warning("pdf_extractor.open_failed", error=str(e))
            raise PdfCorruptError("Could not open file as a PDF.") from e

        try:
            page_count = doc.page_count
            if page_count > self.max_pages:
                raise PdfTooManyPagesError(page_count, self.max_pages)

            page_texts = [page.get_text() for page in doc]
        finally:
            doc.close()

        full_text = "\n\n".join(t.strip() for t in page_texts if t.strip())

        if not full_text.strip():
            raise PdfNoTextError(
                "No extractable text found — the PDF may be scanned/image-only."
            )

        truncated = len(full_text) > self.max_chars
        text = full_text[: self.max_chars]

        return ExtractedPdfText(
            text=text,
            page_count=page_count,
            char_count=len(text),
            truncated=truncated,
        )
