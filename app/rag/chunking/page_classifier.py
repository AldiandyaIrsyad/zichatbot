"""Per-page classification for the hybrid PDF processing pipeline.

Each page is independently classified into a :class:`PageType` from the ratio
of visual-garbage elements the parser detected, so :class:`IngestWorker` can
route visual-heavy pages (SOP flowcharts, diagrams) to VLM full-page extraction
while keeping text-rich and table-rich pages in the standard pipeline.

Design:
- Page-scoped, not document-scoped, so mixed documents (formal-text cover pages
  plus flowchart body pages) are handled without any per-document label.
- ``VISUAL`` replaces the parser output entirely; ``TABLE_RICH`` only
  transforms table elements in place. A page can be both ``TABLE_RICH`` and
  ``TEXT_RICH``.
- Garbage detection keys on text length and alnum density — simpler and more
  robust than pattern-matching repeated characters, which has too many OCR
  edge cases.

Pure Python (no infra imports), per the ``thesis/`` purity rule.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

from app.rag.chunking.models import ContentType, ParsedElement


class PageType(str, Enum):
    """The dominant structural type of a PDF page.

    TEXT_RICH: primarily narrative text/titles/list items (parser output used
    as-is). TABLE_RICH: has well-formed HTML tables (converted to Markdown;
    other elements kept). VISUAL: dominated by images/figures with garbage or
    empty OCR (flowcharts, diagrams, scans) — parser output discarded, VLM
    renders the full page. MIXED: substantive text plus visual elements but not
    VISUAL — treated as TEXT_RICH with table conversion.
    """

    TEXT_RICH = "text_rich"
    TABLE_RICH = "table_rich"
    VISUAL = "visual"
    MIXED = "mixed"


@dataclass(frozen=True)
class PageClassification:
    """Classification result for a single PDF page: its 1-indexed number,
    dominant :class:`PageType`, element/image/table/text counts, the count of
    garbage (empty/short-text) image elements, and the derived ``image_ratio``
    (image/element) and ``garbage_ratio`` (garbage/image, 0 if no images).
    """

    page_number: int
    page_type: PageType
    element_count: int
    image_count: int
    table_count: int
    text_element_count: int
    garbage_image_count: int
    image_ratio: float
    garbage_ratio: float


# Element types that count as "text" for classification
_TEXT_ELEMENT_TYPES = frozenset({
    "Title", "NarrativeText", "ListItem", "Header",
    "Footer", "UncategorizedText", "Address", "FigureCaption",
    "EmailAddress", "Formula",
})

# Element types that count as "image/visual" for classification
_VISUAL_ELEMENT_TYPES = frozenset({"Image", "Figure"})

# Element types that count as "table" for classification
_TABLE_ELEMENT_TYPES = frozenset({"Table"})

# Max text length for an image element to count as "garbage" OCR purely on
# length (empty strings, single-char noise like "L"/"6", 2-3 char fragments).
_GARBAGE_TEXT_MAX_LEN = 3

# For longer-but-still-short OCR text, also flag as garbage when it's mostly
# non-alphanumeric (e.g. "~   ~if ~!11" from a misread letterhead/seal) —
# catches noise the length check misses without misclassifying short legitimate
# text (page numbers, short captions), which is predominantly alphanumeric.
_GARBAGE_DENSITY_MAX_LEN = 20
_GARBAGE_MIN_ALNUM_RATIO = 0.5


def _is_garbage_ocr_text(text: str) -> bool:
    """Heuristic garbage-OCR detector for one image/figure element's text.

    Combines a length check (very short text is always garbage) with a
    length-and-density check (short text that's mostly non-alphanumeric, e.g.
    OCR noise from a misread letterhead).
    """
    stripped = text.strip()
    if len(stripped) <= _GARBAGE_TEXT_MAX_LEN:
        return True
    if len(stripped) <= _GARBAGE_DENSITY_MAX_LEN:
        alnum_count = sum(1 for ch in stripped if ch.isalnum())
        alnum_ratio = alnum_count / len(stripped)
        if alnum_ratio < _GARBAGE_MIN_ALNUM_RATIO:
            return True
    return False

# Default thresholds for VISUAL classification
DEFAULT_IMAGE_RATIO_THRESHOLD = 0.5
DEFAULT_GARBAGE_RATIO_THRESHOLD = 0.7

# A page whose PDF-native text layer is at or below this many characters is
# treated as scan-only regardless of the element-level garbage ratio. This
# catches a failure mode the garbage-ratio check structurally can't: the
# parser's OCR pass over a scan-only page sometimes emits its noisy output as
# separate NarrativeText/UncategorizedText elements (mis-read but real-word-
# shaped text) rather than as the Image element's own text. Those fragments are
# too long and alnum-dense to trip ``_is_garbage_ocr_text``, so they inflate
# text_element_count and dilute image_ratio below threshold. A near-empty native
# text layer (PyMuPDF's ``page.get_text()``) is a more direct signal — near-zero
# iff the page is genuinely a scanned image with no real text layer.
NATIVE_TEXT_LEN_THRESHOLD = 30

# Prompt for VLM full-page extraction (Indonesian output)
VLM_PAGE_EXTRACTION_PROMPT = (
    "Ekstrak SEMUA konten dari halaman dokumen ini dalam format Markdown yang bersih. "
    "Untuk tabel, gunakan sintaks tabel Markdown. "
    "Untuk bagan alir (flowchart) atau diagram proses, deskripsikan setiap langkah "
    "secara berurutan menggunakan daftar bernomor, sertakan aktor yang terlibat, "
    "keputusan (decision points), dan urutan kejadian. "
    "Untuk teks biasa, pertahankan heading dan paragraf. "
    "Untuk prosedur atau SOP, jelaskan setiap langkah dengan jelas. "
    "Keluarkan HANYA konten Markdown, tanpa komentar tambahan. "
    "Gunakan Bahasa Indonesia."
)


def classify_page(
    elements: List[ParsedElement],
    page_number: int,
    image_ratio_threshold: float = DEFAULT_IMAGE_RATIO_THRESHOLD,
    garbage_ratio_threshold: float = DEFAULT_GARBAGE_RATIO_THRESHOLD,
    native_text_len: Optional[int] = None,
) -> PageClassification:
    """Classify a single PDF page based on its Unstructured element composition.

    The classification drives downstream routing in the ingestion pipeline:
    - ``VISUAL`` → discard Unstructured output, run VLM full-page extraction
    - ``TABLE_RICH`` → convert HTML tables to Markdown, keep text elements
    - ``TEXT_RICH`` / ``MIXED`` → keep Unstructured output as-is (with table conversion)

    Args:
        elements: All ParsedElement objects whose ``page_number`` metadata
            matches this page. Must be non-empty.
        page_number: The 1-indexed page number (for the result dataclass only).
        image_ratio_threshold: Minimum fraction of elements that must be images
            for a page to be considered potentially VISUAL. Default 0.5.
        garbage_ratio_threshold: Minimum fraction of image elements that must
            have garbage text (≤ 3 chars) for the page to be classified VISUAL.
            Default 0.7.
        native_text_len: Length of the PDF page's native (embedded) text
            layer, e.g. from PyMuPDF's ``page.get_text()``, if the caller has
            it available. When at or below ``NATIVE_TEXT_LEN_THRESHOLD`` and
            the page has at least one image element, the page is classified
            VISUAL regardless of the garbage-ratio check — see
            ``NATIVE_TEXT_LEN_THRESHOLD`` docstring for why the garbage-ratio
            check alone misses this case. ``None`` (default) skips this
            signal entirely, preserving prior behavior for callers that
            don't have PDF-level access (this module stays infra-free).

    Returns:
        :class:`PageClassification` describing this page's type and statistics.
    """
    if not elements:
        return PageClassification(
            page_number=page_number,
            page_type=PageType.TEXT_RICH,
            element_count=0,
            image_count=0,
            table_count=0,
            text_element_count=0,
            garbage_image_count=0,
            image_ratio=0.0,
            garbage_ratio=0.0,
        )

    total = len(elements)
    image_count = 0
    table_count = 0
    text_element_count = 0
    garbage_image_count = 0

    for el in elements:
        etype = el.element_type
        if etype in _VISUAL_ELEMENT_TYPES:
            image_count += 1
            if _is_garbage_ocr_text(el.text):
                garbage_image_count += 1
        elif etype in _TABLE_ELEMENT_TYPES:
            table_count += 1
        elif etype in _TEXT_ELEMENT_TYPES:
            text_element_count += 1
        # Unknown types not counted in any category

    image_ratio = image_count / total
    garbage_ratio = garbage_image_count / image_count if image_count > 0 else 0.0

    is_scan_only = (
        native_text_len is not None
        and native_text_len <= NATIVE_TEXT_LEN_THRESHOLD
        and image_count > 0
    )

    # --- Classification logic ---
    # VISUAL: more than half the elements are images AND most image OCR is
    # garbage (flowchart/diagram signal), OR the page's native PDF text
    # layer is near-empty despite having an image (scan-only signal — see
    # NATIVE_TEXT_LEN_THRESHOLD).
    if (image_ratio >= image_ratio_threshold and garbage_ratio >= garbage_ratio_threshold) or is_scan_only:
        page_type = PageType.VISUAL
    elif table_count > 0:
        # Any substantive tables present → TABLE_RICH (may also have text)
        page_type = PageType.TABLE_RICH
    elif image_count > 0 and text_element_count > 0:
        # Some images but mostly text — don't go VLM, treat as mixed
        page_type = PageType.MIXED
    else:
        page_type = PageType.TEXT_RICH

    return PageClassification(
        page_number=page_number,
        page_type=page_type,
        element_count=total,
        image_count=image_count,
        table_count=table_count,
        text_element_count=text_element_count,
        garbage_image_count=garbage_image_count,
        image_ratio=image_ratio,
        garbage_ratio=garbage_ratio,
    )


def group_elements_by_page(
    elements: List[ParsedElement],
) -> Dict[Optional[int], List[ParsedElement]]:
    """Group parsed elements by their page_number metadata.

    Elements without a page_number are grouped under the key ``None``
    and treated as TEXT_RICH during classification.

    Args:
        elements: Flat list of parsed elements from the document parser.

    Returns:
        Dictionary mapping page_number (int | None) → list of elements.
    """
    groups: Dict[Optional[int], List[ParsedElement]] = {}
    for el in elements:
        page = el.metadata.get("page_number")
        groups.setdefault(page, []).append(el)
    return groups


def classify_all_pages(
    elements: List[ParsedElement],
    image_ratio_threshold: float = DEFAULT_IMAGE_RATIO_THRESHOLD,
    garbage_ratio_threshold: float = DEFAULT_GARBAGE_RATIO_THRESHOLD,
) -> Dict[Optional[int], PageClassification]:
    """Classify all pages in a document from a flat element list.

    Convenience wrapper around :func:`group_elements_by_page` and
    :func:`classify_page`.

    Args:
        elements: All parsed elements from the document parser.
        image_ratio_threshold: See :func:`classify_page`.
        garbage_ratio_threshold: See :func:`classify_page`.

    Returns:
        Dictionary mapping page_number → :class:`PageClassification`.
    """
    groups = group_elements_by_page(elements)
    return {
        page: classify_page(
            page_elements,
            page_number=page or 0,
            image_ratio_threshold=image_ratio_threshold,
            garbage_ratio_threshold=garbage_ratio_threshold,
        )
        for page, page_elements in groups.items()
    }
