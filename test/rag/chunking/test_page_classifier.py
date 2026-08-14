"""Unit tests for the per-page classifier module."""

import pytest
from typing import Any, Dict, List

from app.rag.chunking.models import ParsedElement
from app.rag.chunking.page_classifier import (
    PageType,
    PageClassification,
    classify_page,
    classify_all_pages,
    group_elements_by_page,
    VLM_PAGE_EXTRACTION_PROMPT,
    _is_garbage_ocr_text,
)


def _make_el(
    element_type: str,
    text: str = "",
    page: int = 1,
    extra_meta: Dict[str, Any] = {},
) -> ParsedElement:
    """Helper to quickly build a ParsedElement for tests."""
    return ParsedElement(
        element_type=element_type,
        text=text,
        metadata={"page_number": page, **extra_meta},
    )


# ---------------------------------------------------------------------------
# classify_page — TEXT_RICH pages
# ---------------------------------------------------------------------------

class TestClassifyPageTextRich:
    """A page with only narrative/title elements is TEXT_RICH."""

    def test_all_narrative_text(self) -> None:
        elements = [
            _make_el("NarrativeText", "Teks paragraf pertama."),
            _make_el("NarrativeText", "Teks paragraf kedua."),
            _make_el("Title", "Judul Bab"),
        ]
        cls = classify_page(elements, page_number=1)
        assert cls.page_type == PageType.TEXT_RICH
        assert cls.image_count == 0
        assert cls.table_count == 0
        assert cls.text_element_count == 3

    def test_empty_elements_returns_text_rich(self) -> None:
        cls = classify_page([], page_number=1)
        assert cls.page_type == PageType.TEXT_RICH
        assert cls.element_count == 0

    def test_list_items_are_text(self) -> None:
        elements = [_make_el("ListItem", "Item satu"), _make_el("ListItem", "Item dua")]
        cls = classify_page(elements, page_number=2)
        assert cls.page_type == PageType.TEXT_RICH


# ---------------------------------------------------------------------------
# classify_page — TABLE_RICH pages
# ---------------------------------------------------------------------------

class TestClassifyPageTableRich:
    """A page with at least one Table element is TABLE_RICH."""

    def test_single_table_element(self) -> None:
        elements = [
            _make_el("Title", "Tabel Data"),
            _make_el("Table", "<table><tr><td>A</td></tr></table>"),
        ]
        cls = classify_page(elements, page_number=1)
        assert cls.page_type == PageType.TABLE_RICH
        assert cls.table_count == 1

    def test_table_with_text_is_table_rich(self) -> None:
        """A page with a table AND narrative text is still TABLE_RICH."""
        elements = [
            _make_el("NarrativeText", "Penjelasan tabel berikut."),
            _make_el("Table", "<table><tr><td>Data</td></tr></table>"),
            _make_el("NarrativeText", "Tabel di atas menunjukkan..."),
        ]
        cls = classify_page(elements, page_number=3)
        assert cls.page_type == PageType.TABLE_RICH

    def test_multiple_tables(self) -> None:
        elements = [
            _make_el("Table", "<table><tr><td>A</td></tr></table>"),
            _make_el("Table", "<table><tr><td>B</td></tr></table>"),
        ]
        cls = classify_page(elements, page_number=4)
        assert cls.page_type == PageType.TABLE_RICH
        assert cls.table_count == 2


# ---------------------------------------------------------------------------
# classify_page — VISUAL pages
# ---------------------------------------------------------------------------

class TestClassifyPageVisual:
    """SOP flowchart pages: many images, most with empty/garbage text."""

    def test_typical_sop_flowchart_page(self) -> None:
        """Simulates a flowchart page: 80% images, 90% garbage (≤3 chars)."""
        elements = (
            [_make_el("Image", "") for _ in range(8)]   # 8 empty images
            + [_make_el("Image", "L") for _ in range(1)]  # 1 garbage image
            + [_make_el("UncategorizedText", "6")]        # 1 garbage text
        )
        cls = classify_page(elements, page_number=2)
        assert cls.page_type == PageType.VISUAL
        assert cls.image_count == 9
        assert cls.garbage_image_count == 9  # all 9 images have ≤3 char text
        assert cls.image_ratio >= 0.5
        assert cls.garbage_ratio >= 0.7

    def test_majority_images_with_garbage_text(self) -> None:
        """Exactly at the threshold: 50% images, 70% garbage."""
        # 5 images, 4 garbage (80%), 5 text elements → image_ratio = 0.5
        elements = (
            [_make_el("Image", "") for _ in range(4)]
            + [_make_el("Image", "Valid OCR text here")]
            + [_make_el("NarrativeText", "Teks biasa.") for _ in range(5)]
        )
        cls = classify_page(elements, page_number=3)
        # image_ratio = 5/10 = 0.5 → meets threshold
        # garbage_ratio = 4/5 = 0.8 → meets threshold
        assert cls.page_type == PageType.VISUAL

    def test_high_image_count_but_low_garbage_ratio_is_not_visual(self) -> None:
        """60% images but most have meaningful text → not VISUAL."""
        elements = (
            [_make_el("Image", "Deskripsi gambar yang cukup panjang") for _ in range(6)]
            + [_make_el("NarrativeText", "Teks.") for _ in range(4)]
        )
        cls = classify_page(elements, page_number=4)
        assert cls.page_type != PageType.VISUAL

    def test_low_image_ratio_is_not_visual(self) -> None:
        """Only 20% images even if all garbage → not VISUAL."""
        elements = (
            [_make_el("Image", "") for _ in range(2)]
            + [_make_el("NarrativeText", "Teks " * 10) for _ in range(8)]
        )
        cls = classify_page(elements, page_number=5)
        assert cls.page_type != PageType.VISUAL

    def test_custom_thresholds(self) -> None:
        """Custom thresholds can make a borderline page VISUAL."""
        elements = [_make_el("Image", "") for _ in range(3)] + [
            _make_el("NarrativeText", "Satu teks.") for _ in range(7)
        ]
        # Default: image_ratio=0.3 → not VISUAL
        cls_default = classify_page(elements, page_number=1)
        assert cls_default.page_type != PageType.VISUAL

        # Custom lower threshold: image_ratio=0.3 ≥ 0.3 → VISUAL
        cls_custom = classify_page(
            elements,
            page_number=1,
            image_ratio_threshold=0.3,
            garbage_ratio_threshold=0.9,
        )
        assert cls_custom.page_type == PageType.VISUAL


# ---------------------------------------------------------------------------
# classify_page — native_text_len scan-only override
# ---------------------------------------------------------------------------

class TestClassifyPageNativeTextLen:
    """A near-empty native PDF text layer forces VISUAL even when
    Unstructured's own OCR noise slips past the garbage-ratio check by
    surfacing as real-word-shaped (but mis-read) text elements rather than
    as the image element's own text — a real corpus failure mode where a
    2-page scan-only "Keputusan Rektor" decree (0 native chars/page, 1 image
    each) produced chunks like "PNRIMA PENGHARGAAN DESAIN LOGO PINGATAN..."
    (55 chars, mostly alphanumeric — too long/dense for the length+density
    heuristic) sitting alongside a kept empty-text Image element, diluting
    image_ratio below 0.5 and keeping the page out of VISUAL.
    """

    def test_scan_only_page_forced_visual_despite_real_looking_ocr_noise(self) -> None:
        elements = [
            _make_el("Image", ""),
            _make_el("NarrativeText", "PNRIMA PENGHARGAAN DESAIN LOGO PINGATAN DIES NATALIS U"),
        ]
        # Without the native-text signal: image_ratio=0.5, garbage_ratio=1.0
        # (the Image's own text is empty) — already VISUAL by ratio alone in
        # this 2-element case, so widen it to make the ratio-based path fail
        # on its own, isolating the native_text_len signal.
        elements += [_make_el("NarrativeText", "Baris OCR keliru lainnya.") for _ in range(3)]
        cls_without_signal = classify_page(elements, page_number=1)
        assert cls_without_signal.page_type != PageType.VISUAL  # confirms ratio path alone fails here

        cls_with_signal = classify_page(elements, page_number=1, native_text_len=0)
        assert cls_with_signal.page_type == PageType.VISUAL

    def test_native_text_len_none_preserves_prior_behavior(self) -> None:
        """Default (no PDF-level signal available) is unaffected."""
        elements = [
            _make_el("Image", "Deskripsi gambar yang cukup panjang") for _ in range(6)
        ] + [_make_el("NarrativeText", "Teks.") for _ in range(4)]
        cls = classify_page(elements, page_number=1, native_text_len=None)
        assert cls.page_type != PageType.VISUAL

    def test_native_text_len_above_threshold_does_not_force_visual(self) -> None:
        """A real text-rich page with one legitimate embedded figure isn't
        misclassified just because it happens to have an image element."""
        elements = [
            _make_el("NarrativeText", "Paragraf isi dokumen yang panjang.") for _ in range(5)
        ] + [_make_el("Image", "Grafik pendukung dengan keterangan jelas.")]
        cls = classify_page(elements, page_number=1, native_text_len=500)
        assert cls.page_type != PageType.VISUAL

    def test_native_text_len_requires_at_least_one_image(self) -> None:
        """Near-zero native text with no image element shouldn't force
        VISUAL — there'd be nothing for the VLM to render/extract from."""
        elements = [_make_el("NarrativeText", "x")]
        cls = classify_page(elements, page_number=1, native_text_len=0)
        assert cls.page_type != PageType.VISUAL

    def test_native_text_len_at_threshold_boundary(self) -> None:
        elements = [_make_el("Image", "")] + [
            _make_el("NarrativeText", "Teks OCR keliru yang cukup panjang untuk lolos.") for _ in range(3)
        ]
        at_threshold = classify_page(elements, page_number=1, native_text_len=30)
        assert at_threshold.page_type == PageType.VISUAL

        above_threshold = classify_page(elements, page_number=1, native_text_len=31)
        assert above_threshold.page_type != PageType.VISUAL


# ---------------------------------------------------------------------------
# classify_page — MIXED pages
# ---------------------------------------------------------------------------

class TestClassifyPageMixed:
    """Pages with some images but dominated by text → MIXED."""

    def test_mixed_image_and_text(self) -> None:
        elements = [
            _make_el("NarrativeText", "Teks panjang " * 5),
            _make_el("Image", "Gambar dengan caption yang berarti"),
            _make_el("NarrativeText", "Teks lanjutan " * 5),
        ]
        cls = classify_page(elements, page_number=1)
        assert cls.page_type == PageType.MIXED


# ---------------------------------------------------------------------------
# group_elements_by_page
# ---------------------------------------------------------------------------

class TestGroupElementsByPage:
    def test_groups_by_page_number(self) -> None:
        elements = [
            _make_el("Title", "Judul", page=1),
            _make_el("NarrativeText", "Teks", page=1),
            _make_el("Table", "<table/>", page=2),
        ]
        groups = group_elements_by_page(elements)
        assert set(groups.keys()) == {1, 2}
        assert len(groups[1]) == 2
        assert len(groups[2]) == 1

    def test_elements_without_page_go_to_none(self) -> None:
        el = ParsedElement(element_type="NarrativeText", text="No page", metadata={})
        groups = group_elements_by_page([el])
        assert None in groups
        assert len(groups[None]) == 1


# ---------------------------------------------------------------------------
# classify_all_pages
# ---------------------------------------------------------------------------

class TestClassifyAllPages:
    def test_multi_page_document(self) -> None:
        elements = (
            # Page 1: text-rich
            [_make_el("NarrativeText", "Teks halaman 1.", page=1) for _ in range(3)]
            # Page 2: visual (SOP flowchart)
            + [_make_el("Image", "", page=2) for _ in range(8)]
            + [_make_el("UncategorizedText", "6", page=2) for _ in range(2)]
            # Page 3: table-rich
            + [
                _make_el("NarrativeText", "Penjelasan.", page=3),
                _make_el("Table", "<table/>", page=3),
            ]
        )
        result = classify_all_pages(elements)
        assert result[1].page_type == PageType.TEXT_RICH
        assert result[2].page_type == PageType.VISUAL
        assert result[3].page_type == PageType.TABLE_RICH


# ---------------------------------------------------------------------------
# VLM_PAGE_EXTRACTION_PROMPT
# ---------------------------------------------------------------------------

class TestVLMPrompt:
    def test_prompt_is_indonesian(self) -> None:
        # Spot-check for Indonesian keywords
        assert "Markdown" in VLM_PAGE_EXTRACTION_PROMPT
        assert "Bahasa Indonesia" in VLM_PAGE_EXTRACTION_PROMPT
        assert "flowchart" in VLM_PAGE_EXTRACTION_PROMPT.lower() or "bagan" in VLM_PAGE_EXTRACTION_PROMPT.lower()

    def test_prompt_is_non_empty(self) -> None:
        assert len(VLM_PAGE_EXTRACTION_PROMPT) > 50


# ---------------------------------------------------------------------------
# _is_garbage_ocr_text — length + density heuristic
# ---------------------------------------------------------------------------

class TestIsGarbageOcrText:
    """Regression coverage for the garbage-OCR detector.

    The original _GARBAGE_TEXT_MAX_LEN=3 check missed longer noise like
    '~   ~if ~!11' (12 chars, a real misread letterhead from the corpus) —
    it needs both the pure-length check (still short noise) and a
    length+density check (longer noise that's mostly non-alphanumeric).
    """

    def test_empty_text_is_garbage(self) -> None:
        assert _is_garbage_ocr_text("") is True

    def test_tiny_noise_is_garbage(self) -> None:
        assert _is_garbage_ocr_text("L") is True
        assert _is_garbage_ocr_text("6") is True
        assert _is_garbage_ocr_text("qp") is True

    def test_real_corpus_example_is_now_garbage(self) -> None:
        """'~   ~if ~!11' (12 chars) previously passed the ≤3-char check."""
        assert _is_garbage_ocr_text("~   ~if ~!11") is True

    def test_short_legitimate_text_is_not_garbage(self) -> None:
        """Short but real, mostly-alphanumeric OCR text should survive."""
        assert _is_garbage_ocr_text("Rp5.000.000") is False
        assert _is_garbage_ocr_text("Halaman 3") is False
        assert _is_garbage_ocr_text("Gambar 1") is False

    def test_long_real_sentence_is_not_garbage(self) -> None:
        assert _is_garbage_ocr_text(
            "PERATURAN REKTOR UNIVERSITAS PENDIDIKAN INDONESIA"
        ) is False

    def test_long_mostly_symbolic_text_beyond_density_window_not_flagged(self) -> None:
        """Density check only applies up to _GARBAGE_DENSITY_MAX_LEN — longer
        symbol-heavy strings aren't covered by this heuristic (by design, to
        avoid false-positiving on legitimate long text with punctuation)."""
        long_noise = "~" * 25
        assert _is_garbage_ocr_text(long_noise) is False
