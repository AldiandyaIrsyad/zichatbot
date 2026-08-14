"""Tests for PdfTextExtractor (fast PyMuPDF-based extraction for chat attachments)."""

from __future__ import annotations

import fitz
import pytest

from app.chat.infra.pdf_text_extractor import (
    PdfCorruptError,
    PdfNoTextError,
    PdfTextExtractor,
    PdfTooManyPagesError,
)


def _make_pdf_bytes(page_texts: list[str]) -> bytes:
    """Build a minimal in-memory PDF with one page per string in page_texts."""
    doc = fitz.open()
    for text in page_texts:
        page = doc.new_page()
        if text:
            page.insert_text((72, 72), text)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


class TestExtract:
    def test_extracts_text_from_each_page(self) -> None:
        extractor = PdfTextExtractor(max_pages=10, max_chars=10_000)
        pdf_bytes = _make_pdf_bytes(["Halaman pertama", "Halaman kedua"])

        result = extractor.extract(pdf_bytes)

        assert "Halaman pertama" in result.text
        assert "Halaman kedua" in result.text
        assert result.page_count == 2
        assert result.truncated is False

    def test_too_many_pages_raises(self) -> None:
        extractor = PdfTextExtractor(max_pages=2, max_chars=10_000)
        pdf_bytes = _make_pdf_bytes(["Satu", "Dua", "Tiga"])

        with pytest.raises(PdfTooManyPagesError):
            extractor.extract(pdf_bytes)

    def test_no_extractable_text_raises(self) -> None:
        extractor = PdfTextExtractor(max_pages=10, max_chars=10_000)
        pdf_bytes = _make_pdf_bytes(["", ""])

        with pytest.raises(PdfNoTextError):
            extractor.extract(pdf_bytes)

    def test_corrupt_file_raises(self) -> None:
        extractor = PdfTextExtractor(max_pages=10, max_chars=10_000)

        with pytest.raises(PdfCorruptError):
            extractor.extract(b"this is not a pdf")

    def test_truncates_and_flags_when_over_char_cap(self) -> None:
        extractor = PdfTextExtractor(max_pages=10, max_chars=5)
        pdf_bytes = _make_pdf_bytes(["Ini teks yang cukup panjang untuk dipotong"])

        result = extractor.extract(pdf_bytes)

        assert result.truncated is True
        assert len(result.text) == 5
