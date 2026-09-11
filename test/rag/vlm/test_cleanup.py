"""Tests for stripping image-meta framing from VLM output.

The framing ("Gambar ini adalah halaman ketiga dari sebuah dokumen resmi…")
is near-identical across every scanned decree, so it retrieves for almost any
question about an institutional document. Stripping it must never take the
transcription with it.
"""
from __future__ import annotations

from app.rag.vlm.cleanup import (
    clean_vlm_output,
    looks_like_vlm_meta_description,
    strip_emoji,
    strip_vlm_meta_description,
)

REAL_SAMPLE = """Gambar ini adalah halaman ketiga dari sebuah dokumen resmi, yaitu **Peraturan Rektor Universitas Pendidikan Indonesia (UPI)** tentang ketentuan Tarif UKT. Dokumen ini berisi Pasal 3 hingga Pasal 6.

---

### **Deskripsi Detail Dokumen**

---

## **Pasal 3: Kelompok Tarif UKT Berdasarkan Kemampuan Ekonomi**

Pasal ini mengatur pembagian mahasiswa baru menjadi 8 kelompok tarif UKT.

| Kode | Kemampuan Ekonomi | Kelompok Tarif UKT |
|------|-------------------|--------------------|
| a.   | Penghasilan <= Rp500.000 | Kelompok I |
| h.   | Penghasilan > Rp9.000.001 | Kelompok VIII |"""


class TestStripsFraming:
    def test_removes_opening_narration(self) -> None:
        out = strip_vlm_meta_description(REAL_SAMPLE)
        assert "Gambar ini adalah halaman" not in out
        assert out.startswith("## **Pasal 3:")

    def test_removes_boilerplate_heading(self) -> None:
        assert "Deskripsi Detail Dokumen" not in strip_vlm_meta_description(REAL_SAMPLE)

    def test_keeps_the_transcription_verbatim(self) -> None:
        out = strip_vlm_meta_description(REAL_SAMPLE)
        assert "| h.   | Penghasilan > Rp9.000.001 | Kelompok VIII |" in out
        assert "8 kelompok tarif UKT" in out

    def test_removes_handoff_connector_line(self) -> None:
        text = "Gambar yang disajikan adalah halaman pertama.\n\nBerikut adalah deskripsi detail dari dokumen ini:\n\n### Pasal 1\n\nIsi pasal."
        out = strip_vlm_meta_description(text)
        assert "Berikut adalah deskripsi" not in out
        assert out.startswith("### Pasal 1")


class TestLeavesRealContentAlone:
    def test_untouched_when_no_framing(self) -> None:
        text = "## Pasal 1\n\nSetiap mahasiswa wajib membayar UKT.\n\n## Pasal 2\n\nPembayaran dilakukan tiap semester."
        assert strip_vlm_meta_description(text) == text

    def test_mid_document_figure_reference_is_not_narration(self) -> None:
        # "Gambar 2 menunjukkan…" inside a transcription is real content; only
        # a *leading* narration paragraph is framing.
        text = "## Pasal 5\n\nGambar 2 menunjukkan alur pengajuan keringanan UKT.\n\nPemohon mengisi formulir."
        assert strip_vlm_meta_description(text) == text

    def test_table_only_chunk_untouched(self) -> None:
        text = "| Kode | Kelompok |\n| --- | --- |\n| a. | Kelompok I |"
        assert strip_vlm_meta_description(text) == text

    def test_pure_narration_is_kept_rather_than_emptied(self) -> None:
        # Nothing left to keep -> return the original, so the chunk still has
        # (weak) content instead of becoming empty.
        text = "Gambar ini adalah halaman kedua dari sebuah dokumen resmi."
        assert strip_vlm_meta_description(text) == text

    def test_empty_and_blank_input(self) -> None:
        assert strip_vlm_meta_description("") == ""
        assert strip_vlm_meta_description("   ") == "   "


class TestDetector:
    def test_detects_narration_and_headings(self) -> None:
        assert looks_like_vlm_meta_description(REAL_SAMPLE)
        assert looks_like_vlm_meta_description("### Deskripsi Umum Dokumen\n\nIsi.")

    def test_does_not_flag_clean_content(self) -> None:
        assert not looks_like_vlm_meta_description("## Pasal 1\n\nSetiap mahasiswa wajib membayar UKT.")
        assert not looks_like_vlm_meta_description("")


class TestEmojiDecoratedHeadings:
    """The VLM decorates its boilerplate headings with emoji, which slipped
    past the first version of the pattern and survived a corpus-wide clean."""

    def test_emoji_heading_is_removed(self) -> None:
        text = "## \U0001F4C4 **Deskripsi Umum Dokumen**\n\n- **Judul**: KEPUTUSAN REKTOR\n\n## Pasal 1\n\nIsi pasal."
        out = strip_vlm_meta_description(text)
        assert "Deskripsi Umum Dokumen" not in out
        assert out.startswith("- **Judul**: KEPUTUSAN REKTOR")

    def test_emoji_heading_is_detected(self) -> None:
        assert looks_like_vlm_meta_description("### \U0001F510 **Analisis Detail**\n\nIsi.")

    def test_real_heading_with_a_similar_word_survives(self) -> None:
        text = "## Ketentuan Umum\n\nIsi pasal."
        assert strip_vlm_meta_description(text) == text


class TestStripEmoji:
    """The VLM decorates its output with emoji the source document never had.
    They cost embedding tokens and leak into answers and citation badges."""

    def test_heading_decoration_collapses_cleanly(self) -> None:
        assert strip_emoji("## \U0001F4C4 **Deskripsi Umum**") == "## **Deskripsi Umum**"

    def test_leading_emoji_is_removed_without_leaving_indent(self) -> None:
        assert strip_emoji("\u2705 **Dokumen ini sah.**") == "**Dokumen ini sah.**"

    def test_preserves_the_glyphs_a_tariff_table_needs(self) -> None:
        # "<=" is load-bearing in the UKT bracket table; en dashes and curly
        # quotes are ordinary legal typography.
        row = "| a. | Penghasilan \u2264 Rp. 500.000 | Kelompok I |"
        assert strip_emoji(row) == row
        assert strip_emoji("Rp 500.001 \u2013 Rp 1.000.000") == "Rp 500.001 \u2013 Rp 1.000.000"
        assert strip_emoji("Pasal 3 \u2192 Pasal 6 mengatur \u201ctarif UKT\u201d.") == (
            "Pasal 3 \u2192 Pasal 6 mengatur \u201ctarif UKT\u201d."
        )

    def test_plain_text_untouched(self) -> None:
        text = "## Pasal 1\n\nSetiap mahasiswa wajib membayar UKT."
        assert strip_emoji(text) == text

    def test_empty_input(self) -> None:
        assert strip_emoji("") == ""


class TestCleanVlmOutput:
    def test_removes_narration_and_emoji_together(self) -> None:
        text = (
            "Gambar ini adalah halaman kedua dari sebuah dokumen resmi.\n\n"
            "## \U0001F4C4 **Deskripsi Umum Dokumen**\n\n"
            "## Pasal 3\n\n"
            "| a. | Penghasilan \u2264 Rp. 500.000 | Kelompok I |"
        )
        out = clean_vlm_output(text)
        assert "Gambar ini adalah" not in out
        assert "Deskripsi Umum Dokumen" not in out
        assert "\U0001F4C4" not in out
        assert "| a. | Penghasilan \u2264 Rp. 500.000 | Kelompok I |" in out
