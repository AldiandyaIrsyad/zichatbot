"""Tests for the shared sentence-splitting helpers used by RAM."""
from __future__ import annotations

from app.guardrails.ram.text_utils import (
    is_unciteable_statement,
    split_sentences,
    split_sentences_and_tail,
    split_sentences_with_seps,
)


class TestSplitSentencesAndTail:
    def test_splits_on_sentence_boundaries(self) -> None:
        pairs, tail = split_sentences_and_tail("Ini kalimat pertama. Ini kalimat kedua!")
        assert pairs == [("Ini kalimat pertama.", " ")]
        assert tail == "Ini kalimat kedua!"

    def test_separator_preserves_paragraph_break(self) -> None:
        pairs, tail = split_sentences_and_tail("Paragraf satu.\n\nParagraf dua.")
        assert pairs == [("Paragraf satu.", "\n\n")]
        assert tail == "Paragraf dua."

    def test_tail_keeps_trailing_whitespace_verbatim(self) -> None:
        # The whole point: streaming callers concatenate the next chunk onto
        # the tail, so stripping here would weld two words together.
        _, tail = split_sentences_and_tail("Ini adalah Peraturan ")
        assert tail == "Ini adalah Peraturan "

    def test_tail_keeps_leading_whitespace_verbatim(self) -> None:
        _, tail = split_sentences_and_tail("   Belum selesai")
        assert tail == "   Belum selesai"

    def test_whitespace_only_input_is_all_tail(self) -> None:
        assert split_sentences_and_tail(" ") == ([], " ")

    def test_numbered_list_markers_stay_attached_to_their_item(self) -> None:
        pairs, tail = split_sentences_and_tail("1. Langkah pertama. 2. Langkah kedua.")
        assert [text for text, _ in pairs] == ["1. Langkah pertama."]
        assert tail == "2. Langkah kedua."

    def test_newline_separated_list_items_split_cleanly(self) -> None:
        pairs, tail = split_sentences_and_tail(
            "1. Langkah pertama\n2. Langkah kedua\n3. Langkah ketiga"
        )
        assert [text for text, _ in pairs] == ["1. Langkah pertama", "2. Langkah kedua"]
        assert tail == "3. Langkah ketiga"

    def test_empty_string(self) -> None:
        assert split_sentences_and_tail("") == ([], "")


class TestSplitSentencesWithSeps:
    def test_unterminated_tail_comes_back_stripped_with_empty_separator(self) -> None:
        assert split_sentences_with_seps("Satu. Dua") == [("Satu.", " "), ("Dua", "")]

    def test_whitespace_only_returns_no_pairs(self) -> None:
        assert split_sentences_with_seps("   \n\n  ") == []

    def test_empty_string_returns_empty_list(self) -> None:
        assert split_sentences_with_seps("") == []


class TestAbbreviations:
    def test_rupiah_amount_is_not_a_sentence_end(self) -> None:
        # "Rp. 500.000" split at "Rp." stranded a claim on the currency prefix,
        # so the verification badge rendered inside the amount ("Rp.i 500.000").
        text = "Biaya UKT adalah Rp. 500.000,- (lima ratus ribu rupiah). Keputusan ini berlaku."
        pairs, tail = split_sentences_and_tail(text)
        assert [t for t, _ in pairs] == ["Biaya UKT adalah Rp. 500.000,- (lima ratus ribu rupiah)."]
        assert tail == "Keputusan ini berlaku."

    def test_titles_are_not_sentence_ends(self) -> None:
        text = "Ditandatangani oleh Prof. Dr. Didi Sukyadi. Berlaku sejak ditetapkan."
        assert split_sentences(text) == [
            "Ditandatangani oleh Prof. Dr. Didi Sukyadi.",
            "Berlaku sejak ditetapkan.",
        ]

    def test_acronym_in_parens_still_ends_a_sentence(self) -> None:
        # The abbreviation guard must not swallow real boundaries.
        text = "Universitas Pendidikan Indonesia (UPI). Berikut penjelasannya."
        assert split_sentences(text) == [
            "Universitas Pendidikan Indonesia (UPI).",
            "Berikut penjelasannya.",
        ]


class TestIsUnciteableStatement:
    """Refusals and advice have no citable source, so badging them
    "Klaim tanpa kutipan sumber" puts a warning on an honest refusal."""

    def test_refusal_about_own_knowledge(self) -> None:
        assert is_unciteable_statement(
            "Berdasarkan konteks, saya tidak memiliki informasi mengenai tarif 2026."
        )

    def test_statement_about_the_context_itself(self) -> None:
        assert is_unciteable_statement(
            "Namun, tidak ada data tarif nominal UKT untuk 2026 dalam konteks yang diberikan."
        )

    def test_advice_to_the_user(self) -> None:
        assert is_unciteable_statement("Saya sarankan untuk merujuk pada peraturan terbaru.")
        assert is_unciteable_statement("Anda perlu mengetahui program studi yang dituju.")
        assert is_unciteable_statement("Silakan hubungi Direktorat Keuangan.")

    def test_real_claims_about_the_corpus_keep_their_badge(self) -> None:
        """A negation about the documents is still a checkable claim."""
        for claim in (
            "Dokumen yang tersedia hanya memuat ketentuan UKT Tahun Akademik 2022/2023.",
            "Dokumen yang tersedia tidak mencakup seluruh program studi.",
            "Kelompok tarif UKT dibagi menjadi 8 kelompok berdasarkan penghasilan.",
            "Tarif UKT Pendidikan Sejarah adalah Rp3.390.000.",
        ):
            assert not is_unciteable_statement(claim), claim

    def test_empty_is_not_unciteable(self) -> None:
        assert not is_unciteable_statement("")
        assert not is_unciteable_statement("   ")
