"""Tests for the shared sentence-splitting helper used by RAM."""
from __future__ import annotations

from app.guardrails.ram.text_utils import split_sentences, split_table_windows


class TestSplitSentences:
    def test_splits_on_sentence_boundaries(self) -> None:
        result = split_sentences("Ini kalimat pertama. Ini kalimat kedua!")
        assert result == ["Ini kalimat pertama.", "Ini kalimat kedua!"]

    def test_numbered_list_markers_stay_attached_to_their_item(self) -> None:
        text = "1. Langkah pertama. 2. Langkah kedua."
        result = split_sentences(text)
        assert result == ["1. Langkah pertama.", "2. Langkah kedua."]

    def test_numbered_list_without_trailing_period_not_split_at_marker(self) -> None:
        # No real sentence boundary here besides the list markers' own
        # periods — the whole thing should stay as one unit rather than
        # being torn apart at "1." / "2.".
        text = "1. Ajukan surat ke bagian TU 2. Tunggu verifikasi"
        result = split_sentences(text)
        assert result == [text]

    def test_newline_separated_list_items_split_cleanly(self) -> None:
        text = "1. Langkah pertama\n2. Langkah kedua\n3. Langkah ketiga"
        result = split_sentences(text)
        assert result == ["1. Langkah pertama", "2. Langkah kedua", "3. Langkah ketiga"]

    def test_multiple_newlines_collapse_to_single_boundary(self) -> None:
        result = split_sentences("Paragraf satu.\n\n\nParagraf dua.")
        assert result == ["Paragraf satu.", "Paragraf dua."]

    def test_empty_string_returns_empty_list(self) -> None:
        assert split_sentences("") == []

    def test_whitespace_only_returns_empty_list(self) -> None:
        assert split_sentences("   \n\n  ") == []


def _make_table(num_data_rows: int) -> str:
    header = "| Nama | Nilai |"
    separator = "| --- | --- |"
    rows = [f"| Baris{i} | {i} |" for i in range(1, num_data_rows + 1)]
    return "\n".join([header, separator] + rows)


class TestSplitTableWindows:
    def test_header_repeated_in_every_window_not_just_the_first(self) -> None:
        # Regression test for the reported bug: with 7 data rows and the
        # default window size of 3 / step 2, later windows must still
        # carry the header — not just the first one.
        table = _make_table(7)
        windows = split_table_windows(table)
        assert len(windows) > 1
        for window in windows:
            assert window.startswith("| Nama | Nilai |")
            assert "| --- | --- |" in window

    def test_overlap_behavior(self) -> None:
        table = _make_table(5)
        windows = split_table_windows(table, rows_per_window=3, row_step=2)
        # Window 0: rows 1-3, window 1: rows 3-5 (1-row overlap on "Baris3")
        assert "Baris1" in windows[0] and "Baris3" in windows[0]
        assert "Baris3" in windows[1] and "Baris5" in windows[1]

    def test_no_synthetic_trailing_period_appended(self) -> None:
        table = _make_table(2)
        windows = split_table_windows(table)
        for window in windows:
            assert not window.rstrip().endswith(".")

    def test_non_markdown_table_returns_empty_list(self) -> None:
        assert split_table_windows("Ini bukan tabel, hanya paragraf biasa.") == []
        assert split_table_windows("<table><tr><td>1</td></tr></table>") == []

    def test_single_data_row_produces_one_window_with_header(self) -> None:
        table = _make_table(1)
        windows = split_table_windows(table)
        assert len(windows) == 1
        assert "| Nama | Nilai |" in windows[0]
        assert "Baris1" in windows[0]

    def test_empty_string_returns_empty_list(self) -> None:
        assert split_table_windows("") == []
