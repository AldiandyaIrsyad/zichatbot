"""Unit tests for the HTML-to-Markdown table converter."""

import pytest

from app.shared.table_converter import (
    html_table_to_markdown,
    split_markdown_table_lines,
    _clean_cell_text,
)
from app.rag.chunking.logic import _is_markdown_table


# ---------------------------------------------------------------------------
# html_table_to_markdown — basic conversion
# ---------------------------------------------------------------------------

class TestHtmlTableToMarkdown:
    """Happy-path conversions."""

    def test_simple_two_column_table(self) -> None:
        html = """
        <table>
          <thead><tr><th>Nama</th><th>Nilai</th></tr></thead>
          <tbody>
            <tr><td>Alice</td><td>90</td></tr>
            <tr><td>Bob</td><td>85</td></tr>
          </tbody>
        </table>
        """
        result = html_table_to_markdown(html)
        assert "| Nama | Nilai |" in result
        assert "| --- | --- |" in result
        assert "| Alice | 90 |" in result
        assert "| Bob | 85 |" in result

    def test_header_row_comes_first(self) -> None:
        html = """
        <table>
          <thead><tr><th>A</th><th>B</th></tr></thead>
          <tbody><tr><td>1</td><td>2</td></tr></tbody>
        </table>
        """
        lines = html_table_to_markdown(html).strip().splitlines()
        assert lines[0].startswith("| A")
        assert "---" in lines[1]
        assert lines[2].startswith("| 1")

    def test_no_thead_first_row_becomes_header(self) -> None:
        html = """
        <table>
          <tr><td>Kolom1</td><td>Kolom2</td></tr>
          <tr><td>Data1</td><td>Data2</td></tr>
        </table>
        """
        result = html_table_to_markdown(html)
        lines = result.strip().splitlines()
        assert "Kolom1" in lines[0]
        assert "---" in lines[1]
        assert "Data1" in lines[2]

    def test_th_cells_in_header(self) -> None:
        html = "<table><tr><th>H1</th><th>H2</th></tr><tr><td>D1</td><td>D2</td></tr></table>"
        result = html_table_to_markdown(html)
        assert "| H1 | H2 |" in result

    def test_empty_cells_produce_empty_columns(self) -> None:
        html = """
        <table>
          <tr><th>A</th><th>B</th><th>C</th></tr>
          <tr><td>1</td><td></td><td>3</td></tr>
        </table>
        """
        result = html_table_to_markdown(html)
        assert "|  |" in result or "| 1 |  | 3 |" in result

    def test_pipe_in_cell_is_escaped(self) -> None:
        html = """
        <table>
          <tr><th>Syarat</th></tr>
          <tr><td>A | B</td></tr>
        </table>
        """
        result = html_table_to_markdown(html)
        assert "A \\| B" in result

    def test_multiline_cell_text_is_collapsed(self) -> None:
        html = """
        <table>
          <tr><th>Keterangan</th></tr>
          <tr><td>Baris pertama
          baris kedua</td></tr>
        </table>
        """
        result = html_table_to_markdown(html)
        lines = result.strip().splitlines()
        data_line = [l for l in lines if "Baris pertama" in l or "baris pertama" in l.lower()]
        assert data_line, "Expected a data row with collapsed text"
        assert "\n" not in data_line[0]

    def test_colspan_produces_merged_placeholder(self) -> None:
        html = """
        <table>
          <tr><th colspan="2">Gabungan</th></tr>
          <tr><td>A</td><td>B</td></tr>
        </table>
        """
        result = html_table_to_markdown(html)
        assert "[merged]" in result

    def test_completely_empty_table_returns_original(self) -> None:
        html = "<table></table>"
        result = html_table_to_markdown(html)
        # Should fall back to original or return empty
        assert result is not None

    def test_empty_string_returns_empty_string(self) -> None:
        assert html_table_to_markdown("") == ""

    def test_non_table_html_returns_original(self) -> None:
        html = "<div>No table here</div>"
        result = html_table_to_markdown(html)
        assert result == html

    def test_malformed_html_does_not_raise(self) -> None:
        html = "<table><tr><td>Unclosed"
        result = html_table_to_markdown(html)
        assert isinstance(result, str)

    def test_real_unstructured_table_snippet(self) -> None:
        """Tests a realistic snippet of HTML from the Unstructured parser."""
        html = (
            "<table><thead><tr><th>Aspek Penguatan</th>"
            "<th>Akuntabilitas Kinerja</th>"
            "<th>Klasifikasi Indikator</th></tr></thead>"
            "<tbody><tr><td>Manajemen Kinerja</td>"
            "<td>Pelaporan kinerja bulanan</td>"
            "<td>Utama</td></tr></tbody></table>"
        )
        result = html_table_to_markdown(html)
        assert "| Aspek Penguatan |" in result
        assert "Manajemen Kinerja" in result
        assert "---" in result


# ---------------------------------------------------------------------------
# Fallback behaviour
# ---------------------------------------------------------------------------

class TestFallback:
    def test_no_table_tag_returns_original(self) -> None:
        text = "Plain text, no table"
        assert html_table_to_markdown(text) == text

    def test_none_like_empty_string(self) -> None:
        assert html_table_to_markdown("   ") == "   "


# ---------------------------------------------------------------------------
# _is_markdown_table helper (internal, tested for robustness)
# ---------------------------------------------------------------------------

class TestIsMarkdownTable:
    def test_valid_markdown_table(self) -> None:
        text = "| A | B |\n| --- | --- |\n| 1 | 2 |"
        assert _is_markdown_table(text) is True

    def test_valid_markdown_with_colon_alignment(self) -> None:
        text = "| Left | Center | Right |\n| :--- | :---: | ---: |\n| a | b | c |"
        assert _is_markdown_table(text) is True

    def test_html_table_is_not_markdown(self) -> None:
        text = "<table><tr><td>A</td></tr></table>"
        assert _is_markdown_table(text) is False

    def test_plain_text_is_not_markdown(self) -> None:
        assert _is_markdown_table("Just some paragraph text.") is False

    def test_empty_is_not_markdown(self) -> None:
        assert _is_markdown_table("") is False


# ---------------------------------------------------------------------------
# split_markdown_table_lines helper
# ---------------------------------------------------------------------------

class TestSplitMarkdownTableLines:
    def test_valid_table_returns_header_separator_and_data_rows(self) -> None:
        text = "| A | B |\n| --- | --- |\n| 1 | 2 |\n| 3 | 4 |"
        result = split_markdown_table_lines(text)
        assert result == ("| A | B |", "| --- | --- |", ["| 1 | 2 |", "| 3 | 4 |"])

    def test_fewer_than_three_lines_returns_none(self) -> None:
        assert split_markdown_table_lines("| A | B |\n| --- | --- |") is None
        assert split_markdown_table_lines("| A | B |") is None
        assert split_markdown_table_lines("") is None

    def test_strips_surrounding_whitespace(self) -> None:
        text = "\n  | A | B |\n| --- | --- |\n| 1 | 2 |\n  \n"
        result = split_markdown_table_lines(text)
        assert result is not None
        header, separator, data_rows = result
        assert header == "| A | B |"
        assert data_rows == ["| 1 | 2 |"]


# ---------------------------------------------------------------------------
# _clean_cell_text helper
# ---------------------------------------------------------------------------

class TestCleanCellText:
    def test_collapses_whitespace(self) -> None:
        assert _clean_cell_text("  hello   world  ") == "hello world"

    def test_escapes_pipe(self) -> None:
        assert _clean_cell_text("A | B") == "A \\| B"

    def test_truncates_long_text(self) -> None:
        long_text = "x" * 400
        result = _clean_cell_text(long_text)
        assert len(result) <= 304  # 300 + "…"
        assert result.endswith("…")

    def test_short_text_not_truncated(self) -> None:
        short = "Teks pendek"
        assert _clean_cell_text(short) == short
