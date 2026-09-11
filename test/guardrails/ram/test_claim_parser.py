"""Tests for the deterministic Markdown claim parser."""

from __future__ import annotations

from app.guardrails.ram.claim_parser import (
    extract_citations,
    parse_table_block,
    split_cells,
    split_claims,
)
from app.guardrails.ram.interfaces import ClaimUnit


class TestExtractCitations:
    def test_single_marker(self) -> None:
        clean, ids = extract_citations("Statuta ditetapkan.[CIT:1]")
        assert clean == "Statuta ditetapkan."
        assert ids == (1,)

    def test_multiple_markers(self) -> None:
        clean, ids = extract_citations("Klaim.[CIT:1,2]")
        assert clean == "Klaim."
        assert ids == (1, 2)

    def test_no_marker(self) -> None:
        clean, ids = extract_citations("Teks biasa.")
        assert clean == "Teks biasa."
        assert ids == ()

    def test_deduplicates_ids(self) -> None:
        _, ids = extract_citations("[CIT:1,1,2]")
        assert ids == (1, 2)


class TestSplitCells:
    def test_splits_row(self) -> None:
        assert split_cells("| A | B |") == ["A", "B"]

    def test_splits_row_without_outer_pipes(self) -> None:
        assert split_cells("A | B") == ["A", "B"]


class TestSplitClaims:
    def test_complete_sentence_with_citation(self) -> None:
        units, remainder = split_claims("Fakta satu.[CIT:1]")
        assert remainder == ""
        assert len(units) == 1
        unit = units[0]
        assert unit.kind == "prose"
        assert unit.text == "Fakta satu."
        assert unit.citation_ids == (1,)

    def test_incomplete_sentence_is_remainder(self) -> None:
        units, remainder = split_claims("Fakta yang belum seles")
        assert units == []
        assert remainder == "Fakta yang belum seles"

    def test_two_complete_sentences(self) -> None:
        units, remainder = split_claims("A.[CIT:1] B.[CIT:2]")
        assert remainder == ""
        assert [u.text for u in units] == ["A.", "B."]
        assert [u.citation_ids for u in units] == [(1,), (2,)]

    def test_list_item(self) -> None:
        units, remainder = split_claims("- Syarat: KTP.[CIT:1]\n")
        assert remainder == ""
        assert len(units) == 1
        assert units[0].kind == "list_item"
        assert units[0].text == "- Syarat: KTP."
        assert units[0].citation_ids == (1,)

    def test_table_row_is_verbatim(self) -> None:
        units, remainder = split_claims("| A | B |\n")
        assert remainder == ""
        assert units[0].kind == "table_row"
        # markers retained for the table parser
        assert "[CIT" not in units[0].text

    def test_incomplete_sentence_remainder_keeps_trailing_space(self) -> None:
        # Regression: a stripped remainder welds the next stream chunk onto the
        # previous word ("Peraturan" + "Menteri" -> "PeraturanMenteri").
        _, remainder = split_claims("Ini adalah Peraturan ")
        assert remainder == "Ini adalah Peraturan "

    def test_citation_terminated_claim_gets_a_separator(self) -> None:
        # The marker ends the claim without trailing whitespace, so the parser
        # has to synthesise the gap before the next sentence.
        units, remainder = split_claims("Pasal 5 mengatur X.[CIT:1]")
        assert remainder == ""
        assert units[-1].separator == " "


def _stream(text: str, chunks: list[str]) -> str:
    """Replay ``chunks`` through the streaming loop in ChatService and
    reassemble what a client would receive (badges aside)."""
    buffer = ""
    out = ""
    for chunk in chunks:
        buffer += chunk
        units, buffer = split_claims(buffer)
        for unit in units:
            out += unit.text + unit.separator
    return out + buffer


class TestStreamingFidelity:
    """Whitespace must survive arbitrary chunk boundaries.

    Whole-answer tests structurally cannot catch this: the bug only appears
    when a chunk ends mid-sentence, which is exactly what a token stream does.
    """

    def test_word_by_word_stream_preserves_every_space(self) -> None:
        text = (
            "Golongan UKT (Uang Kuliah Tunggal) di Universitas Pendidikan "
            "Indonesia dibagi menjadi 8 kelompok."
        )
        # Trailing-space deltas, the shape that triggers the bug.
        chunks = [w + " " for w in text.split(" ")[:-1]] + [text.split(" ")[-1]]
        assert _stream(text, chunks) == text

    def test_character_by_character_stream_preserves_every_space(self) -> None:
        text = "Kelompok I: penghasilan orang tua. Kelompok II: penghasilan wali."
        assert _stream(text, list(text)) == text

    def test_stream_split_after_citation_marker_keeps_the_gap(self) -> None:
        units_text = _stream(
            "", ["Fakta satu.[CIT:1]", " Fakta dua.[CIT:2]"]
        )
        assert units_text == "Fakta satu. Fakta dua. "

    def test_whitespace_only_chunk_is_not_dropped(self) -> None:
        assert _stream("", [" ", "Kata"]) == " Kata"


class TestClauseUnits:
    def test_only_the_last_clause_ends_the_sentence(self) -> None:
        class _Splitter:
            @staticmethod
            def split_clauses(sentence: str) -> list[str]:
                return ["Pasal 1 mengatur ayat (1)", "dan Pasal 2 mengatur ayat (2)"]

        units, _ = split_claims(
            "Pasal 1 mengatur ayat (1) dan Pasal 2 mengatur ayat (2).[CIT:1]",
            _Splitter(),
        )
        assert [u.is_sentence_end for u in units] == [False, True]
        # Clause text is passed through untouched — no token-join respacing.
        assert units[0].text == "Pasal 1 mengatur ayat (1)"

    def test_single_clause_sentence_ends_the_sentence(self) -> None:
        units, _ = split_claims("Fakta tunggal.[CIT:1]")
        assert [u.is_sentence_end for u in units] == [True]


class TestParseTableBlock:
    def test_parses_header_separator_and_cell_claims(self) -> None:
        table_text = (
            "| Nama | Nilai |\n"
            "| --- | --- |\n"
            "| A | 90 [CIT:1] |\n"
            "| B | 80 |\n"
        )
        table = parse_table_block(table_text)
        assert table is not None
        assert table.header == "| Nama | Nilai |"
        assert table.separator == "| --- | --- |"
        assert len(table.claims) == 1
        claim = table.claims[0]
        assert (claim.row_index, claim.col_index) == (0, 1)
        assert claim.cell_text == "90"
        assert claim.citation_ids == (1,)

    def test_non_table_returns_none(self) -> None:
        assert parse_table_block("Bukan tabel.") is None


class TestListItemCitationInheritance:
    """A bulleted list carries its citation on the sentence introducing it,
    never on each bullet. Without inheritance every bullet was badged
    "Unverified" — a claim-without-a-source label printed directly beneath
    its source. Observed at 8 false badges in one answer."""

    LEAD_IN = "Kelompok tarif dibagi menjadi 3 kelompok, yaitu [CIT:4]:\n"

    def _units(self, text: str):
        units, _ = split_claims(text)
        return units

    def test_bullets_inherit_lead_in_citation(self) -> None:
        text = self.LEAD_IN + "- Kelompok I: Rp500.000\n- Kelompok II: Rp1.000.000\n"
        bullets = [u for u in self._units(text) if u.kind == "list_item"]
        assert len(bullets) == 2
        assert all(u.citation_ids == (4,) for u in bullets)

    def test_bullet_keeps_its_own_citation(self) -> None:
        text = self.LEAD_IN + "- Kelompok I: Rp500.000 [CIT:7]\n"
        bullets = [u for u in self._units(text) if u.kind == "list_item"]
        assert bullets[0].citation_ids == (7,)

    def test_uncited_lead_in_leaves_bullets_uncited(self) -> None:
        """Inheritance must not invent a citation that was never given."""
        text = "Kelompok tarif dibagi menjadi 2 kelompok, yaitu:\n- Kelompok I\n"
        bullets = [u for u in self._units(text) if u.kind == "list_item"]
        assert bullets[0].citation_ids == ()

    def test_intervening_prose_ends_the_lists_scope(self) -> None:
        text = (
            self.LEAD_IN
            + "- Kelompok I: Rp500.000\n"
            + "Penjelasan lain yang tidak berhubungan dengan daftar tersebut.\n"
            + "- Butir tanpa sumber\n"
        )
        bullets = [u for u in self._units(text) if u.kind == "list_item"]
        assert bullets[0].citation_ids == (4,)
        assert bullets[-1].citation_ids == ()

    def test_list_without_any_lead_in_stays_uncited(self) -> None:
        bullets = [u for u in self._units("- Butir pertama\n") if u.kind == "list_item"]
        assert bullets[0].citation_ids == ()
