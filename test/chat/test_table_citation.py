"""Tests for table-aware citation handling in ChatService's streaming pipeline.

Verifies that:
- ``_is_table_row`` correctly identifies Markdown table lines.
- ``_handle_complete_proposition`` never injects a citation marker onto a
  table row's line (which would corrupt GFM table syntax), instead
  accumulating rows and assessing the whole block once, emitting the
  citation as its own paragraph.
- Prose-only sequences are unaffected (no regression for the common case).
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from app.chat.application.chat_service import ChatService
from app.guardrails.ram.interfaces import NLIResult


def _make_chat_service(ram_service: AsyncMock) -> ChatService:
    return ChatService(
        chat_repo=AsyncMock(),
        llm_conn=AsyncMock(),
        search_service=AsyncMock(),
        ivm_service=AsyncMock(),
        relevance_service=AsyncMock(),
        ram_service=ram_service,
        model_name="test-model",
        system_prompt="",
    )


async def _collect(agen) -> list[str]:
    return [chunk async for chunk in agen]


class TestIsTableRow:
    def test_data_row_is_a_table_row(self) -> None:
        assert ChatService._is_table_row("| PT ABC | Menang | 2020 |") is True

    def test_header_and_separator_rows_are_table_rows(self) -> None:
        assert ChatService._is_table_row("| Nama | Nilai |") is True
        assert ChatService._is_table_row("| --- | --- |") is True

    def test_ordinary_prose_is_not_a_table_row(self) -> None:
        assert ChatService._is_table_row("Ini adalah kalimat biasa.") is False

    def test_prose_with_a_mid_sentence_pipe_is_not_a_table_row(self) -> None:
        assert ChatService._is_table_row("Nilai |x| adalah 5.") is False

    def test_empty_string_is_not_a_table_row(self) -> None:
        assert ChatService._is_table_row("") is False

    def test_leading_trailing_whitespace_is_tolerated(self) -> None:
        assert ChatService._is_table_row("   | A | B |   ") is True


class TestHandleCompleteProposition:
    @pytest.mark.asyncio
    async def test_table_row_yielded_verbatim_with_no_assessment_call(self) -> None:
        ram_service = AsyncMock()
        service = _make_chat_service(ram_service)
        table_rows: list[tuple[str, str]] = []

        out = await _collect(service._handle_complete_proposition(
            "| PT ABC | Menang | 2020 |", "\n",
            ram_contexts=[], premise="", skip_ram=False, table_rows=table_rows,
        ))

        assert out == ["| PT ABC | Menang | 2020 |\n"]
        ram_service.assess_sentence.assert_not_called()
        assert table_rows == [("| PT ABC | Menang | 2020 |", "\n")]

    @pytest.mark.asyncio
    async def test_trailing_prose_flushes_accumulated_table_as_one_assessment(self) -> None:
        ram_service = AsyncMock()
        ram_service.assess_sentence = AsyncMock(
            return_value=NLIResult(label="entailment", entailment_score=0.9, contradiction_score=0.0,
                                    source_title="Doc", page=3, doc_id="d1")
        )
        service = _make_chat_service(ram_service)
        table_rows: list[tuple[str, str]] = [
            ("| Nama | Nilai |", "\n"),
            ("| --- | --- |", "\n"),
            ("| Budi | 90 |", "\n"),
        ]

        out = await _collect(service._handle_complete_proposition(
            "Itu adalah rangkumannya.", "\n",
            ram_contexts=[], premise="", skip_ram=False, table_rows=table_rows,
        ))

        # Two assess_sentence calls: one for the flushed table block, one
        # for the trailing prose proposition itself.
        assert ram_service.assess_sentence.await_count == 2
        table_assessed_text = ram_service.assess_sentence.await_args_list[0].args[0]
        assert table_assessed_text == "| Nama | Nilai |\n| --- | --- |\n| Budi | 90 |"

        # Accumulator cleared after flush.
        assert table_rows == []

        # First yielded chunk is the citation paragraph — never glued onto a "|" line.
        assert out[0].startswith("\n")
        assert not out[0].lstrip().startswith("|")
        assert "Supported" in out[0]

        # The prose proposition itself is yielded afterwards (with its own citation).
        assert out[-1].startswith("Itu adalah rangkumannya.")

    @pytest.mark.asyncio
    async def test_prose_only_sequence_matches_pre_existing_inline_behavior(self) -> None:
        ram_service = AsyncMock()
        ram_service.assess_sentence = AsyncMock(
            return_value=NLIResult(label="entailment", entailment_score=0.8, contradiction_score=0.0)
        )
        service = _make_chat_service(ram_service)
        table_rows: list[tuple[str, str]] = []

        out = await _collect(service._handle_complete_proposition(
            "Ini adalah fakta penting", "\n",
            ram_contexts=[], premise="", skip_ram=False, table_rows=table_rows,
        ))

        assert len(out) == 1
        assert out[0].startswith("Ini adalah fakta penting.")
        assert "Supported" in out[0]
        assert table_rows == []

    @pytest.mark.asyncio
    async def test_short_proposition_below_threshold_skips_assessment(self) -> None:
        ram_service = AsyncMock()
        service = _make_chat_service(ram_service)
        table_rows: list[tuple[str, str]] = []

        out = await _collect(service._handle_complete_proposition(
            "Ya", "\n",
            ram_contexts=[], premise="", skip_ram=False, table_rows=table_rows,
        ))

        ram_service.assess_sentence.assert_not_called()
        assert out == ["Ya.\n"]

    @pytest.mark.asyncio
    async def test_skip_ram_yields_no_citation(self) -> None:
        ram_service = AsyncMock()
        service = _make_chat_service(ram_service)
        table_rows: list[tuple[str, str]] = []

        out = await _collect(service._handle_complete_proposition(
            "Ini adalah fakta penting", "\n",
            ram_contexts=[], premise="", skip_ram=True, table_rows=table_rows,
        ))

        ram_service.assess_sentence.assert_not_called()
        assert out == ["Ini adalah fakta penting.\n"]
