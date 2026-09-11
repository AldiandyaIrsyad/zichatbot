"""Tests for table-aware citation handling in ChatService's streaming pipeline.

Verifies that:
- table-row claim units are buffered (not emitted) until the block ends;
- a trailing non-row unit flushes the accumulated table with in-cell badges;
- markers inside cells are stripped and replaced by assessment badges;
- skip_ram strips markers without running any NLI call.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from app.chat.application.chat_service import ChatService
from app.guardrails.ram.interfaces import ClaimUnit, NLIResult


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


class TestHandleClaim:
    @pytest.mark.asyncio
    async def test_table_row_buffered_without_output(self) -> None:
        ram_service = AsyncMock()
        service = _make_chat_service(ram_service)
        table_rows: list[tuple[str, str]] = []

        out = await _collect(service._handle_claim(
            ClaimUnit(text="| PT ABC | Menang | 2020 |", kind="table_row", separator="\n"),
            ram_contexts=[], skip_ram=False, table_rows=table_rows,
        ))

        assert out == []
        ram_service.assess_claim.assert_not_awaited()
        assert table_rows == [("| PT ABC | Menang | 2020 |", "\n")]

    @pytest.mark.asyncio
    async def test_trailing_prose_flushes_table_with_cell_badge(self) -> None:
        ram_service = AsyncMock()
        ram_service.assess_claim = AsyncMock(
            return_value=NLIResult(
                label="entailment", entailment_score=0.9, neutral_score=0.05,
                contradiction_score=0.05, source_title="Doc", page=3, doc_id="d1",
            )
        )
        service = _make_chat_service(ram_service)
        table_rows: list[tuple[str, str]] = [
            ("| Nama | Nilai |", "\n"),
            ("| --- | --- |", "\n"),
            ("| Budi | 90 [CIT:1] |", "\n"),
        ]

        out = await _collect(service._handle_claim(
            ClaimUnit(text="Itu adalah rangkumannya.", kind="prose", citation_ids=(), separator="\n"),
            ram_contexts=[], skip_ram=False, table_rows=table_rows,
        ))

        # Only the cited cell is assessed; the uncited trailing prose is
        # flagged unverified instead of calling NLI.
        assert ram_service.assess_claim.await_count == 1
        assert table_rows == []

        table_chunk = out[0]
        assert "| Nama | Nilai |" in table_chunk
        assert "| Budi | 90" in table_chunk
        assert "[CIT" not in table_chunk
        assert "Supported:0.90" in table_chunk

        assert out[1].startswith("Itu adalah rangkumannya.")
        assert "Unverified" in out[1]

    @pytest.mark.asyncio
    async def test_prose_cited_claim_gets_badge(self) -> None:
        ram_service = AsyncMock()
        ram_service.assess_claim = AsyncMock(
            return_value=NLIResult(
                label="entailment", entailment_score=0.8, neutral_score=0.1,
                contradiction_score=0.1,
            )
        )
        service = _make_chat_service(ram_service)
        table_rows: list[tuple[str, str]] = []

        out = await _collect(service._handle_claim(
            ClaimUnit(text="Ini adalah fakta penting.", kind="prose", citation_ids=(1,), separator="\n"),
            ram_contexts=[], skip_ram=False, table_rows=table_rows,
        ))

        ram_service.assess_claim.assert_awaited_once()
        assert len(out) == 1
        assert out[0].startswith("Ini adalah fakta penting.")
        assert "Supported:0.80" in out[0]
        assert table_rows == []

    @pytest.mark.asyncio
    async def test_skip_ram_strips_markers_without_assessment(self) -> None:
        ram_service = AsyncMock()
        service = _make_chat_service(ram_service)
        table_rows: list[tuple[str, str]] = [
            ("| Nama | Nilai |", "\n"),
            ("| --- | --- |", "\n"),
            ("| Budi | 90 [CIT:1] |", "\n"),
        ]

        out = await _collect(service._handle_claim(
            ClaimUnit(text="Selesai.", kind="prose", citation_ids=(), separator="\n"),
            ram_contexts=[], skip_ram=True, table_rows=table_rows,
        ))

        ram_service.assess_claim.assert_not_awaited()
        table_chunk = out[0]
        assert "[CIT" not in table_chunk
        assert "Unverified" not in table_chunk
        assert "Unverified" not in out[1]

