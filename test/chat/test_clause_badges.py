"""Tests for per-clause assessment with a single per-sentence badge.

A compound sentence is split into clauses so each can be verified separately by
NLI, but the reader must still see one sentence: no full stop and no badge in
the middle of it. Verifies that ``_handle_claim`` assesses every clause yet
emits one badge, reflecting the weakest clause, at the sentence end.
"""

from __future__ import annotations

from typing import Any, List
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


def _result(label: str, entailment: float, contradiction: float = 0.0) -> NLIResult:
    return NLIResult(
        label=label,
        entailment_score=entailment,
        neutral_score=1.0 - entailment - contradiction,
        contradiction_score=contradiction,
        source_title="Doc",
    )


async def _emit(service: ChatService, units: List[ClaimUnit], skip_ram: bool = False) -> str:
    pending: List[Any] = []
    out = ""
    for unit in units:
        async for chunk in service._handle_claim(
            unit, ram_contexts=[], skip_ram=skip_ram,
            table_rows=[], pending_clauses=pending,
        ):
            out += chunk
    return out


class TestWorstResult:
    def test_contradiction_beats_entailment(self) -> None:
        worst = ChatService._worst_result([
            _result("entailment", 0.95),
            _result("contradiction", 0.05, contradiction=0.88),
        ])
        assert worst.label == "contradiction"

    def test_lowest_entailment_wins_without_a_contradiction(self) -> None:
        worst = ChatService._worst_result([
            _result("entailment", 0.95),
            _result("neutral", 0.30),
        ])
        assert worst.entailment_score == 0.30

    def test_all_none_returns_none(self) -> None:
        assert ChatService._worst_result([None, None]) is None


class TestSentenceBadge:
    @pytest.mark.asyncio
    async def test_mid_sentence_clause_gets_no_period_and_no_badge(self) -> None:
        ram_service = AsyncMock()
        ram_service.assess_claim = AsyncMock(return_value=_result("entailment", 0.9))
        service = _make_chat_service(ram_service)

        out = await _emit(service, [
            ClaimUnit(text="Kelompok I ditetapkan", citation_ids=(1,),
                      separator=" ", is_sentence_end=False),
            ClaimUnit(text="dan Kelompok II menyusul.", citation_ids=(1,),
                      separator="\n", is_sentence_end=True),
        ])

        assert out.startswith("Kelompok I ditetapkan dan Kelompok II menyusul.")
        # One badge, at the end — not one per clause.
        assert out.count("Supported:") == 1
        assert "ditetapkan. *(" not in out
        # Both clauses were still verified independently.
        assert ram_service.assess_claim.await_count == 2

    @pytest.mark.asyncio
    async def test_badge_reflects_the_weakest_clause(self) -> None:
        ram_service = AsyncMock()
        ram_service.assess_claim = AsyncMock(side_effect=[
            _result("entailment", 0.95),
            _result("contradiction", 0.02, contradiction=0.91),
        ])
        service = _make_chat_service(ram_service)

        out = await _emit(service, [
            ClaimUnit(text="Klaim benar", citation_ids=(1,),
                      separator=" ", is_sentence_end=False),
            ClaimUnit(text="tetapi klaim ini salah.", citation_ids=(2,),
                      separator="", is_sentence_end=True),
        ])

        assert "Contradicted:0.91" in out

    @pytest.mark.asyncio
    async def test_uncited_sentence_flagged_on_total_length(self) -> None:
        # The trailing clause alone is under MIN_ASSESSABLE_LENGTH; the
        # sentence as a whole is not, so it still gets flagged.
        service = _make_chat_service(AsyncMock())

        out = await _emit(service, [
            ClaimUnit(text="Ini pernyataan panjang tanpa sitasi",
                      separator=" ", is_sentence_end=False),
            ClaimUnit(text="dan itu.", separator="", is_sentence_end=True),
        ])

        assert out.endswith("dan itu. *(Unverified)*")

    @pytest.mark.asyncio
    async def test_single_clause_sentence_unchanged(self) -> None:
        ram_service = AsyncMock()
        ram_service.assess_claim = AsyncMock(return_value=_result("entailment", 0.9))
        service = _make_chat_service(ram_service)

        out = await _emit(service, [
            ClaimUnit(text="Fakta tunggal.", citation_ids=(1,), separator=" "),
        ])

        assert out.startswith("Fakta tunggal. *(Supported:0.90")
        assert out.endswith(")* ")  # badge, then the unit's own separator

    @pytest.mark.asyncio
    async def test_skip_ram_emits_clauses_without_periods_or_badges(self) -> None:
        service = _make_chat_service(AsyncMock())

        out = await _emit(service, [
            ClaimUnit(text="Klausa satu", separator=" ", is_sentence_end=False),
            ClaimUnit(text="dan klausa dua.", separator="", is_sentence_end=True),
        ], skip_ram=True)

        assert out == "Klausa satu dan klausa dua."
