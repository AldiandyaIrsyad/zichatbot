"""Tests for the independent guardrail ablation switches.

``process_chat_message`` exposes three defenses that can be disabled
separately — ``skip_ivm`` (safety + relevance), ``skip_ram`` (per-sentence
assessment), and ``skip_nonce`` (the anti-injection delimiter). They exist so
an experiment can attribute an effect to one defense instead of to
"guardrails on vs off" as a single block.

What these tests pin:
    - each switch disables its own defense and nothing else;
    - ``skip_guardrails`` still means IVM + RAM together (the pre-existing
      Experiment 4 baseline must keep measuring what it measured before), and
      does NOT disable the nonce;
    - an explicit switch overrides the shorthand.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.chat.application.chat_service import ChatService
from app.guardrails.ram.interfaces import NLIResult


@dataclass
class FakeContext:
    text: str = "Isi peraturan terkait."
    score: float = 0.9
    source_title: str = "Peraturan X"
    page: Optional[int] = 1
    breadcrumbs: List[str] = field(default_factory=list)
    content_type: str = "text"
    chunk_id: str = "chunk-1"
    path: str = "1"
    doc_id: str = "doc-1"
    child_text: str = ""
    parent_chunk_id: str = ""


@dataclass
class FakeDocument:
    content: str = "Isi peraturan terkait."
    title: str = "Peraturan X"
    doc_id: str = "doc-1"
    released_date: Optional[str] = None


async def _fake_llm_stream(**kwargs):
    yield "Statuta UPI mengatur ketentuan ini secara rinci.[CIT:1]"


def _make_chat_service() -> tuple[ChatService, dict]:
    chat_repo = AsyncMock()
    chat_repo.get_session_by_id = AsyncMock(return_value=None)
    chat_repo.create_session = AsyncMock(return_value=MagicMock(id="sess-1"))

    llm_conn = AsyncMock()
    llm_conn.stream_chat = MagicMock(side_effect=lambda **kwargs: _fake_llm_stream(**kwargs))

    search_service = AsyncMock()
    search_service.search = AsyncMock(return_value=[FakeContext()])
    search_service.aggregate_documents = AsyncMock(return_value=[FakeDocument()])

    ivm_service = AsyncMock()
    relevance_service = AsyncMock()

    ram_service = AsyncMock()
    ram_service.assess_claim = AsyncMock(
        return_value=NLIResult(
            label="entailment", entailment_score=0.9, contradiction_score=0.0,
            neutral_score=0.0,
            source_title="Peraturan X", page=1, doc_id="doc-1",
        )
    )

    service = ChatService(
        chat_repo=chat_repo,
        llm_conn=llm_conn,
        search_service=search_service,
        ivm_service=ivm_service,
        relevance_service=relevance_service,
        ram_service=ram_service,
        model_name="test-model",
        system_prompt="Sistem dasar.",
    )
    return service, {
        "ivm_service": ivm_service,
        "relevance_service": relevance_service,
        "ram_service": ram_service,
        "llm_conn": llm_conn,
    }


async def _drain(agen) -> list[str]:
    return [chunk async for chunk in agen]


def _system_prompt_sent(mocks: dict) -> str:
    return mocks["llm_conn"].stream_chat.call_args.kwargs["messages"][0]["content"]


def _user_turn_sent(mocks: dict) -> str:
    return mocks["llm_conn"].stream_chat.call_args.kwargs["messages"][-1]["content"]


class TestDefaults:
    @pytest.mark.asyncio
    async def test_all_defenses_active_by_default(self) -> None:
        service, mocks = _make_chat_service()

        await _drain(service.process_chat_message("sess-1", "Apa isi Statuta UPI?"))

        mocks["ivm_service"].check_malicious.assert_awaited_once()
        mocks["relevance_service"].check_relevance.assert_awaited_once()
        mocks["ram_service"].assess_claim.assert_awaited()
        assert "<user_input_" in _user_turn_sent(mocks)


class TestIndividualSwitches:
    @pytest.mark.asyncio
    async def test_skip_ivm_leaves_ram_and_nonce_alone(self) -> None:
        service, mocks = _make_chat_service()

        await _drain(service.process_chat_message("sess-1", "Apa isi Statuta UPI?", skip_ivm=True))

        mocks["ivm_service"].check_malicious.assert_not_awaited()
        mocks["relevance_service"].check_relevance.assert_not_awaited()
        mocks["ram_service"].assess_claim.assert_awaited()
        assert "<user_input_" in _user_turn_sent(mocks)

    @pytest.mark.asyncio
    async def test_skip_ram_leaves_ivm_and_nonce_alone(self) -> None:
        service, mocks = _make_chat_service()

        await _drain(service.process_chat_message("sess-1", "Apa isi Statuta UPI?", skip_ram=True))

        mocks["ivm_service"].check_malicious.assert_awaited_once()
        mocks["relevance_service"].check_relevance.assert_awaited_once()
        mocks["ram_service"].assess_claim.assert_not_awaited()
        assert "<user_input_" in _user_turn_sent(mocks)

    @pytest.mark.asyncio
    async def test_skip_nonce_leaves_ivm_and_ram_alone(self) -> None:
        service, mocks = _make_chat_service()

        await _drain(service.process_chat_message("sess-1", "Apa isi Statuta UPI?", skip_nonce=True))

        mocks["ivm_service"].check_malicious.assert_awaited_once()
        mocks["ram_service"].assess_claim.assert_awaited()

        user_turn = _user_turn_sent(mocks)
        assert "<user_input_" not in user_turn
        assert "Apa isi Statuta UPI?" in user_turn
        # The system prompt's delimiter-defense paragraph must go too — it
        # names tags that no longer exist.
        assert "user_input_" not in _system_prompt_sent(mocks)


class TestSkipGuardrailsShorthand:
    @pytest.mark.asyncio
    async def test_shorthand_disables_ivm_and_ram_together(self) -> None:
        service, mocks = _make_chat_service()

        await _drain(
            service.process_chat_message("sess-1", "Apa isi Statuta UPI?", skip_guardrails=True)
        )

        mocks["ivm_service"].check_malicious.assert_not_awaited()
        mocks["relevance_service"].check_relevance.assert_not_awaited()
        mocks["ram_service"].assess_claim.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_shorthand_does_not_disable_the_nonce(self) -> None:
        """The delimiter is a separate layer from the IVM/RAM pair.

        Folding it into the shorthand would silently change what the existing
        Experiment 4 baseline arm measures.
        """
        service, mocks = _make_chat_service()

        await _drain(
            service.process_chat_message("sess-1", "Apa isi Statuta UPI?", skip_guardrails=True)
        )

        assert "<user_input_" in _user_turn_sent(mocks)

    @pytest.mark.asyncio
    async def test_explicit_switch_overrides_the_shorthand(self) -> None:
        service, mocks = _make_chat_service()

        await _drain(
            service.process_chat_message(
                "sess-1", "Apa isi Statuta UPI?", skip_guardrails=True, skip_ram=False
            )
        )

        mocks["ivm_service"].check_malicious.assert_not_awaited()
        mocks["ram_service"].assess_claim.assert_awaited()
