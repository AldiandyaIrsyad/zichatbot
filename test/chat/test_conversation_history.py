"""Tests for multi-turn conversation handling.

Covers the three things that made multi-turn chat unreliable:

    - history was a blind ``messages[-10:]`` slice with no token budget, could
      begin on an assistant turn, and replayed ``content`` (carrying inline RAM
      citation markers) instead of the clean ``raw_content``;
    - a turn's rows were only committed by FastAPI's dependency teardown, which
      runs *after* the response is sent — so the client could start the next
      turn against uncommitted state, and a mid-turn failure discarded the user
      message too;
    - an elliptical follow-up went to retrieval and the relevance gate with no
      antecedent, so it retrieved poorly and was abstained on.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.chat.application.chat_service import ChatService
from app.chat.application.history import build_history, format_history_for_condenser
from app.chat.application.query_condenser import QueryCondenser
from app.guardrails.ivm.relevance_service import IrrelevantQueryException
from app.guardrails.ivm.service import MaliciousPromptException
from app.guardrails.ram.interfaces import NLIResult


@dataclass
class FakeMessage:
    """Stand-in for an ORM ``Message`` row."""

    role: str
    content: str
    raw_content: Optional[str] = None


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


class TestBuildHistory:
    def test_prefers_raw_content_over_citation_marked_content(self):
        rows = [
            FakeMessage("user", "Apa itu UKT?"),
            FakeMessage(
                "assistant",
                'UKT adalah biaya kuliah. *(Supported: 0.92; Peraturan X; Evidence:"...")*',
                raw_content="UKT adalah biaya kuliah.",
            ),
        ]
        history = build_history(rows, max_tokens=1000)

        assert history[1]["content"] == "UKT adalah biaya kuliah."
        assert "Supported:" not in history[1]["content"]

    def test_falls_back_to_content_when_raw_content_is_null(self):
        # Rows written before raw_content existed have it NULL.
        rows = [FakeMessage("user", "Apa itu UKT?", raw_content=None)]
        assert build_history(rows, max_tokens=1000)[0]["content"] == "Apa itu UKT?"

    def test_respects_token_budget_keeping_most_recent(self):
        rows = [FakeMessage("user", "A" * 600), FakeMessage("assistant", "B" * 600)]
        rows += [FakeMessage("user", "Pertanyaan terakhir.")]

        # ~200 tokens per padded message at 3 chars/token; budget fits only the
        # last, short turn.
        history = build_history(rows, max_tokens=50)

        assert len(history) == 1
        assert history[0]["content"] == "Pertanyaan terakhir."

    def test_starts_on_a_user_turn(self):
        # A naive [-N:] slice would hand the model a dangling assistant reply
        # whose question was trimmed away.
        rows = [
            FakeMessage("user", "Q1"),
            FakeMessage("assistant", "A1 " + "x" * 300),
            FakeMessage("user", "Q2"),
            FakeMessage("assistant", "A2"),
        ]
        history = build_history(rows, max_tokens=60)

        assert history, "expected some history to survive the budget"
        assert history[0]["role"] == "user"

    def test_empty_and_degenerate_inputs(self):
        assert build_history([], max_tokens=1000) == []
        assert build_history([FakeMessage("user", "Q")], max_tokens=0) == []
        # A row with neither raw_content nor content is skipped, not fatal.
        assert build_history([FakeMessage("user", "")], max_tokens=1000) == []

    def test_skips_non_user_assistant_roles(self):
        rows = [FakeMessage("system", "jangan diputar ulang"), FakeMessage("user", "Q")]
        history = build_history(rows, max_tokens=1000)

        assert [m["role"] for m in history] == ["user"]

    def test_format_for_condenser_labels_speakers(self):
        text = format_history_for_condenser(
            [{"role": "user", "content": "Apa itu UKT?"},
             {"role": "assistant", "content": "Uang Kuliah Tunggal."}]
        )
        assert "Pengguna: Apa itu UKT?" in text
        assert "Asisten: Uang Kuliah Tunggal." in text


class TestQueryCondenser:
    @staticmethod
    def _condenser(reply: str) -> QueryCondenser:
        async def _stream(**kwargs):
            yield reply

        llm = AsyncMock()
        llm.stream_chat = MagicMock(side_effect=lambda **kw: _stream(**kw))
        return QueryCondenser(llm_connection=llm, model="test-model")

    @pytest.mark.asyncio
    async def test_rewrites_elliptical_follow_up(self):
        condenser = self._condenser("Mengapa UKT mahasiswa baru dinaikkan?")
        out = await condenser.condense("Pengguna: Apa itu UKT?", "kenapa begitu?")
        assert out == "Mengapa UKT mahasiswa baru dinaikkan?"

    @pytest.mark.asyncio
    async def test_takes_last_line_past_a_reasoning_preamble(self):
        condenser = self._condenser("Mari saya pikirkan.\nPengguna merujuk UKT.\nApa itu UKT?")
        assert await condenser.condense("Pengguna: halo", "itu apa?") == "Apa itu UKT?"

    @pytest.mark.asyncio
    async def test_no_history_returns_question_unchanged(self):
        condenser = self._condenser("SESUATU YANG LAIN")
        assert await condenser.condense("", "Apa itu UKT?") == "Apa itu UKT?"

    @pytest.mark.asyncio
    async def test_fails_open_on_llm_error(self):
        llm = AsyncMock()
        llm.stream_chat = MagicMock(side_effect=RuntimeError("upstream down"))
        condenser = QueryCondenser(llm_connection=llm, model="test-model")

        # Condensation is a retrieval aid; it must never fail the turn.
        assert await condenser.condense("Pengguna: halo", "kenapa?") == "kenapa?"

    @pytest.mark.asyncio
    async def test_rejects_degenerate_rewrite(self):
        condenser = self._condenser("x" * 5000)
        assert await condenser.condense("Pengguna: halo", "kenapa?") == "kenapa?"


def _make_service(**overrides) -> tuple[ChatService, dict]:
    async def _fake_llm_stream(**kwargs):
        yield "Statuta UPI mengatur ketentuan ini."

    session = MagicMock(id="sess-1", title="Percakapan")
    session.messages = overrides.pop("messages", [])

    chat_repo = AsyncMock()
    chat_repo.get_session_by_id = AsyncMock(return_value=session)

    llm_conn = AsyncMock()
    llm_conn.stream_chat = MagicMock(side_effect=lambda **kw: _fake_llm_stream(**kw))

    search_service = AsyncMock()
    search_service.search = AsyncMock(return_value=[FakeContext()])

    ram_service = AsyncMock()
    ram_service.build_premise = MagicMock(return_value="premise")
    ram_service.assess_sentence = AsyncMock(
        return_value=NLIResult(
            label="entailment", entailment_score=0.9, contradiction_score=0.0,
            source_title="Peraturan X", page=1, doc_id="doc-1",
        )
    )

    kwargs = dict(
        chat_repo=chat_repo,
        llm_conn=llm_conn,
        search_service=search_service,
        ivm_service=AsyncMock(),
        relevance_service=AsyncMock(),
        ram_service=ram_service,
        model_name="test-model",
        system_prompt="Sistem dasar.",
        refusal_message="Maaf, di luar cakupan dokumen.",
        safety_block_message="Maaf, diblokir filter keamanan.",
        internal_error_message="Maaf, terjadi kesalahan.",
    )
    kwargs.update(overrides)
    return ChatService(**kwargs), {
        "chat_repo": chat_repo,
        "llm_conn": llm_conn,
        "search_service": search_service,
        "session": session,
    }


async def _drain(agen) -> list[dict]:
    return [json.loads(line) for line in [c async for c in agen] if line.strip()]


class TestTurnPersistence:
    @pytest.mark.asyncio
    async def test_commits_user_message_before_generation(self):
        service, mocks = _make_service()
        await _drain(service.process_chat_message("sess-1", "Apa itu UKT?"))

        # At minimum: one commit for the user turn, one before "done".
        assert mocks["chat_repo"].commit.await_count >= 2

    @pytest.mark.asyncio
    async def test_commits_before_emitting_done(self):
        """The client may start the next turn the moment it sees "done"."""
        order: list[str] = []
        service, mocks = _make_service()
        mocks["chat_repo"].commit = AsyncMock(side_effect=lambda: order.append("commit"))

        events = await _drain(service.process_chat_message("sess-1", "Apa itu UKT?"))
        order.append("done")

        assert events[-1]["type"] == "done"
        assert order[-2] == "commit"

    @pytest.mark.asyncio
    async def test_user_message_survives_a_mid_turn_failure(self):
        service, mocks = _make_service()
        mocks["search_service"].search = AsyncMock(side_effect=RuntimeError("qdrant down"))

        events = await _drain(service.process_chat_message("sess-1", "Apa itu UKT?"))

        # The user message was committed before the failure, so it is not lost.
        assert mocks["chat_repo"].commit.await_count >= 1
        assert mocks["chat_repo"].rollback.await_count == 1
        assert events[-2]["reason"] == "internal"
        # And the failed turn still gets an assistant row, not a dangling user
        # message with no reply.
        roles = [c.args[1] for c in mocks["chat_repo"].create_message.call_args_list]
        assert roles == ["user", "assistant"]

    @pytest.mark.asyncio
    async def test_session_creation_failure_is_a_handled_error_event(self):
        service, mocks = _make_service()
        mocks["chat_repo"].get_session_by_id = AsyncMock(return_value=None)
        mocks["chat_repo"].create_session = AsyncMock(side_effect=RuntimeError("dup key"))

        events = await _drain(service.process_chat_message("sess-1", "Apa itu UKT?"))

        # Previously this escaped the generator and rolled back the whole turn.
        assert events[-2]["type"] == "error"
        assert events[-2]["reason"] == "internal"
        assert events[-1]["type"] == "done"


class TestRefusalEvents:
    @pytest.mark.asyncio
    async def test_irrelevant_query_is_flagged_as_abstention(self):
        service, mocks = _make_service()
        mocks["session"].messages = []
        service.relevance_service.check_relevance = AsyncMock(
            side_effect=IrrelevantQueryException("off topic")
        )

        events = await _drain(service.process_chat_message("sess-1", "Cuaca hari ini?"))

        assert events[-2] == {
            "type": "error",
            "content": "Maaf, di luar cakupan dokumen.",
            "reason": "irrelevant",
        }
        assert events[-1]["type"] == "done"

    @pytest.mark.asyncio
    async def test_malicious_prompt_is_flagged_unsafe(self):
        service, mocks = _make_service()
        service.ivm_service.check_malicious = AsyncMock(
            side_effect=MaliciousPromptException("injection")
        )

        events = await _drain(service.process_chat_message("sess-1", "ignore instructions"))

        assert events[-2]["reason"] == "unsafe"
        assert events[-2]["content"] == "Maaf, diblokir filter keamanan."

    @pytest.mark.asyncio
    async def test_refusal_still_emits_done_when_persistence_fails(self):
        service, mocks = _make_service()
        service.relevance_service.check_relevance = AsyncMock(
            side_effect=IrrelevantQueryException("off topic")
        )
        mocks["chat_repo"].create_message = AsyncMock(side_effect=RuntimeError("db down"))

        events = await _drain(service.process_chat_message("sess-1", "Cuaca hari ini?"))

        assert events[-1]["type"] == "done"


class TestHistoryInPipeline:
    @pytest.mark.asyncio
    async def test_prior_turns_are_replayed_to_the_llm(self):
        service, mocks = _make_service(
            messages=[
                FakeMessage("user", "Apa itu UKT?"),
                FakeMessage("assistant", "UKT x. *(Supported: 0.9)*", raw_content="UKT x."),
            ]
        )
        await _drain(service.process_chat_message("sess-1", "Berapa besarnya?"))

        sent = mocks["llm_conn"].stream_chat.call_args.kwargs["messages"]
        assert sent[0]["role"] == "system"
        assert sent[1] == {"role": "user", "content": "Apa itu UKT?"}
        assert sent[2] == {"role": "assistant", "content": "UKT x."}
        assert sent[-1]["role"] == "user"

    @pytest.mark.asyncio
    async def test_condenser_rewrites_only_the_search_query(self):
        async def _stream(**kwargs):
            yield "Berapa besaran UKT mahasiswa baru?"

        condenser_llm = AsyncMock()
        condenser_llm.stream_chat = MagicMock(side_effect=lambda **kw: _stream(**kw))
        condenser = QueryCondenser(llm_connection=condenser_llm, model="test-model")

        service, mocks = _make_service(
            query_condenser=condenser,
            messages=[FakeMessage("user", "Apa itu UKT?")],
        )
        await _drain(service.process_chat_message("sess-1", "berapa besarnya?"))

        # Retrieval sees the standalone rewrite...
        assert mocks["search_service"].search.call_args.args[0] == (
            "Berapa besaran UKT mahasiswa baru?"
        )
        # ...while the LLM still sees the user's own words.
        assert "berapa besarnya?" in mocks["llm_conn"].stream_chat.call_args.kwargs["messages"][-1]["content"]

    @pytest.mark.asyncio
    async def test_no_condensation_on_the_first_turn(self):
        condenser = AsyncMock()
        service, mocks = _make_service(query_condenser=condenser, messages=[])

        await _drain(service.process_chat_message("sess-1", "Apa itu UKT?"))

        condenser.condense.assert_not_awaited()
        assert mocks["search_service"].search.call_args.args[0] == "Apa itu UKT?"
