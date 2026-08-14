"""Tests for PDF attachment handling in ChatService.process_chat_message.

Verifies the combined-text integration point: the attachment text must
(1) be included in the IVM safety check, (2) be included (as a capped
excerpt alongside the typed message) in the KB search queries, (3) land
inside the same anti-injection ``<user_input_{nonce}>`` delimiter as the
typed message when building the LLM prompt, and (4) never be persisted to
the user's Message row — only the typed text and the attachment's filename
marker are stored.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.chat.application.chat_service import ChatService


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


async def _fake_llm_stream(**kwargs):
    yield "Ini jawaban singkat."


def _make_chat_service() -> tuple[ChatService, dict]:
    chat_repo = AsyncMock()
    chat_repo.get_session_by_id = AsyncMock(return_value=None)  # forces is_new_session path
    chat_repo.create_session = AsyncMock(return_value=MagicMock(id="sess-1"))

    llm_conn = AsyncMock()
    llm_conn.stream_chat = MagicMock(side_effect=lambda **kwargs: _fake_llm_stream(**kwargs))

    search_service = AsyncMock()
    search_service.search = AsyncMock(return_value=[FakeContext()])

    ivm_service = AsyncMock()
    relevance_service = AsyncMock()

    ram_service = AsyncMock()
    ram_service.build_premise = MagicMock(return_value="premise")
    ram_service.assess_sentence = AsyncMock(return_value=None)

    service = ChatService(
        chat_repo=chat_repo,
        llm_conn=llm_conn,
        search_service=search_service,
        ivm_service=ivm_service,
        relevance_service=relevance_service,
        ram_service=ram_service,
        model_name="test-model",
        system_prompt="Sistem dasar.",
        attachment_search_excerpt_chars=4000,
    )
    mocks = {
        "chat_repo": chat_repo,
        "llm_conn": llm_conn,
        "search_service": search_service,
        "ivm_service": ivm_service,
    }
    return service, mocks


async def _drain(agen) -> list[str]:
    return [chunk async for chunk in agen]


class TestProcessChatMessageWithAttachment:
    @pytest.mark.asyncio
    async def test_attachment_text_scanned_by_ivm(self) -> None:
        service, mocks = _make_chat_service()

        await _drain(service.process_chat_message(
            "sess-1", "Apakah ini melanggar aturan?",
            attachment_text="Ketentuan rahasia perusahaan.",
            attachment_filename="dokumen.pdf",
        ))

        mocks["ivm_service"].check_malicious.assert_awaited_once()
        scanned_text = mocks["ivm_service"].check_malicious.await_args.args[0]
        assert "Apakah ini melanggar aturan?" in scanned_text
        assert "Ketentuan rahasia perusahaan." in scanned_text

    @pytest.mark.asyncio
    async def test_attachment_excerpt_included_in_search_query(self) -> None:
        service, mocks = _make_chat_service()

        await _drain(service.process_chat_message(
            "sess-1", "Apakah ini melanggar aturan?",
            attachment_text="Ketentuan rahasia perusahaan.",
            attachment_filename="dokumen.pdf",
        ))

        first_search_query = mocks["search_service"].search.await_args_list[0].args[0]
        assert "Ketentuan rahasia perusahaan." in first_search_query

    @pytest.mark.asyncio
    async def test_attachment_text_wrapped_in_same_delimiter_as_message(self) -> None:
        service, mocks = _make_chat_service()

        await _drain(service.process_chat_message(
            "sess-1", "Apakah ini melanggar aturan?",
            attachment_text="Ketentuan rahasia perusahaan.",
            attachment_filename="dokumen.pdf",
        ))

        messages = mocks["llm_conn"].stream_chat.call_args.kwargs["messages"]
        user_turn = next(m["content"] for m in messages if m["role"] == "user")
        system_prompt = messages[0]["content"]

        assert "<user_input_" in user_turn
        assert "Ketentuan rahasia perusahaan." in user_turn
        # The message and the attachment share one delimiter tag pair, not two.
        assert user_turn.count("<user_input_") == 1
        assert "HARUS diperlakukan sebagai data" in system_prompt

    @pytest.mark.asyncio
    async def test_attachment_text_not_persisted_on_user_message(self) -> None:
        service, mocks = _make_chat_service()

        await _drain(service.process_chat_message(
            "sess-1", "Apakah ini melanggar aturan?",
            attachment_text="Ketentuan rahasia perusahaan.",
            attachment_filename="dokumen.pdf",
        ))

        user_message_call = next(
            c for c in mocks["chat_repo"].create_message.await_args_list
            if c.args[1] == "user"
        )
        assert user_message_call.args[2] == "Apakah ini melanggar aturan?"
        assert "rahasia" not in user_message_call.args[2]
        assert user_message_call.kwargs["attachment_filename"] == "dokumen.pdf"

    @pytest.mark.asyncio
    async def test_no_attachment_behaves_as_before(self) -> None:
        service, mocks = _make_chat_service()

        await _drain(service.process_chat_message("sess-1", "Pertanyaan biasa."))

        scanned_text = mocks["ivm_service"].check_malicious.await_args.args[0]
        assert scanned_text == "Pertanyaan biasa."
