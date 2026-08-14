"""Endpoint tests for POST /api/chat/attachments/extract.

Uses a minimal FastAPI app (just the chat router) with dependency overrides
for AttachmentService/ChatConfig, so these tests don't need a live DB,
Qdrant, or Infinity server.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.chat.api import router
from app.chat.application.attachment_service import AttachmentResult
from app.chat.config import ChatConfig, get_chat_config
from app.chat.dependency import get_attachment_service
from app.chat.infra.pdf_text_extractor import PdfCorruptError, PdfNoTextError, PdfTooManyPagesError
from app.guardrails.ivm.service import MaliciousPromptException


def _make_client(attachment_service: AsyncMock) -> TestClient:
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_attachment_service] = lambda: attachment_service
    app.dependency_overrides[get_chat_config] = lambda: ChatConfig(
        _env_file=None, attachment_max_file_size_mb=1,
    )
    return TestClient(app)


class TestExtractAttachment:
    def test_rejects_non_pdf(self) -> None:
        client = _make_client(AsyncMock())

        res = client.post(
            "/api/chat/attachments/extract",
            files={"file": ("notes.txt", b"hello", "text/plain")},
        )

        assert res.status_code == 400

    def test_rejects_oversized_file(self) -> None:
        client = _make_client(AsyncMock())
        big = b"%PDF-1.4\n" + b"0" * (2 * 1024 * 1024)  # over the 1MB test cap

        res = client.post(
            "/api/chat/attachments/extract",
            files={"file": ("big.pdf", big, "application/pdf")},
        )

        assert res.status_code == 413

    def test_returns_extracted_text_on_success(self) -> None:
        attachment_service = AsyncMock()
        attachment_service.process_upload = AsyncMock(return_value=AttachmentResult(
            filename="aturan.pdf", text="Isi peraturan.", page_count=1, char_count=14, truncated=False,
        ))
        client = _make_client(attachment_service)

        res = client.post(
            "/api/chat/attachments/extract",
            files={"file": ("aturan.pdf", b"%PDF-1.4\nfake", "application/pdf")},
        )

        assert res.status_code == 200
        body = res.json()
        assert body["text"] == "Isi peraturan."
        assert body["filename"] == "aturan.pdf"
        assert body["truncated"] is False

    @pytest.mark.parametrize(
        "exc,expected_status",
        [
            (PdfTooManyPagesError(100, 40), 422),
            (PdfNoTextError("no text"), 422),
            (PdfCorruptError("bad file"), 422),
            (MaliciousPromptException("blocked"), 400),
        ],
    )
    def test_maps_extraction_and_safety_errors(self, exc: Exception, expected_status: int) -> None:
        attachment_service = AsyncMock()
        attachment_service.process_upload = AsyncMock(side_effect=exc)
        client = _make_client(attachment_service)

        res = client.post(
            "/api/chat/attachments/extract",
            files={"file": ("aturan.pdf", b"%PDF-1.4\nfake", "application/pdf")},
        )

        assert res.status_code == expected_status
