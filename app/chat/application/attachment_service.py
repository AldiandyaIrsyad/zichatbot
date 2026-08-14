"""Application service for chat PDF attachments.

Extracts text from an uploaded PDF and runs the same IVM safety check
(``IVMService.check_malicious``) the typed message gets, since attachment text
joins the LLM prompt. This is a fail-fast UX check; the authoritative check
runs again in ``ChatService.process_chat_message`` on the combined text, so a
client can't bypass scanning by tampering between the two requests.
"""
from dataclasses import dataclass

import structlog

from app.chat.domain.interfaces import IAttachmentExtractor
from app.guardrails.ivm.service import IVMService

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class AttachmentResult:
    """Outcome of one attachment upload: the safety-checked, possibly
    truncated PDF text plus page/char counts and a truncation flag."""

    filename: str
    text: str
    page_count: int
    char_count: int
    truncated: bool


class AttachmentService:
    """Application service: validates, extracts, and safety-checks a
    chat-uploaded PDF before it's handed back to the client for inclusion
    in a subsequent ``/stream`` request.
    """

    def __init__(self, extractor: IAttachmentExtractor, ivm_service: IVMService):
        """Wire the extraction port and the safety checker (reused from the
        main pipeline so attachment text gets the same prompt-injection scan).
        """
        self.extractor = extractor
        self.ivm_service = ivm_service

    async def process_upload(self, filename: str, file_bytes: bytes) -> AttachmentResult:
        """Extract PDF text and scan it for malicious content.

        Raises:
            PdfExtractionError: Unparseable, too many pages, or no text.
            MaliciousPromptException: Extracted text fails the IVM check.
        """
        extracted = self.extractor.extract(file_bytes)

        await self.ivm_service.check_malicious(extracted.text)

        return AttachmentResult(
            filename=filename,
            text=extracted.text,
            page_count=extracted.page_count,
            char_count=extracted.char_count,
            truncated=extracted.truncated,
        )
