"""Domain ports (Protocol interfaces) for the Chat bounded context.

The application/domain layers import only these Protocols; concrete adapters
live in ``app/chat/infra/`` and are injected at the composition root
(``app/chat/dependency.py``).

Ports → adapters: :class:`ILLMConnection` → ``llm_connection.LLMConnection``;
:class:`IChatRepository` → ``postgres_chat_repo.PostgresChatRepository``;
:class:`IAttachmentExtractor` → ``pdf_text_extractor.PdfTextExtractor``.
"""

from dataclasses import dataclass
from typing import Protocol, AsyncIterator, Callable, List, Optional, Dict, Any
from app.chat.domain.models import Session, Message


# Re-exported so ``from app.chat.domain.interfaces import LLMUsage`` keeps
# working; the definitions live in ``usage.py`` alongside the collector.
from app.chat.domain.usage import LLMUsage, UsageCallback  # noqa: F401


class ILLMConnection(Protocol):
    """Port for an LLM backend. Implemented by
    ``app/chat/infra/llm_connection.py::LLMConnection`` (async OpenAI-compatible
    client) and, for the eval harness,
    ``app/thesis/_eval/_shared/clients.py::EvalLLMClient``.
    """

    def stream_chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float = 0.0,
        *,
        on_usage: Optional[UsageCallback] = None,
        call: str = "generation",
    ) -> AsyncIterator[str]:
        """Stream a chat completion, yielding incremental text fragments.

        ``on_usage`` receives the call's token accounting once the stream ends.
        Optional: adapters that cannot report usage simply never call it.
        """
        ...

    async def generate(
        self,
        model: str,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float = 0.0,
        *,
        on_usage: Optional[UsageCallback] = None,
        call: str = "generation",
    ) -> str:
        """Return a complete (non-streaming) chat completion as one string
        (e.g. for HyDE hypothetical document generation).
        """
        ...

    async def close(self) -> None:
        """Release the underlying HTTP client / connection pool."""
        ...


class IChatRepository(Protocol):
    """Port for chat session/message persistence. Implemented by
    ``app/chat/infra/postgres_chat_repo.py::PostgresChatRepository``.
    """

    async def get_all_sessions(self) -> List[Session]:
        """Return all chat sessions, newest first."""
        ...

    async def create_session(self, session_id: str, title: str) -> Session:
        """Create and persist a new chat session, or return the existing one if
        that ID was already inserted concurrently (idempotent)."""
        ...

    async def get_session_by_id(self, session_id: str, load_messages: bool = False) -> Optional[Session]:
        """Fetch a session by ID, optionally eager-loading its messages."""
        ...

    async def update_session_title(self, session: Session, new_title: str) -> Session:
        """Rename an existing session."""
        ...

    async def create_message(
        self,
        session_id: str,
        role: str,
        content: str,
        raw_content: Optional[str] = None,
        context: Optional[str] = None,
        sources: Optional[List[Dict[str, Any]]] = None,
        attachment_filename: Optional[str] = None,
    ) -> Message:
        """Persist one message. ``content`` is the final rendered text (with
        citations); ``raw_content`` the pre-citation LLM output; ``context``
        the joined RAG text; ``sources`` the per-chunk citation dicts.
        """
        ...

    async def delete_session(self, session_id: str) -> bool:
        """Delete a session and all its messages. Returns True if deleted."""
        ...

    async def commit(self) -> None:
        """Commit the current transaction, making prior writes durable.

        Needed because FastAPI unwinds ``yield`` dependencies — including
        ``get_db_session``'s commit — only *after* the response has been sent
        (``fastapi/routing.py``: the dependency exit stack closes after
        ``await response(...)``). For a streaming chat turn that means the
        client is told the turn is done while the rows are still uncommitted,
        so the next request can read stale state. Callers commit explicitly at
        turn boundaries instead of relying on teardown.
        """
        ...

    async def rollback(self) -> None:
        """Roll back the current transaction, e.g. after a failed flush leaves
        it in an aborted state and later writes must still succeed."""
        ...


@dataclass(frozen=True)
class ExtractedPdfText:
    """Result of PDF text extraction: the plain text plus page/char counts and
    a truncation flag.
    """

    text: str
    page_count: int
    char_count: int
    truncated: bool


class IAttachmentExtractor(Protocol):
    """Port for extracting text from a chat-uploaded document.

    Unlike ``app.kb.domain.interfaces.IDocumentParser`` (permanent KB ingestion
    via Unstructured + VLM, which can take minutes), this must complete
    synchronously within one chat request, so implementations favor fast native
    extraction over OCR/VLM. Implemented by
    ``app/chat/infra/pdf_text_extractor.py::PdfTextExtractor``.
    """

    def extract(self, file_bytes: bytes) -> ExtractedPdfText:
        """Extract text from the given PDF bytes.

        Raises:
            PdfCorruptError: The PDF is unreadable.
            PdfNoTextError: No extractable text (e.g. scanned).
            PdfTooManyPagesError: Exceeds the page limit.
        """
        ...
