"""Infra adapter barrel for the Chat bounded context.

Re-exports the concrete adapters implementing the ports declared in
``app/chat/domain/interfaces.py``. See that module's "Ports → adapters map"
for which class fulfills which interface, and ``app/chat/dependency.py``
for where each is constructed and injected.
"""

from .llm_connection import LLMConnection
from .nli_client import NLIClient
from .pdf_text_extractor import (
    PdfCorruptError,
    PdfExtractionError,
    PdfNoTextError,
    PdfTextExtractor,
    PdfTooManyPagesError,
)
from .postgres_chat_repo import PostgresChatRepository
from .prompt_guard_client import PromptGuardClient

__all__ = [
    "LLMConnection",
    "NLIClient",
    "PdfCorruptError",
    "PdfExtractionError",
    "PdfNoTextError",
    "PdfTextExtractor",
    "PdfTooManyPagesError",
    "PostgresChatRepository",
    "PromptGuardClient",
]
