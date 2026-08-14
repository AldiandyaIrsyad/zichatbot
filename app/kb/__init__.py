"""
Knowledge Base (KB) Domain.

Responsible for document ingestion, storage, and search operations.
Strictly isolated from conversation state or LLM logic.
"""

from .api import router
from .application.search_service import SearchService
from .domain.models import RetrievedContext

__all__ = [
    "router",
    "SearchService",
    "RetrievedContext"
]
