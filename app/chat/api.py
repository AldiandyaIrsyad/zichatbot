"""
JSON API endpoints for the Chat module.
"""

from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

from app.chat.application.attachment_service import AttachmentService
from app.chat.application.chat_service import ChatService
from app.chat.config import ChatConfig, get_chat_config
from app.chat.dependency import get_attachment_service, get_chat_service, get_query_expander
from app.chat.infra import PdfCorruptError, PdfNoTextError, PdfTooManyPagesError
from app.kb.api import DocumentSearchResultItem
from app.kb.application.search_service import SearchService
from app.kb.dependency import (
    get_kb_repo,
    get_reranker,
    get_retrieval_strategy,
    get_text_embedder,
    get_vector_store,
)
from app.kb.domain.interfaces import IQueryExpander
from app.guardrails.ivm.service import MaliciousPromptException

router = APIRouter()

class ChatRequest(BaseModel):
    """Request body for ``POST /api/chat/sessions/{session_id}/stream``.

    ``attachment_text``/``attachment_filename`` carry the client-held result of
    a prior ``/api/chat/attachments/extract`` call; the attachment is never
    persisted server-side, so the client resends it each turn it applies to.
    """

    message: str
    attachment_text: Optional[str] = None
    attachment_filename: Optional[str] = None

@router.post("/api/chat/sessions", status_code=201)
async def create_session(service: ChatService = Depends(get_chat_service)) -> Any:
    """Create a new chat session."""
    return await service.create_session()

@router.get("/api/chat/sessions")
async def list_sessions(service: ChatService = Depends(get_chat_service)) -> Any:
    """List all chat sessions."""
    return await service.list_sessions()

@router.get("/api/chat/sessions/{session_id}")
async def get_session(session_id: str, service: ChatService = Depends(get_chat_service)) -> Any:
    """Get chat session details."""
    session = await service.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session

@router.delete("/api/chat/sessions/{session_id}")
async def delete_session(session_id: str, service: ChatService = Depends(get_chat_service)) -> Any:
    """Delete a chat session."""
    success = await service.delete_session(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "success"}

@router.post("/api/chat/sessions/{session_id}/stream")
async def chat_stream(
    session_id: str,
    request: ChatRequest,
    skip_guardrails: bool = Query(default=False, description="Skip IVM + RAM (baseline mode)"),
    skip_ivm: Optional[bool] = Query(default=None, description="Skip only the IVM safety/relevance checks"),
    skip_ram: Optional[bool] = Query(default=None, description="Skip only the RAM per-sentence assessment"),
    skip_nonce: bool = Query(default=False, description="Skip only the anti-injection delimiter"),
    service: ChatService = Depends(get_chat_service),
) -> Any:
    """Stream a chat response from the LLM, passing through IVM and RAM.

    The defenses bypass independently so an experiment can attribute an effect
    to one of them: ``skip_ivm`` (safety + relevance), ``skip_ram``
    (per-sentence assessment), ``skip_nonce`` (the anti-injection delimiter).
    ``skip_guardrails`` is the shorthand for ``skip_ivm`` + ``skip_ram``; an
    explicit value overrides it, and it never implies ``skip_nonce``.
    """
    return StreamingResponse(
        service.process_chat_message(
            session_id,
            request.message,
            skip_guardrails=skip_guardrails,
            attachment_text=request.attachment_text,
            attachment_filename=request.attachment_filename,
            skip_ivm=skip_ivm,
            skip_ram=skip_ram,
            skip_nonce=skip_nonce,
        ),
        media_type="application/x-ndjson"
    )

@router.post("/api/chat/attachments/extract")
async def extract_attachment(
    file: UploadFile,
    service: AttachmentService = Depends(get_attachment_service),
    config: ChatConfig = Depends(get_chat_config),
) -> Any:
    """Extract text from an uploaded PDF for use as chat context.

    Stateless and session-independent — persists nothing. The client holds the
    returned text in memory and includes it in the subsequent ``/stream`` call,
    where the combined text is re-scanned by IVM before generation. The safety
    check here is a fast-feedback UX nicety, not the authoritative gate.
    """
    filename = file.filename or "upload.pdf"
    if not filename.lower().endswith(".pdf") and file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    max_bytes = config.attachment_max_file_size_mb * 1024 * 1024
    file_bytes = await file.read(max_bytes + 1)
    if len(file_bytes) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large (max {config.attachment_max_file_size_mb} MB).",
        )

    try:
        result = await service.process_upload(filename, file_bytes)
    except PdfTooManyPagesError:
        raise HTTPException(
            status_code=422,
            detail=f"This document has too many pages (max {config.attachment_max_pages}).",
        )
    except PdfNoTextError:
        raise HTTPException(
            status_code=422,
            detail="No text could be extracted — scanned/image-only PDFs aren't supported yet.",
        )
    except PdfCorruptError:
        raise HTTPException(
            status_code=422,
            detail="Could not read this file — it may be corrupted.",
        )
    except MaliciousPromptException:
        raise HTTPException(
            status_code=400,
            detail=config.safety_block_message,
        )

    return {
        "filename": result.filename,
        "page_count": result.page_count,
        "char_count": result.char_count,
        "truncated": result.truncated,
        "text": result.text,
    }


@router.get("/api/chat/search", response_model=List[DocumentSearchResultItem])
async def chat_search(
    q: str = Query(..., min_length=1, description="Search query"),
    top_k: int = Query(default=15, ge=1, le=100, description="Number of documents to return"),
    mode: str = Query(default="hybrid", description="Retrieval mode: hybrid, dense, or sparse"),
    rerank: bool = Query(default=True, description="Apply the cross-encoder reranker"),
    hyde: bool = Query(default=True, description="Apply HyDE query expansion before retrieval"),
    embedder=Depends(get_text_embedder),
    vstore=Depends(get_vector_store),
    repo=Depends(get_kb_repo),
    reranker=Depends(get_reranker),
    expander: Optional[IQueryExpander] = Depends(get_query_expander),
    strategy=Depends(get_retrieval_strategy),
) -> List[DocumentSearchResultItem]:
    """Retrieval endpoint with a per-request HyDE toggle (Experiment 2 ablation).

    Identical to ``/api/kb/search`` except it can inject the chat pipeline's
    ``HyDEExpander`` — which ``/api/kb/search`` structurally cannot, since
    ``kb/`` must not import ``chat/infra``. Living in the chat domain makes HyDE
    the single toggled variable on an otherwise-identical path: ``hyde=false``
    reproduces ``/api/kb/search``. HyDE engages only when ``CHAT_HYDE_ENABLED``
    is set; with ``hyde=false`` the expander is dropped and the raw query is
    embedded. The configured retrieval strategy (config default) still applies.
    """
    search_service = SearchService(
        text_embedder=embedder,
        vector_store=vstore,
        kb_repo=repo,
        reranker=reranker,
        query_expander=expander if hyde else None,
        retrieval_strategy=strategy,
    )
    documents = await search_service.search_documents(
        query=q, top_k=top_k, mode=mode, rerank=rerank
    )
    return [
        DocumentSearchResultItem(
            doc_id=d.doc_id,
            title=d.title,
            released_date=d.released_date,
            content=d.content,
            score=d.score,
        )
        for d in documents
    ]
