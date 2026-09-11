"""JSON API endpoints for knowledge base administration."""
import os
from datetime import datetime, timezone, timedelta
from fastapi import APIRouter, BackgroundTasks, Depends, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Any, List, Optional
from starlette.responses import JSONResponse

from app.kb.application.kb_service import KBApplicationService
from app.kb.application.search_service import SearchService
from app.kb.dependency import get_kb_service, get_search_service


def _parse_released_date(value: Optional[str]) -> Optional[datetime]:
    """Parse an ISO date string into a timezone-aware datetime, or None.

    Raises 422 on malformed input so the upload form fails fast rather than
    silently dropping the document's release date.
    """
    if not value or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip())
    except ValueError:
        raise HTTPException(status_code=422, detail=f"Invalid released_date: {value}")
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _parse_released_to(value: Optional[str]) -> Optional[datetime]:
    """Parse an inclusive release-date upper bound into an exclusive datetime.

    A bare date (``YYYY-MM-DD``) means "on or before this day", so one day is
    added and callers compare with an exclusive ``<``.
    """
    parsed = _parse_released_date(value)
    if parsed and value and len(value.strip()) == 10:
        parsed = parsed + timedelta(days=1)
    return parsed


class NaiveSearchResultItem(BaseModel):
    """A single naive title-search match returned to the client."""
    id: str
    title: str

router = APIRouter()

class PDFUpdateRequest(BaseModel):
    """Request body for toggling a document's active (retrievable) status."""
    active: bool


class BulkStatusRequest(BaseModel):
    """Request body for bulk-activating/deactivating documents."""
    ids: List[str]
    active: bool


class BulkDeleteRequest(BaseModel):
    """Request body for bulk-deleting documents."""
    ids: List[str]

class SearchResultItem(BaseModel):
    """A single search result returned to the client."""
    chunk_id: str
    parent_chunk_id: str
    doc_id: str
    text: str
    score: float
    source_title: str
    page: Optional[int] = None
    breadcrumbs: List[str] = []


class DocumentSearchResultItem(BaseModel):
    """A document-level search result — one object per unique document.

    Replaces the chunk-level ``SearchResultItem`` on the retrieval endpoints.
    ``content`` is the concatenated retrieved sections for that document.
    """

    doc_id: str
    title: str
    released_date: Optional[datetime] = None
    content: str
    score: float = 0.0

def _pdf_to_dict(p: Any) -> dict:
    """Serialize a PDFDocument row for admin responses."""
    return {
        "id": str(p.id),
        "title": p.title,
        "description": p.description,
        "active": p.active,
        "ingestion_status": p.ingestion_status,
        "created_at": p.created_at.isoformat() if p.created_at else None,
        "released_date": p.released_date.isoformat() if p.released_date else None,
    }


@router.get("/api/admin/pdfs")
async def get_pdfs(service: KBApplicationService = Depends(get_kb_service)) -> Any:
    """Retrieve all uploaded PDF documents with their metadata."""
    pdfs = await service.list_pdfs()
    return [_pdf_to_dict(p) for p in pdfs]


@router.get("/api/admin/pdfs/search")
async def search_pdfs(
    search: Optional[str] = Query(default=None, description="Substring match on title/description"),
    active: Optional[bool] = Query(default=None, description="Filter by active status"),
    released_from: Optional[str] = Query(default=None, description="Release date lower bound (ISO)"),
    released_to: Optional[str] = Query(default=None, description="Release date upper bound (ISO)"),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=200),
    service: KBApplicationService = Depends(get_kb_service),
) -> Any:
    """Paginated, filterable document listing for the admin dashboard."""
    items, total = await service.search_pdfs(
        search=search,
        active=active,
        released_from=_parse_released_date(released_from),
        released_to=_parse_released_to(released_to),
        page=page,
        page_size=page_size,
    )
    return {
        "items": [_pdf_to_dict(p) for p in items],
        "total": total,
        "page": page,
        "page_size": page_size,
        "pages": (total + page_size - 1) // page_size if total else 0,
    }


@router.post("/api/admin/pdfs/bulk/status")
async def bulk_update_pdf_status(
    req: BulkStatusRequest,
    service: KBApplicationService = Depends(get_kb_service),
) -> Any:
    """Activate/deactivate many documents at once."""
    return await service.bulk_update_status(req.ids, req.active)


@router.post("/api/admin/pdfs/bulk/delete")
async def bulk_delete_pdfs(
    req: BulkDeleteRequest,
    service: KBApplicationService = Depends(get_kb_service),
) -> Any:
    """Delete many documents at once."""
    return await service.bulk_delete(req.ids)

@router.post("/api/admin/pdfs", status_code=202)
async def upload_pdf(
    background_tasks: BackgroundTasks,
    title: str = Form(...), 
    description: str = Form(""), 
    file: UploadFile = Form(...), 
    released_date: str = Form(default=None),
    service: KBApplicationService = Depends(get_kb_service)
) -> Any:
    """Upload a new PDF document and trigger async ingestion."""
    pdf = await service.upload_pdf(
        title,
        description,
        file,
        background_tasks,
        _parse_released_date(released_date),
    )
    return JSONResponse(
        status_code=202,
        content={
            "id": pdf.id,
            "title": pdf.title,
            "ingestion_status": pdf.ingestion_status,
            "status": "accepted",
        },
    )

@router.post("/api/admin/pdfs/batch", status_code=202)
async def upload_pdfs_batch(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = Form(...),
    titles: List[str] = Form(...),
    descriptions: List[str] = Form(default=[]),
    released_dates: List[str] = Form(default=[]),
    service: KBApplicationService = Depends(get_kb_service),
) -> Any:
    """Upload multiple PDF documents at once and trigger async ingestion.

    Each file is paired with a title (by index). Descriptions and released
    dates are optional and matched by index; missing entries default to empty
    string / None. Filenames are used as default titles by the frontend, but
    the user can edit them in a table before submitting.
    """
    if len(files) != len(titles):
        raise HTTPException(
            status_code=422,
            detail=f"Number of files ({len(files)}) must match number of titles ({len(titles)})",
        )
    # Pad descriptions to match files length
    desc_list = list(descriptions)
    while len(desc_list) < len(files):
        desc_list.append("")

    parsed_dates = [_parse_released_date(d) for d in released_dates]

    results, failures = await service.upload_pdfs_batch(
        files=files,
        titles=titles,
        descriptions=desc_list,
        bg_tasks=background_tasks,
        released_dates=parsed_dates,
    )
    return JSONResponse(
        status_code=202,
        content={
            "status": "accepted" if not failures else "partial",
            "count": len(results),
            "documents": [
                {
                    "id": pdf.id,
                    "title": pdf.title,
                    "ingestion_status": pdf.ingestion_status,
                }
                for pdf in results
            ],
            "failed_count": len(failures),
            "failures": failures,
        },
    )

@router.put("/api/admin/pdfs/{pdf_id}/status")
async def update_pdf_status(
    pdf_id: str, 
    req: PDFUpdateRequest, 
    service: KBApplicationService = Depends(get_kb_service)
) -> Any:
    """Update the active status of a PDF document."""
    pdf = await service.update_pdf_status(pdf_id, req.active)
    if not pdf:
        raise HTTPException(status_code=404, detail="PDF not found")
    return {"id": pdf.id, "active": pdf.active, "status": "success"}

@router.delete("/api/admin/pdfs/{pdf_id}")
async def delete_pdf(pdf_id: str, service: KBApplicationService = Depends(get_kb_service)) -> Any:
    """Delete a PDF document from the knowledge base."""
    success = await service.delete_pdf(pdf_id)
    if not success:
        raise HTTPException(status_code=404, detail="PDF not found")
    return {"status": "success", "message": "PDF deleted"}

@router.get("/api/admin/pdfs/{pdf_id}/ingestion-status")
async def get_ingestion_status(
    pdf_id: str,
    service: KBApplicationService = Depends(get_kb_service),
) -> Any:
    """Check the ingestion processing status of a PDF document."""
    result = await service.get_ingestion_status(pdf_id)
    if not result:
        raise HTTPException(status_code=404, detail="PDF not found")
    return result


@router.post("/api/admin/pdfs/{pdf_id}/reingest", status_code=202)
async def reingest_pdf(
    pdf_id: str,
    background_tasks: BackgroundTasks,
    service: KBApplicationService = Depends(get_kb_service),
) -> Any:
    """Re-trigger ingestion for a PDF document (e.g. after a crash left it stuck)."""
    pdf = await service.kb_repo.get_pdf_by_id(pdf_id)
    if not pdf:
        raise HTTPException(status_code=404, detail="PDF not found")
    background_tasks.add_task(service.ingest_worker.ingest_document, doc_id=pdf_id)
    return {"id": pdf_id, "status": "reingest_triggered"}


@router.get("/api/kb/pdfs/{pdf_id}/download")
async def download_pdf(pdf_id: str, service: KBApplicationService = Depends(get_kb_service)) -> Any:
    """Serve the original PDF file for a knowledge base document."""
    pdf = await service.kb_repo.get_pdf_by_id(pdf_id)
    if not pdf or not os.path.exists(str(pdf.pdf_path)):
        raise HTTPException(status_code=404, detail="PDF not found")
    return FileResponse(str(pdf.pdf_path), filename=f"{pdf.title}.pdf", media_type="application/pdf")


@router.get("/api/kb/search", response_model=List[DocumentSearchResultItem])
async def search_knowledge_base(
    q: str = Query(..., min_length=1, description="Search query"),
    top_k: int = Query(default=15, ge=1, le=100, description="Number of documents to return"),
    session_id: Optional[str] = Query(default=None, description="Optional session scope"),
    mode: str = Query(default="hybrid", description="Retrieval mode: hybrid, dense, or sparse"),
    rerank: bool = Query(default=True, description="Apply the cross-encoder reranker (set false for ablations)"),
    search_service: SearchService = Depends(get_search_service),
) -> List[DocumentSearchResultItem]:
    """Search the knowledge base, returning one object per unique document.

    Returns hybrid (dense + sparse, RRF fusion) results by default. Set ``mode``
    to ``dense``/``sparse`` and ``rerank=false`` for ablations comparing the
    fusion strategies themselves rather than reranked variants. Each result
    carries the document's title, release date, and the concatenated retrieved
    sections. Used by the chat pipeline and the retrieval evaluation script.
    """
    documents = await search_service.search_documents(
        query=q, top_k=top_k, session_id=session_id, mode=mode, rerank=rerank
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


@router.get("/api/kb/naive-search", response_model=List[NaiveSearchResultItem])
async def naive_search_knowledge_base(
    q: str = Query(..., min_length=1, description="Search query"),
    service: KBApplicationService = Depends(get_kb_service),
) -> List[NaiveSearchResultItem]:
    """Literal, word-order-sensitive title substring search.

    Reproduces the behavior of naive title-only JDIH portals for the
    /demo comparison page — not part of the real RAG retrieval path.
    """
    pdfs = await service.naive_title_search(q)
    return [NaiveSearchResultItem(id=str(p.id), title=str(p.title)) for p in pdfs]
