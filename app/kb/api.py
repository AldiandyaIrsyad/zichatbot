"""JSON API endpoints for knowledge base administration."""
import os
from fastapi import APIRouter, BackgroundTasks, Depends, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Any, List, Optional
from starlette.responses import JSONResponse

from app.kb.application.kb_service import KBApplicationService
from app.kb.application.search_service import SearchService
from app.kb.dependency import get_kb_service, get_search_service


class NaiveSearchResultItem(BaseModel):
    """A single naive title-search match returned to the client."""
    id: str
    title: str

router = APIRouter()

class PDFUpdateRequest(BaseModel):
    """Request body for toggling a document's active (retrievable) status."""
    active: bool

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

@router.get("/api/admin/pdfs")
async def get_pdfs(service: KBApplicationService = Depends(get_kb_service)) -> Any:
    """Retrieve all uploaded PDF documents with their metadata."""
    pdfs = await service.list_pdfs()
    return [
        {
            "id": str(p.id),
            "title": p.title,
            "description": p.description,
            "active": p.active,
            "ingestion_status": p.ingestion_status,
            "created_at": p.created_at.isoformat() if p.created_at else None,
        }
        for p in pdfs
    ]

@router.post("/api/admin/pdfs", status_code=202)
async def upload_pdf(
    background_tasks: BackgroundTasks,
    title: str = Form(...), 
    description: str = Form(""), 
    file: UploadFile = Form(...), 
    service: KBApplicationService = Depends(get_kb_service)
) -> Any:
    """Upload a new PDF document and trigger async ingestion."""
    pdf = await service.upload_pdf(title, description, file, background_tasks)
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
    service: KBApplicationService = Depends(get_kb_service),
) -> Any:
    """Upload multiple PDF documents at once and trigger async ingestion.

    Each file is paired with a title (by index). Descriptions are optional
    and matched by index; missing descriptions default to empty string.
    Filenames are used as default titles by the frontend, but the user can
    edit them in a table before submitting.
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

    results, failures = await service.upload_pdfs_batch(
        files=files,
        titles=titles,
        descriptions=desc_list,
        bg_tasks=background_tasks,
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


@router.get("/api/kb/search", response_model=List[SearchResultItem])
async def search_knowledge_base(
    q: str = Query(..., min_length=1, description="Search query"),
    top_k: int = Query(default=15, ge=1, le=100, description="Number of results to return"),
    session_id: Optional[str] = Query(default=None, description="Optional session scope"),
    mode: str = Query(default="hybrid", description="Retrieval mode: hybrid, dense, or sparse"),
    rerank: bool = Query(default=True, description="Apply the cross-encoder reranker (set false for ablations)"),
    search_service: SearchService = Depends(get_search_service),
) -> List[SearchResultItem]:
    """Search the knowledge base for contexts relevant to a query.

    Returns hybrid (dense + sparse, RRF fusion) results by default. Set ``mode``
    to ``dense``/``sparse`` and ``rerank=false`` for ablations comparing the
    fusion strategies themselves rather than reranked variants. Used by the chat
    pipeline and the retrieval evaluation script.
    """
    contexts = await search_service.search(
        query=q, top_k=top_k, session_id=session_id, mode=mode, rerank=rerank
    )
    return [
        SearchResultItem(
            chunk_id=c.chunk_id,
            parent_chunk_id=c.parent_chunk_id,
            doc_id=c.doc_id,
            text=c.text,
            score=c.score,
            source_title=c.source_title,
            page=c.page,
            breadcrumbs=c.breadcrumbs,
        )
        for c in contexts
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
