"""
Application service for KB administration workflows.
"""

import os
import re
import shutil
import uuid
import structlog
from typing import List, Optional, Any
from fastapi import BackgroundTasks, UploadFile

from typing import Optional

from app.kb.domain.interfaces import IKBRepository, IVectorStore
from app.kb.domain.models import PDFDocument
from app.kb.application.ingest_worker import IngestWorker

logger = structlog.get_logger(__name__)

# Cap on the on-disk filename stem (before the UUID prefix/extension). Long
# filenames break two ways: the filesystem raises ENAMETOOLONG near 255 bytes,
# and Unstructured Cloud's job API (which receives this same filename, since
# the client reads it back off disk) silently returns no output once the name
# is long. The original name/title are preserved in PDFDocument.title/
# description; only the on-disk/upstream-visible name is shortened.
_MAX_FILENAME_STEM_LEN = 60


def _safe_upload_filename(original_filename: str) -> str:
    """Build a short, unique, filesystem- and Unstructured-Cloud-safe filename
    for on-disk storage. Always UUID-prefixed for uniqueness.
    """
    stem, ext = os.path.splitext(original_filename or "upload.pdf")
    ext = ext or ".pdf"
    safe_stem = re.sub(r"[^A-Za-z0-9._-]", "_", stem)[:_MAX_FILENAME_STEM_LEN]
    return f"{uuid.uuid4().hex[:12]}_{safe_stem}{ext}"


class KBApplicationService:
    """Service handling administration of PDF documents and triggering ingestion."""

    def __init__(
        self,
        kb_repo: IKBRepository,
        vector_store: IVectorStore,
        ingest_worker: IngestWorker,
        upload_dir: str = "./uploads/knowledge_base",
    ):
        """Wire the service's collaborators and ensure the upload directory exists.

        Args:
            kb_repo: Postgres-backed :class:`IKBRepository` for document/chunk
                persistence.
            vector_store: :class:`IVectorStore` (Qdrant) for chunk vectors.
            ingest_worker: Orchestrates the parse→embed→upsert pipeline;
                triggered as a FastAPI background task, not awaited here.
            upload_dir: Directory where uploaded PDFs are saved on disk.
        """
        self.kb_repo = kb_repo
        self.vector_store = vector_store
        self.ingest_worker = ingest_worker
        self.upload_dir = upload_dir

        if not os.path.exists(self.upload_dir):
            os.makedirs(self.upload_dir, exist_ok=True)

    async def list_pdfs(self) -> List[PDFDocument]:
        """Return all KB documents (passthrough to ``IKBRepository``)."""
        return await self.kb_repo.get_all_pdfs()

    async def naive_title_search(self, query: str) -> List[PDFDocument]:
        """Passthrough to ``IKBRepository.search_titles_naive`` — see there
        for why this literal ILIKE search exists alongside the real
        hybrid-search pipeline."""
        return await self.kb_repo.search_titles_naive(query)

    async def upload_pdf(self, title: str, description: str, file: UploadFile, bg_tasks: BackgroundTasks) -> PDFDocument:
        """Save one uploaded PDF to disk, record it, and schedule ingestion.

        Touches the filesystem (safe-named file under ``upload_dir``) and
        ``IKBRepository`` (the ``PDFDocument`` row). Embedding and Qdrant
        upsert happen later, asynchronously, in ``ingest_worker.ingest_document``.
        """
        file_path = os.path.join(self.upload_dir, _safe_upload_filename(file.filename))

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        pdf_doc = await self.kb_repo.create_pdf(title=title, description=description, pdf_path=file_path)
        
        # Trigger ingestion in the background
        bg_tasks.add_task(self.ingest_worker.ingest_document, doc_id=str(pdf_doc.id))
        
        return pdf_doc

    async def upload_pdfs_batch(
        self,
        files: List[UploadFile],
        titles: List[str],
        descriptions: List[str],
        bg_tasks: BackgroundTasks,
    ) -> tuple[List[PDFDocument], List[dict[str, str]]]:
        """Upload multiple PDFs and trigger background ingestion for each.

        Saved filenames are UUID-prefixed and length-capped (see
        ``_safe_upload_filename``). Each file is isolated: on success its
        ``PDFDocument`` row is committed immediately, so a later file's failure
        can't roll back already-succeeded files. On failure the session is
        rolled back (clearing its errored state) and the loop continues.

        Returns:
            (successfully created PDFDocuments, [{"filename", "error"}, ...]).
        """
        results: List[PDFDocument] = []
        failures: List[dict[str, str]] = []

        for file, title, description in zip(files, titles, descriptions):
            safe_filename = file.filename or "upload.pdf"
            try:
                file_path = os.path.join(self.upload_dir, _safe_upload_filename(file.filename))

                with open(file_path, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)

                pdf_doc = await self.kb_repo.create_pdf(
                    title=title,
                    description=description,
                    pdf_path=file_path,
                )
                bg_tasks.add_task(self.ingest_worker.ingest_document, doc_id=str(pdf_doc.id))
                await self.kb_repo.commit()
                results.append(pdf_doc)
            except Exception as exc:
                logger.error("kb.upload.file_failed", filename=safe_filename, error=str(exc))
                await self.kb_repo.rollback()
                failures.append({"filename": safe_filename, "error": str(exc)})

        logger.info("kb.upload.batch_complete", succeeded=len(results), failed=len(failures))
        return results, failures

    async def update_pdf_status(self, pdf_id: str, active: bool) -> Optional[PDFDocument]:
        """Toggle a document's active flag in both Postgres and Qdrant.

        Inactive documents are excluded from retrieval (``hybrid_search``
        filters on the ``is_active`` payload), so the flag must be mirrored into
        the vector store, not just the Postgres row.
        """
        doc = await self.kb_repo.update_pdf_active_status(pdf_id, active)
        if doc:
            await self.vector_store.update_payload(pdf_id, {"is_active": active})
        return doc

    async def delete_pdf(self, pdf_id: str) -> bool:
        """Delete a document across all three systems: the Postgres row (and
        cascaded chunks), its Qdrant vectors, and the PDF file on disk. Returns
        False with no side effects if it doesn't exist.
        """
        doc = await self.kb_repo.get_pdf_by_id(pdf_id)
        if not doc:
            return False

        # Delete from postgres
        await self.kb_repo.delete_pdf(pdf_id)

        # Delete from qdrant
        await self.vector_store.delete_by_doc_id(pdf_id)

        # Remove file
        if os.path.exists(str(doc.pdf_path)):
            os.remove(str(doc.pdf_path))

        return True

    async def get_ingestion_status(self, pdf_id: str) -> Optional[dict[str, Any]]:
        """Return the latest ingestion task's status for a document, or
        None if no ingestion task has been recorded for it."""
        task = await self.kb_repo.get_ingestion_task_by_doc_id(pdf_id)
        if not task:
            return None
        return {
            "status": task.status,
            "error_message": task.error_message,
            "created_at": task.created_at,
            "completed_at": task.completed_at,
        }
