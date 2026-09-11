"""PostgreSQL repository for the Knowledge Base.

Fulfills: ``app/kb/domain/interfaces.py::IKBRepository``.
Wired in: ``app/kb/dependency.py::get_kb_repo``.
"""

import structlog
from datetime import datetime, timezone
from typing import List, Optional

from sqlalchemy import delete, func, or_, text, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import joinedload

from app.kb.domain.interfaces import IKBRepository
from app.kb.domain.models import PDFDocument, ParentChunk, IngestionTask, ChildChunk

logger = structlog.get_logger(__name__)


def _escape_ilike(value: str) -> str:
    """Escape ILIKE wildcards (``%``, ``_``, and the escape char ``\\``) so a
    user-supplied search term is matched literally rather than as a pattern."""
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


class PostgresKBRepository(IKBRepository):
    """Database operations for the Knowledge Base domain."""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_all_pdfs(self) -> List[PDFDocument]:
        """Return all KB documents, newest first."""
        result = await self.db.execute(select(PDFDocument).order_by(PDFDocument.created_at.desc()))
        return list(result.scalars().all())

    async def query_pdfs(
        self,
        search: Optional[str] = None,
        active: Optional[bool] = None,
        released_from: Optional[datetime] = None,
        released_to: Optional[datetime] = None,
        offset: int = 0,
        limit: int = 50,
    ) -> tuple[List[PDFDocument], int]:
        """Paginated, filtered document listing.

        ``search`` matches title/description (case-insensitive substring);
        ``active`` filters the retrieval flag; ``released_from``/``released_to``
        bound the release date inclusively. Returns (page_items, total_count).
        """
        conditions = []
        if search:
            pattern = f"%{_escape_ilike(search)}%"
            conditions.append(
                or_(
                    PDFDocument.title.ilike(pattern, escape="\\"),
                    PDFDocument.description.ilike(pattern, escape="\\"),
                )
            )
        if active is not None:
            conditions.append(PDFDocument.active.is_(active))
        if released_from is not None:
            conditions.append(PDFDocument.released_date >= released_from)
        if released_to is not None:
            conditions.append(PDFDocument.released_date < released_to)

        count_stmt = select(func.count()).select_from(PDFDocument)
        items_stmt = select(PDFDocument)
        if conditions:
            count_stmt = count_stmt.where(*conditions)
            items_stmt = items_stmt.where(*conditions)

        total = (await self.db.execute(count_stmt)).scalar() or 0
        result = await self.db.execute(
            items_stmt.order_by(PDFDocument.created_at.desc()).offset(offset).limit(limit)
        )
        return list(result.scalars().all()), int(total)

    async def get_pdf_by_id(self, pdf_id: str) -> Optional[PDFDocument]:
        """Fetch a single document by primary key, or None if not found."""
        query = select(PDFDocument).where(PDFDocument.id == pdf_id)
        result = await self.db.execute(query)
        return result.scalars().first()

    async def create_pdf(
        self,
        title: str,
        description: str,
        pdf_path: str,
        released_date: Optional[datetime] = None,
    ) -> PDFDocument:
        """Insert a new PDFDocument row and flush/refresh it so callers get
        back a fully-populated instance (e.g. server-generated ``id``)."""
        new_pdf = PDFDocument(
            title=title,
            description=description,
            pdf_path=pdf_path,
            released_date=released_date,
        )
        self.db.add(new_pdf)
        await self.db.flush()
        await self.db.refresh(new_pdf)
        logger.info(
            "kb.repository.pdf_created",
            pdf_id=new_pdf.id,
            title=title,
            released_date=released_date.isoformat() if released_date else None,
        )
        return new_pdf

    async def update_pdf_active_status(self, pdf_id: str, active: bool) -> Optional[PDFDocument]:
        """Set a document's ``active`` flag; returns None if it doesn't exist.

        Only touches the Postgres row — mirroring this into Qdrant's
        payload (so inactive docs are excluded from retrieval) is the
        caller's responsibility, see
        ``app/kb/application/kb_service.py::KBApplicationService.update_pdf_status``.
        """
        pdf = await self.get_pdf_by_id(pdf_id)
        if pdf:
            pdf.active = active  # type: ignore
            await self.db.flush()
            await self.db.refresh(pdf)
            logger.info("kb.repository.pdf_status_updated", pdf_id=pdf_id, active=active)
            return pdf
        return None

    async def bulk_update_active_status(self, pdf_ids: List[str], active: bool) -> List[str]:
        """Toggle ``active`` on many documents in a single UPDATE; returns the
        ids that actually existed (mirroring into Qdrant is the caller's job,
        see ``KBApplicationService.bulk_update_status``)."""
        if not pdf_ids:
            return []
        result = await self.db.execute(
            select(PDFDocument.id).where(PDFDocument.id.in_(pdf_ids))
        )
        existing_ids = [row[0] for row in result.all()]
        if not existing_ids:
            return []
        await self.db.execute(
            update(PDFDocument)
            .where(PDFDocument.id.in_(existing_ids))
            .values(active=active)
        )
        await self.db.flush()
        logger.info("kb.repository.pdf_status_bulk_updated", count=len(existing_ids), active=active)
        return existing_ids

    async def delete_pdf(self, pdf_id: str) -> bool:
        """Delete a document row; its chunks cascade via FK on-delete rules.
        Returns True if a row was found and deleted, False otherwise.
        Does not touch Qdrant vectors or the on-disk PDF file — see
        ``KBApplicationService.delete_pdf`` for the multi-system delete."""
        pdf = await self.get_pdf_by_id(pdf_id)
        if pdf:
            await self.db.delete(pdf)
            await self.db.flush()
            logger.info("kb.repository.pdf_deleted", pdf_id=pdf_id)
            return True
        return False

    async def bulk_delete_pdfs(self, pdf_ids: List[str]) -> List[str]:
        """Delete many documents in a single DELETE; returns the ids that
        actually existed. Chunk rows cascade via their ``ondelete="CASCADE"``
        foreign keys (parent_chunks/child_chunks/ingestion_tasks). Qdrant
        vectors and on-disk files are the caller's responsibility."""
        if not pdf_ids:
            return []
        result = await self.db.execute(
            select(PDFDocument.id).where(PDFDocument.id.in_(pdf_ids))
        )
        existing_ids = [row[0] for row in result.all()]
        if not existing_ids:
            return []
        await self.db.execute(
            delete(PDFDocument).where(PDFDocument.id.in_(existing_ids))
        )
        await self.db.flush()
        logger.info("kb.repository.pdf_bulk_deleted", count=len(existing_ids))
        return existing_ids

    async def search_titles_naive(self, query: str) -> List[PDFDocument]:
        """Literal, case-insensitive, word-order-sensitive title substring
        match — deliberately reproduces the title-only search behavior of
        naive JDIH portals (e.g. jdih.upi.edu) for the /demo comparison
        page. Not used by the real RAG retrieval path.
        """
        result = await self.db.execute(
            select(PDFDocument)
            .where(PDFDocument.active.is_(True))
            .where(PDFDocument.title.ilike(f"%{_escape_ilike(query)}%", escape="\\"))
            .order_by(PDFDocument.created_at.desc())
        )
        return list(result.scalars().all())

    async def get_pdfs_by_ids(self, pdf_ids: List[str]) -> List[PDFDocument]:
        """Fetch multiple documents by ID in a single query (order not
        guaranteed).
        """
        if not pdf_ids:
            return []
        result = await self.db.execute(
            select(PDFDocument).where(PDFDocument.id.in_(pdf_ids))
        )
        return list(result.scalars().all())

    async def save_parent_chunks(self, chunks: List[ParentChunk]) -> List[ParentChunk]:
        """Bulk-insert parent chunks for a document."""
        if not chunks:
            return []
        self.db.add_all(chunks)
        await self.db.flush()
        logger.info("kb.repository.chunks_saved", count=len(chunks))
        return chunks

    async def get_parent_chunks_by_ids(self, chunk_ids: List[str]) -> List[ParentChunk]:
        """Fetch parent chunks by ID, eager-loading the parent ``document``
        (joinedload) so callers can read the source PDFDocument without a
        second round-trip, ordered by ``chunk_index`` for stable output."""
        if not chunk_ids:
            return []
        result = await self.db.execute(
            select(ParentChunk)
            .options(joinedload(ParentChunk.document))
            .where(ParentChunk.id.in_(chunk_ids))
            .order_by(ParentChunk.chunk_index)
        )
        return list(result.scalars().all())

    async def save_child_chunks(self, chunks: List[ChildChunk]) -> List[ChildChunk]:
        """Persist child chunk records to the database."""
        if not chunks:
            return []
        self.db.add_all(chunks)
        await self.db.flush()
        logger.info("kb.repository.child_chunks_saved", count=len(chunks))
        return chunks

    async def get_child_chunks_by_ids(self, chunk_ids: List[str]) -> List[ChildChunk]:
        """Fetch child chunks by their primary key IDs."""
        if not chunk_ids:
            return []
        result = await self.db.execute(
            select(ChildChunk).where(ChildChunk.id.in_(chunk_ids))
        )
        return list(result.scalars().all())

    async def get_sibling_chunks(self, parent_id: str) -> List[ParentChunk]:
        """Fetch sibling parent chunks sharing the same parent_id."""
        result = await self.db.execute(
            select(ParentChunk)
            .where(ParentChunk.parent_id == parent_id)
            .order_by(ParentChunk.ordinal)
        )
        return list(result.scalars().all())

    async def get_chunks_by_path_prefix(self, path_prefix: str) -> List[ParentChunk]:
        """Fetch parent chunks whose path contains the given segment as a whole
        label (e.g. "pasal_5"), plus anything nested beneath it.

        Paths are rooted at the document UUID (e.g.
        ``<uuid>.bab_i_ketentuan_umum.pasal_1``), so a left-anchored
        ``LIKE 'pasal_5%'`` never matches, and ``LIKE '%pasal_5%'`` would
        false-positive on ``pasal_50``/``pasal_51``. Instead this casts to
        ``ltree`` in-query (the column stays plain ``VARCHAR``, so no migration)
        to match ``path_prefix`` as a whole label, guaranteeing segment-boundary
        correctness.
        """
        if not path_prefix:
            return []
        result = await self.db.execute(
            select(ParentChunk)
            .where(
                text("path::ltree ~ (:lq1)::lquery OR path::ltree ~ (:lq2)::lquery")
                .bindparams(lq1=f"*.{path_prefix}", lq2=f"*.{path_prefix}.*")
            )
            .order_by(ParentChunk.ordinal)
        )
        return list(result.scalars().all())

    async def create_ingestion_task(self, doc_id: str) -> IngestionTask:
        """Create a new ``pending`` ingestion task row for a document."""
        task = IngestionTask(doc_id=doc_id, status="pending")
        self.db.add(task)
        await self.db.flush()
        await self.db.refresh(task)
        logger.info("kb.repository.task_created", task_id=task.id, doc_id=doc_id)
        return task

    async def update_ingestion_task(
        self, task_id: str, status: str, error_message: Optional[str] = None
    ) -> Optional[IngestionTask]:
        """Update a task's status (and optional error message); sets
        ``completed_at`` when transitioning to a terminal status
        (``completed``/``failed``) and clears it otherwise. Returns None if
        the task doesn't exist."""
        result = await self.db.execute(select(IngestionTask).where(IngestionTask.id == task_id))
        task = result.scalars().first()
        if not task:
            return None

        task.status = status
        if error_message is not None:
            task.error_message = error_message
        if status in ("completed", "failed"):
            task.completed_at = datetime.now(timezone.utc)
        else:
            task.completed_at = None

        await self.db.flush()
        logger.info("kb.repository.task_updated", task_id=task_id, status=status)
        return task

    async def get_ingestion_task_by_doc_id(self, doc_id: str) -> Optional[IngestionTask]:
        """Return the most recently created ingestion task for a document
        (a document may have multiple tasks across re-ingestion attempts)."""
        result = await self.db.execute(
            select(IngestionTask)
            .where(IngestionTask.doc_id == doc_id)
            .order_by(IngestionTask.created_at.desc())
            .limit(1)
        )
        return result.scalars().first()

    async def commit(self) -> None:
        """Commit the current transaction on the underlying session."""
        await self.db.commit()

    async def rollback(self) -> None:
        """Roll back the current transaction on the underlying session
        (used by ``upload_pdfs_batch`` to recover the session after a
        per-file failure so the loop can continue)."""
        await self.db.rollback()
