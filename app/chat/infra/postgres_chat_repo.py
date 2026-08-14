"""Postgres repository adapter for the chat bounded context.

Persists :class:`Session` and :class:`Message` ORM entities via the shared
async SQLAlchemy session from ``app/shared/db.py``. Fulfills
``app/chat/domain/interfaces.py::IChatRepository``; wired in
``app/chat/dependency.py::get_chat_repo``.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload

from app.chat.domain.models import Message, Session
from app.chat.domain.interfaces import IChatRepository


class PostgresChatRepository(IChatRepository):
    """Database operations for the chat domain."""

    def __init__(self, db: AsyncSession):
        """Wrap a request-scoped async session.

        Write methods flush but do not commit; the caller decides the
        transaction boundary via :meth:`commit`/:meth:`rollback`. Streaming
        callers must commit explicitly — see :meth:`commit`.
        """
        self.db = db

    async def commit(self) -> None:
        """Commit the current transaction (see ``IChatRepository.commit``)."""
        await self.db.commit()

    async def rollback(self) -> None:
        """Roll back the current transaction (see ``IChatRepository.rollback``)."""
        await self.db.rollback()

    async def get_all_sessions(self) -> List[Session]:
        """Return all sessions ordered oldest-first; does not eager-load
        messages (each ``Session.messages`` stays a lazy relationship)."""
        result = await self.db.execute(select(Session).order_by(Session.created_at.asc()))
        return list(result.scalars().all())

    async def create_session(self, session_id: str, title: str) -> Session:
        """Insert a session row, or return the existing one if that ID is
        already present.

        Uses ``ON CONFLICT DO NOTHING`` rather than a plain INSERT because the
        create and the first stream arrive as two requests racing on the same
        client-minted ID: ``POST /api/chat/sessions`` returns the ID to the
        browser before its transaction commits, so the immediately-following
        ``/stream`` can miss it on read and try to insert it again. A plain
        INSERT raised ``UniqueViolationError`` there and took the whole first
        turn down with it.

        Column defaults are declared Python-side on the ORM model, so they are
        passed explicitly here — a Core insert does not run them.
        """
        stmt = (
            pg_insert(Session.__table__)
            .values(id=session_id, title=title, created_at=datetime.now(timezone.utc))
            .on_conflict_do_nothing(index_elements=["id"])
        )
        await self.db.execute(stmt)
        await self.db.flush()

        existing = await self.get_session_by_id(session_id)
        if existing is None:  # pragma: no cover - the row was just inserted
            raise RuntimeError(f"session {session_id} missing right after insert")
        return existing

    async def get_session_by_id(self, session_id: str, load_messages: bool = False) -> Optional[Session]:
        """Fetch a session by ID. When ``load_messages`` is True, eager-loads
        ``Session.messages`` via ``selectinload`` (a separate SELECT) to
        avoid a lazy-load in async context, which would otherwise raise a
        ``greenlet_spawn`` error once the session detaches."""
        query = select(Session).where(Session.id == session_id)
        if load_messages:
            query = query.options(selectinload(Session.messages))
        result = await self.db.execute(query)
        return result.scalars().first()

    async def update_session_title(self, session: Session, new_title: str) -> Session:
        """Rename an already-loaded session in place and flush the change."""
        session.title = new_title
        await self.db.flush()
        return session

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
        """Insert a new message row and flush (see ``IChatRepository`` for
        the meaning of each field)."""
        new_msg = Message(
            session_id=session_id,
            role=role,
            content=content,
            raw_content=raw_content,
            context=context,
            sources=sources,
            attachment_filename=attachment_filename,
        )
        self.db.add(new_msg)
        await self.db.flush()
        return new_msg

    async def delete_session(self, session_id: str) -> bool:
        """Delete a session (and, via the ORM's ``cascade="all,
        delete-orphan"`` on ``Session.messages``, all of its messages).
        Returns False without raising if no such session exists."""
        session = await self.get_session_by_id(session_id)
        if session:
            await self.db.delete(session)
            await self.db.flush()
            return True
        return False
