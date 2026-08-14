"""Domain models for the Chat module (SQLAlchemy entities).

These are the concrete row shapes persisted by
``app/chat/infra/postgres_chat_repo.py::PostgresChatRepository`` and
referenced by :mod:`app.chat.domain.interfaces` type hints, so
``Session``/``Message`` may be used as plain data shapes despite the ORM
coupling.
"""

import uuid
from datetime import datetime, timezone
from typing import List, Optional

from sqlalchemy import JSON, Column, DateTime, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.shared.db import Base


class Session(Base):
    """A conversation thread shown in the sidebar. ``messages`` load on demand
    (``get_session_by_id``'s ``load_messages`` flag) since list views only need
    ``id``/``title``.
    """
    __tablename__ = "sessions"

    # Client-generated UUID so the frontend can mint an id before the first
    # message round-trip.
    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    # Sidebar title; starts "New Chat", rewritten from the first user message.
    title: Mapped[str] = mapped_column(String, default="New Chat")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    # One-to-many, oldest-first; delete-orphan removes messages with the session.
    messages = relationship(
        "Message",
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="Message.created_at",
    )


class Message(Base):
    """Stores a single message within a chat session."""
    __tablename__ = "messages"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id: Mapped[str] = mapped_column(String, ForeignKey("sessions.id"))
    # "user", "assistant", or "system".
    role: Mapped[str] = mapped_column(String)
    # Text before post-processing: the LLM's raw output before citation markers
    # (identical to `content` for user messages). Kept so formatting changes to
    # `content` don't lose the source text.
    raw_content: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    # Final rendered text shown in the UI; assistant messages include the inline
    # "*(STATUS: SCORE; ...)*" citation markers.
    content: Mapped[str] = mapped_column(Text)
    # Joined RAG context that grounded this assistant message, so the "view RAG
    # context" panel survives a refresh. Null for user messages.
    context: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    # Per-chunk source dicts (title/page/breadcrumbs/text) paired with
    # `context` — the per-source structure the UI renders as citations.
    sources: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    # Filename of a chat-uploaded PDF on this (user) message. Only the filename
    # is stored; the extracted text is single-turn and never persisted.
    attachment_filename: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    session = relationship("Session", back_populates="messages")
