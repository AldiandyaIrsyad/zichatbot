"""
Domain models for the Knowledge Base.
Contains SQLAlchemy entities for database storage and Pydantic models for data transfer.
"""

import uuid
from datetime import datetime, timezone
from typing import List, Optional

from pydantic import BaseModel
from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Index, Integer, String, Text, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.shared.db import Base

class PDFDocument(Base):
    """Stores metadata and ingestion status for a global knowledge base document."""
    __tablename__ = "pdf_documents"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    title = Column(String)
    description = Column(Text)
    pdf_path = Column(String)
    active = Column(Boolean, default=True)
    ingestion_status = Column(String, default="pending")
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    parent_chunks = relationship(
        "ParentChunk",
        back_populates="document",
        cascade="all, delete-orphan",
    )


class ParentChunk(Base):
    """Full-text parent chunks for the Small-to-Big retrieval strategy.

    Sentence-level child chunks are indexed in Qdrant for precision; when a
    child matches, its parent's full text is fetched here for broader LLM
    context. ``content_type`` (text/table/figure/hybrid) drives citation
    attribution and filtering; ``element_metadata`` preserves source metadata
    (raw table HTML, figure image path, table summary).
    """
    __tablename__ = "parent_chunks"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    doc_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("pdf_documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    text: Mapped[str] = mapped_column(Text, nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    page: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    breadcrumbs: Mapped[list[str]] = mapped_column(JSON, default=list, server_default="[]", nullable=False)
    content_type: Mapped[str] = mapped_column(
        String, default="text", server_default="text", nullable=False,
    )
    element_metadata: Mapped[dict] = mapped_column(
        JSON, default=dict, server_default="{}", nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    # ltree hierarchy fields
    parent_id: Mapped[Optional[str]] = mapped_column(
        String, ForeignKey("parent_chunks.id", ondelete="SET NULL"),
        nullable=True, index=True,
    )
    ordinal: Mapped[int] = mapped_column(Integer, default=0, server_default="0", nullable=False)
    path: Mapped[str] = mapped_column(String, default="", server_default="", nullable=False, index=True)
    depth: Mapped[int] = mapped_column(Integer, default=0, server_default="0", nullable=False)

    document = relationship("PDFDocument", back_populates="parent_chunks")
    children = relationship(
        "ChildChunk",
        back_populates="parent",
        cascade="all, delete-orphan",
        foreign_keys="ChildChunk.parent_chunk_id",
    )


class ChildChunk(Base):
    """Sentence-level child chunks — the unit of vector search in Qdrant.

    When a child matches, its parent's full text is retrieved for LLM context
    (Small-to-Big). Persisting child text in Postgres enables chunk-level
    cross-encoder reranking and sibling/cross-ref lookups without a Qdrant
    round-trip. ``path`` is an ltree-style dot path (parent path + ".c" +
    ordinal); ``content_type`` is inherited from the parent.
    """
    __tablename__ = "child_chunks"
    __table_args__ = (
        Index("ix_child_chunks_path", "path"),
        Index("ix_child_chunks_parent_chunk_id", "parent_chunk_id"),
    )

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    parent_chunk_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("parent_chunks.id", ondelete="CASCADE"),
        nullable=False,
    )
    doc_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("pdf_documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    text: Mapped[str] = mapped_column(Text, nullable=False)
    ordinal: Mapped[int] = mapped_column(Integer, default=0, server_default="0", nullable=False)
    path: Mapped[str] = mapped_column(String, default="", server_default="", nullable=False)
    page: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    content_type: Mapped[str] = mapped_column(
        String, default="text", server_default="text", nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    parent = relationship(
        "ParentChunk",
        back_populates="children",
        foreign_keys=[parent_chunk_id],
    )


class IngestionTask(Base):
    """Tracks the async processing status of a PDF document's ingestion.

    States: pending → processing → completed | failed
    """
    __tablename__ = "ingestion_tasks"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    doc_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("pdf_documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    status: Mapped[str] = mapped_column(String, default="pending", nullable=False)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)


class RetrievedContext(BaseModel):
    """A clean domain object representing a context retrieved from the Knowledge Base."""
    chunk_id: str
    parent_chunk_id: str
    doc_id: str
    text: str
    score: float
    source_title: str = ""
    page: Optional[int] = None
    breadcrumbs: List[str] = []
    content_type: str = "text"
    dense_vector: Optional[List[float]] = None
    # ltree hierarchy fields
    child_text: Optional[str] = None
    path: str = ""
    depth: int = 0
    
