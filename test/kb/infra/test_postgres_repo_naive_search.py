"""Integration test for PostgresKBRepository.search_titles_naive().

This method exists to reproduce the title-only, word-order-sensitive search
behavior of naive JDIH portals (e.g. jdih.upi.edu) for the /demo comparison
page — it is intentionally *not* smart. This test locks in that dumbness:
it must match an exact substring but must NOT match when the query words
are reordered, since that reordering-breaks-search behavior is the whole
point of the demo.

Hits the real dev Postgres instance (already required for local
development; see app/shared/config.py) and is skipped if unreachable.
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

from app.kb.domain.models import PDFDocument
from app.kb.infra.postgres_repo import PostgresKBRepository
from app.shared.config import get_db_settings


async def _make_session() -> AsyncSession | None:
    try:
        settings = get_db_settings()
        engine = create_async_engine(settings.async_database_url)
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        session_maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        return session_maker()
    except Exception:
        return None


@pytest.mark.asyncio
async def test_search_titles_naive_matches_substring_but_not_reordered_words() -> None:
    session = await _make_session()
    if session is None:
        pytest.skip("Postgres not reachable — skipping naive title search integration test")

    doc_id = str(uuid.uuid4())
    title = f"Peraturan Kehilangan Motor di Parkiran {doc_id}"
    async with session:
        repo = PostgresKBRepository(session)
        doc = PDFDocument(id=doc_id, title=title, description="", pdf_path="/tmp/x.pdf", active=True)
        session.add(doc)
        await session.flush()

        try:
            exact_substring = await repo.search_titles_naive(f"Kehilangan Motor di Parkiran {doc_id}")
            assert {d.id for d in exact_substring} == {doc_id}

            reordered = await repo.search_titles_naive(f"Motor Kehilangan di Parkiran {doc_id}")
            assert doc_id not in {d.id for d in reordered}

            no_match = await repo.search_titles_naive(f"aturan motor hilang parkiran {doc_id}")
            assert doc_id not in {d.id for d in no_match}
        finally:
            await session.rollback()


@pytest.mark.asyncio
async def test_search_titles_naive_excludes_inactive_documents() -> None:
    session = await _make_session()
    if session is None:
        pytest.skip("Postgres not reachable — skipping naive title search integration test")

    doc_id = str(uuid.uuid4())
    title = f"Peraturan Nonaktif {doc_id}"
    async with session:
        repo = PostgresKBRepository(session)
        doc = PDFDocument(id=doc_id, title=title, description="", pdf_path="/tmp/x.pdf", active=False)
        session.add(doc)
        await session.flush()

        try:
            results = await repo.search_titles_naive(title)
            assert doc_id not in {d.id for d in results}
        finally:
            await session.rollback()
