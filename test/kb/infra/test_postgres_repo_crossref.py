"""Integration test for PostgresKBRepository.get_chunks_by_path_prefix().

This method relies on Postgres-specific `ltree`/`lquery` operators that
can't be meaningfully verified with a mocked session — a mock would only
prove *some* text() clause was executed, not that the SQL actually matches
the right rows. So this test hits the real dev Postgres instance (already
required for local development; see app/shared/config.py) and is skipped
if it's unreachable (e.g. in an environment without docker-compose up).

Regression coverage: get_chunks_by_path_prefix("pasal_5") must match
".pasal_5" as a whole path label (and its descendants), and must NOT
match "pasal_50"/"pasal_51" as substrings — the bug the naive
`LIKE '%pasal_5%'` fix would have reintroduced.
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

from app.kb.domain.models import PDFDocument, ParentChunk
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
async def test_get_chunks_by_path_prefix_matches_whole_label_only() -> None:
    session = await _make_session()
    if session is None:
        pytest.skip("Postgres not reachable — skipping ltree integration test")

    doc_id = str(uuid.uuid4())
    async with session:
        repo = PostgresKBRepository(session)
        doc = PDFDocument(id=doc_id, title="crossref-test", description="", pdf_path="/tmp/x.pdf")
        session.add(doc)
        await session.flush()

        specs = [
            ("p_exact", f"{doc_id}.bab_ii.pasal_5"),
            ("p_child_50", f"{doc_id}.bab_ii.pasal_50"),
            ("p_child_51", f"{doc_id}.bab_ii.pasal_51"),
            ("p_descendant", f"{doc_id}.bab_ii.pasal_5.ayat_2"),
            ("p_unrelated", f"{doc_id}.bab_i.pasal_1"),
        ]
        for i, (pid, path) in enumerate(specs):
            session.add(ParentChunk(id=pid, doc_id=doc_id, text="x", path=path, chunk_index=i))
        await session.flush()

        try:
            # get_chunks_by_path_prefix() is a global lookup (not scoped to
            # one document), so a real dev/prod DB may have unrelated
            # existing "pasal_5" chunks — filter to just this test's rows.
            results = await repo.get_chunks_by_path_prefix("pasal_5")
            matched_ids = {r.id for r in results if r.doc_id == doc_id}
            assert matched_ids == {"p_exact", "p_descendant"}
        finally:
            await session.rollback()
