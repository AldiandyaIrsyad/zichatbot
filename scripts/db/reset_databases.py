"""One-off destructive reset of the app's Postgres schema and Qdrant
collection, plus the local bulk-upload resume state — used before a full
re-ingestion run.

This is an app-level reset: it connects to the already-running Postgres and
Qdrant containers and wipes their contents (same as what app/main.py's
lifespan does at every startup, just with a drop first). It does NOT touch
Docker volumes or unrelated services (loki/grafana) — that's the point.

Wipes:
  - Postgres: drops and recreates every table declared on app.shared.db.Base
    (KB: pdf_documents, parent_chunks, child_chunks, ingestion_tasks; Chat:
    sessions, messages).
  - Qdrant: deletes and recreates the configured collection
    (QDRANT_COLLECTION_NAME, default "knowledge_base") via the same
    QdrantStore.ensure_collection() used at app startup, so vector config /
    payload indexes never drift from production.
  - scripts/upload_state.json: reset to "{}" so a subsequent
    bulk_upload_pdfs.py run re-uploads everything instead of skipping docs
    a prior run already marked "completed".

Does NOT delete uploads/knowledge_base/ (orphaned PDF copies on disk) —
harmless disk usage, not a correctness issue, since re-upload writes fresh
doc_id-prefixed paths.

Usage::

    # Preview what would be wiped, do nothing
    .venv/bin/python -m tools.reset_databases --dry-run

    # Actually wipe everything (requires explicit confirmation)
    .venv/bin/python -m tools.reset_databases --yes
"""

from __future__ import annotations

import argparse
from pathlib import Path

import asyncio
import structlog
from sqlalchemy import text

from app.shared.db import Base, engine

# Import all models so Base.metadata knows about every table (same pattern
# as app/main.py's lifespan).
import app.kb.domain.models  # noqa: F401
import app.chat.domain.models  # noqa: F401

from app.shared.config import get_db_settings
from app.kb.config import get_qdrant_settings
from app.kb.infra.qdrant_store import QdrantStore

logger = structlog.get_logger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_STATE_FILE = REPO_ROOT / "scripts" / "upload_state.json"


async def reset_postgres() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        if engine.dialect.name == "postgresql":
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS ltree"))
        await conn.run_sync(Base.metadata.create_all)
    logger.info("reset.postgres.done", tables=sorted(Base.metadata.tables))
    print(f"  Postgres: dropped + recreated {len(Base.metadata.tables)} tables "
          f"({', '.join(sorted(Base.metadata.tables))})")


async def reset_qdrant() -> None:
    qdrant_config = get_qdrant_settings()
    store = QdrantStore(
        host=qdrant_config.host,
        port=qdrant_config.port,
        collection_name=qdrant_config.collection_name,
    )
    try:
        if await store._client.collection_exists(qdrant_config.collection_name):
            await store._client.delete_collection(qdrant_config.collection_name)
            logger.info("reset.qdrant.deleted", collection=qdrant_config.collection_name)
        await store.ensure_collection()
    finally:
        await store.close()
    print(f"  Qdrant: deleted + recreated collection '{qdrant_config.collection_name}'")


def reset_state_file(state_file: Path) -> None:
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text("{}")
    print(f"  State file: reset {state_file} to {{}}")


async def run(args: argparse.Namespace) -> None:
    db = get_db_settings()
    qdrant = get_qdrant_settings()

    print("=== tools.reset_databases ===")
    print(f"  Postgres target: {db.user}@{db.host}:{db.port}/{db.db}")
    print(f"  Qdrant target:   {qdrant.host}:{qdrant.port} collection '{qdrant.collection_name}'")
    print(f"  State file:      {args.state_file}")
    print()

    if not args.yes:
        print("DRY RUN — nothing was wiped. Re-run with --yes to actually reset.")
        return

    if not args.skip_postgres:
        await reset_postgres()
    else:
        print("  Postgres: skipped (--skip-postgres)")

    if not args.skip_qdrant:
        await reset_qdrant()
    else:
        print("  Qdrant: skipped (--skip-qdrant)")

    if not args.skip_state:
        reset_state_file(args.state_file)
    else:
        print("  State file: skipped (--skip-state)")

    print()
    print("Done. Docker containers/volumes were not touched — only their contents.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--yes", action="store_true", help="actually perform the reset")
    parser.add_argument("--dry-run", action="store_true", help="alias for omitting --yes")
    parser.add_argument("--state-file", type=Path, default=DEFAULT_STATE_FILE)
    parser.add_argument("--skip-postgres", action="store_true")
    parser.add_argument("--skip-qdrant", action="store_true")
    parser.add_argument("--skip-state", action="store_true")
    args = parser.parse_args()

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
