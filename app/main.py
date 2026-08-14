"""FastAPI application entrypoint and composition root.

Configures structured logging, defines the ``lifespan`` (DB schema init +
Qdrant collection on startup, singleton-client cleanup on shutdown), and
mounts the ``app.kb.api``, ``app.chat.api``, and ``app.frontend`` routers.

Importing the domain ``models`` modules registers every ORM table on the
shared ``Base.metadata`` so ``create_all`` in ``lifespan`` sees them all.
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

import structlog
from fastapi import FastAPI
from sqlalchemy import text

from app.shared.config import get_app_settings
from app.shared.logging import setup_logging
from app.shared.middleware import CorrelationIdMiddleware
from app.shared.db import engine, Base

# Import ORM models so their tables register on Base.metadata before
# create_all runs in lifespan().
import app.kb.domain.models  # noqa: F401
import app.chat.domain.models  # noqa: F401

from app.kb.api import router as kb_router
from app.chat.api import router as chat_router

from app.kb.config import get_qdrant_settings
from app.kb.infra.qdrant_store import QdrantStore
from app.kb.dependency import get_vector_store, get_document_parser, get_reranker, get_text_embedder

setup_logging()
logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Startup schema init + shutdown resource cleanup.

    Startup enables the ``ltree`` extension, runs ``create_all``, applies
    guarded ``ALTER TABLE`` migrations for later-added columns (ltree
    hierarchy on ``parent_chunks``, RAG ``context``/``sources`` on
    ``messages``), and ensures the Qdrant collection exists.

    Shutdown closes the ``@lru_cache``-d singleton infra clients from
    ``app.kb.dependency`` that were actually instantiated, so unused ones
    (e.g. the BGE-M3 model) are never loaded just to be closed.
    """
    logger.info("application_startup", status="started")

    # Initialize DB schema
    async with engine.begin() as conn:
        # ltree extension (Postgres only; no-op on SQLite)
        if engine.dialect.name == "postgresql":
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS ltree"))

        await conn.run_sync(Base.metadata.create_all)

        # Guarded ALTER TABLE: create_all won't alter existing tables, so add
        # later columns to parent_chunks explicitly.
        if engine.dialect.name == "postgresql":
            result = await conn.execute(text(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'parent_chunks'"
            ))
            existing_cols = {row[0] for row in result}
            new_cols = {
                "parent_id": "VARCHAR REFERENCES parent_chunks(id) ON DELETE SET NULL",
                "ordinal": "INTEGER DEFAULT 0",
                "path": "VARCHAR DEFAULT ''",
                "depth": "INTEGER DEFAULT 0",
            }
            for col_name, col_def in new_cols.items():
                if col_name not in existing_cols:
                    await conn.execute(text(
                        f"ALTER TABLE parent_chunks ADD COLUMN {col_name} {col_def}"
                    ))
                    logger.info("db.alter_table", table="parent_chunks", column=col_name)

            # Guarded ALTER TABLE for the messages table. JSONB matches how
            # SQLAlchemy's JSON column maps under Postgres, so fresh-DB
            # create_all and this ALTER produce the same schema.
            result = await conn.execute(text(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'messages'"
            ))
            existing_msg_cols = {row[0] for row in result}
            new_msg_cols = {
                "context": "TEXT",
                "sources": "JSONB DEFAULT NULL",
            }
            for col_name, col_def in new_msg_cols.items():
                if col_name not in existing_msg_cols:
                    await conn.execute(text(
                        f"ALTER TABLE messages ADD COLUMN {col_name} {col_def}"
                    ))
                    logger.info("db.alter_table", table="messages", column=col_name)

    # Initialize Qdrant collection (main retrieval)
    kb_config = get_qdrant_settings()
    qdrant_store = QdrantStore(
        host=kb_config.host,
        port=kb_config.port,
        collection_name=kb_config.collection_name,
    )
    await qdrant_store.ensure_collection()
    await qdrant_store.close()

    yield

    # Close the process-lifetime singleton clients from app.kb.dependency
    # (each wraps an httpx.AsyncClient or in-process model that would
    # otherwise leak). Only close getters actually called this run
    # (cache_info().currsize > 0) so an unused one isn't instantiated just to
    # be closed — for get_text_embedder() that would load the whole BGE-M3
    # model on the way out. Per-client try/except so one failure doesn't
    # block the rest.
    for getter in (get_document_parser, get_reranker, get_text_embedder, get_vector_store):
        if getter.cache_info().currsize == 0:
            continue
        closeable = getter()
        if closeable is None:
            continue
        try:
            await closeable.close()
        except Exception as exc:
            logger.warning("shutdown.close_failed", target=type(closeable).__name__, error=str(exc))

    logger.info("application_shutdown", status="stopped")


app_settings = get_app_settings()

fastapi_app = FastAPI(
    title=app_settings.title,
    description="Refactored RAG Chatbot using DDD architecture.",
    lifespan=lifespan,
    version=app_settings.version,
)

# Add Middleware
fastapi_app.add_middleware(CorrelationIdMiddleware)

from app.frontend import router as frontend_router

# Order matters: kb/chat API routers are mounted before the frontend catch-all
# so /api/* paths win over the "/" page route.
fastapi_app.include_router(kb_router)
fastapi_app.include_router(chat_router)
fastapi_app.include_router(frontend_router)
