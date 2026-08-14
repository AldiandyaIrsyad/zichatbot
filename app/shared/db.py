"""Database connection and session management.

Provides the async SQLAlchemy engine and session factory for PostgreSQL,
shared by both bounded contexts (``app/chat/infra/postgres_chat_repo.py`` and
``app/kb/infra/postgres_repo.py``). Both import :class:`Base` as their
declarative base, so all models share one ``metadata`` and one engine.
"""

from typing import AsyncGenerator
import structlog
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import declarative_base, DeclarativeBase

from app.shared.config import get_db_settings

logger = structlog.get_logger(__name__)


class Base(DeclarativeBase):
    """Shared declarative base. Both contexts' ``domain/models.py`` subclass
    this so ``Base.metadata`` collects every table for one-pass ``create_all``.
    """
    pass

from sqlalchemy.ext.asyncio import AsyncEngine


def get_engine() -> AsyncEngine:
    """Create the SQLAlchemy async engine.

    ``pool_pre_ping=True`` runs a ``SELECT 1`` before each checkout so
    server-dropped connections (idle timeout, restart) are replaced instead of
    raising ``OperationalError`` mid-request.
    """
    settings = get_db_settings()
    engine = create_async_engine(
        settings.async_database_url,
        echo=False,
        future=True,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=10,
    )
    return engine


# Process-lifetime singleton engine, created once at import. Closed in
# ``app/main.py``'s lifespan.
engine = get_engine()

# Global session maker. ``expire_on_commit=False`` keeps attributes readable
# after ``commit()`` (the chat pipeline commits mid-request then reads ORM
# objects); ``autoflush=False`` avoids implicit flushes before each query.
async_session_maker = async_sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False, autoflush=False
)


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency yielding an :class:`AsyncSession`. Used by both
    contexts' ``dependency.py`` to inject a session into their Postgres
    repositories. Commits on success, rolls back and re-raises on error,
    always closes.
    """
    async with async_session_maker() as session:
        try:
            yield session
            await session.commit()
        except Exception as exc:
            logger.error("database.session.error", error=str(exc))
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_db() -> None:
    """Create all defined tables. Idempotent (``create_all`` skips existing
    tables). Called during startup in ``app/main.py``'s lifespan.
    """
    logger.info("Initializing database tables...")
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database tables created successfully.")
    except Exception as exc:
        logger.error("database.initialization.failed", error=str(exc))
        raise
