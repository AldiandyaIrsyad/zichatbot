"""Centralised Pydantic-Settings configuration for the whole application.

Single source of truth for environment-driven settings. Each settings group
is a ``BaseSettings`` subclass with its own env prefix, so one ``.env`` file
configures every bounded context (chat, kb) and the shared infrastructure
(logging, database) without cross-talk.

Groups: :class:`AppSettings` (``APP_*``), :class:`LoggerSettings``
(``LOGGER_*``), :class:`DatabaseSettings`` (``POSTGRES_*``). Each ``get_*``
factory is ``@lru_cache``-d to a process-lifetime singleton; tests override
via ``get_*_settings.cache_clear()``.

Bounded-context settings (``ChatConfig``, ``KBConfig``) live in their own
``chat/config.py`` / ``kb/config.py``. Per the ``thesis/`` purity rule the
research core never imports this module — concrete settings reach it only as
plain values passed in by the ``chat``/``kb`` composition roots.
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class AppSettings(BaseSettings):
    """FastAPI application metadata (env prefix ``APP_``): title, version,
    uvicorn host/port, and dev auto-reload."""

    title: str = "Chat Application with PDF Knowledge Base"
    version: str = "1.0.0"
    host: str = "0.0.0.0"
    port: int = Field(default=8000)
    reload: bool = True

    model_config = SettingsConfigDict(env_file=".env", env_prefix="APP_", extra="ignore")


@lru_cache
def get_app_settings() -> AppSettings:
    """Return the cached :class:`AppSettings` singleton."""
    return AppSettings()


class LoggerSettings(BaseSettings):
    """Logging sink configuration (env prefix ``LOGGER_``): Vector TCP
    host/port that ingests JSON logs, and the root log level."""

    vector_host: str = "localhost"
    vector_port: int = Field(default=9000)
    log_level: str = Field(default="INFO")

    model_config = SettingsConfigDict(env_file=".env", env_prefix="LOGGER_", extra="ignore")


@lru_cache
def get_logger_settings() -> LoggerSettings:
    """Return the cached :class:`LoggerSettings` singleton."""
    return LoggerSettings()


class DatabaseSettings(BaseSettings):
    """PostgreSQL connection settings (env prefix ``POSTGRES_``): user,
    password, db, host, and port."""

    user: str = Field(default="postgres")
    password: str = Field(default="postgres")
    db: str = Field(default="postgres")
    port: int = Field(default=5432)
    host: str = Field(default="localhost")

    model_config = SettingsConfigDict(env_file=".env", env_prefix="POSTGRES_", extra="ignore")

    @property
    def async_database_url(self) -> str:
        """Build the asyncpg SQLAlchemy URL from the individual fields."""
        return f"postgresql+asyncpg://{self.user}:{self.password}@{self.host}:{self.port}/{self.db}"


@lru_cache
def get_db_settings() -> DatabaseSettings:
    """Return the cached :class:`DatabaseSettings` singleton."""
    return DatabaseSettings()