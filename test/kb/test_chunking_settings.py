import pytest

from app.kb.config import ChunkingSettings, get_chunking_settings
from app.rag.chunking.logic import DEFAULT_PARENT_MAX_CHARS


@pytest.fixture(autouse=True)
def clear_settings_cache():
    """``get_chunking_settings`` is lru_cached — clear it around env patching."""
    get_chunking_settings.cache_clear()
    yield
    get_chunking_settings.cache_clear()


def test_defaults_to_hierarchical(monkeypatch):
    monkeypatch.delenv("CHUNKING_STRATEGY", raising=False)
    settings = ChunkingSettings(_env_file=None)

    assert settings.strategy == "hierarchical"
    assert settings.parent_max_chars == DEFAULT_PARENT_MAX_CHARS


def test_env_var_selects_fixed(monkeypatch):
    monkeypatch.setenv("CHUNKING_STRATEGY", "fixed")

    assert ChunkingSettings(_env_file=None).strategy == "fixed"


def test_unknown_strategy_rejected(monkeypatch):
    monkeypatch.setenv("CHUNKING_STRATEGY", "semantic")

    with pytest.raises(ValueError):
        ChunkingSettings(_env_file=None)


def test_to_chunking_config_round_trips(monkeypatch):
    monkeypatch.setenv("CHUNKING_STRATEGY", "fixed")
    monkeypatch.setenv("CHUNKING_FIXED_CHILD_MAX_TOKENS", "256")
    monkeypatch.setenv("CHUNKING_FIXED_CHILD_OVERLAP_TOKENS", "32")

    config = ChunkingSettings(_env_file=None).to_chunking_config()

    assert config.strategy == "fixed"
    assert config.fixed_child_max_tokens == 256
    assert config.fixed_child_overlap_tokens == 32
