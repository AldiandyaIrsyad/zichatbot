"""Tests for resolving BGE-M3 to its local snapshot.

Passing the bare Hub repo id makes the Hub client revalidate the repo on every
process start — two API round-trips before any weights load, and a hard failure
when the machine is offline. The weights are already on disk; only the revision
lookup goes out.
"""
from __future__ import annotations

from unittest.mock import patch

from app.kb.infra.bge_m3_embeddings import _resolve_model_source


def _fresh(model: str) -> str:
    """Call through the lru_cache without poisoning it for other tests."""
    _resolve_model_source.cache_clear()
    try:
        return _resolve_model_source(model)
    finally:
        _resolve_model_source.cache_clear()


class TestResolveModelSource:
    def test_local_directory_passes_through(self, tmp_path) -> None:
        assert _fresh(str(tmp_path)) == str(tmp_path)

    def test_cached_repo_resolves_to_snapshot_path(self) -> None:
        with patch(
            "huggingface_hub.snapshot_download", return_value="/cache/snapshots/abc"
        ) as dl:
            assert _fresh("BAAI/bge-m3") == "/cache/snapshots/abc"
        assert dl.call_args.kwargs["local_files_only"] is True

    def test_uncached_repo_falls_back_to_the_hub_id(self) -> None:
        # First run on a fresh machine must still download normally.
        with patch("huggingface_hub.snapshot_download", side_effect=OSError("not cached")):
            assert _fresh("BAAI/bge-m3") == "BAAI/bge-m3"


class TestNliTokenizerCache:
    """The NLI client had the same problem, worse: Tokenizer.from_pretrained
    re-downloads the full ~1.4 MB tokenizer.json on every process start."""

    def test_prefers_the_local_cache_without_revalidating(self) -> None:
        from app.guardrails.nli.infra import sequence_classify_client as mod

        with patch("huggingface_hub.hf_hub_download", return_value="/cache/tokenizer.json") as dl, \
             patch.object(mod.Tokenizer, "from_file", return_value="TOK") as from_file:
            assert mod._load_tokenizer("some/model") == "TOK"

        assert dl.call_args.kwargs["local_files_only"] is True
        from_file.assert_called_once_with("/cache/tokenizer.json")

    def test_downloads_once_when_not_cached(self) -> None:
        from app.guardrails.nli.infra import sequence_classify_client as mod

        calls = []

        def fake_download(repo, filename, **kwargs):
            calls.append(kwargs.get("local_files_only", False))
            if kwargs.get("local_files_only"):
                raise OSError("not cached")
            return "/downloaded/tokenizer.json"

        with patch("huggingface_hub.hf_hub_download", side_effect=fake_download), \
             patch.object(mod.Tokenizer, "from_file", return_value="TOK") as from_file:
            assert mod._load_tokenizer("some/model") == "TOK"

        assert calls == [True, False]  # cache probe first, then a real fetch
        from_file.assert_called_once_with("/downloaded/tokenizer.json")
