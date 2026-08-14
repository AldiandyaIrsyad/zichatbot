"""Tests for the purpose-built prompt-guard server's label resolution.

An inverted label mapping is the worst failure this service can have: the guard
passes attacks and blocks legitimate queries while every health check and
training metric still looks correct. The upstream Llama-Prompt-Guard-2-86M
checkpoint declares no ``id2label`` at all, so the mapping has to come from
somewhere — and which somewhere it came from must be visible.

Verifies:
    - a checkpoint that names its labels wins over the fallback
    - a checkpoint that names nothing falls back to the documented order
    - transformers' synthesised LABEL_0/LABEL_1 counts as "named nothing"
    - the provenance string distinguishes the two, so startup logs are honest
    - a fine-tune that inverts the order is respected, not silently corrected
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

SERVICE = Path(__file__).resolve().parents[2] / "services" / "prompt_guard" / "main.py"


def _load_module():
    """Import the service module by path (it lives outside the app package)."""
    spec = importlib.util.spec_from_file_location("prompt_guard_service", SERVICE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pytest.importorskip("torch", reason="prompt-guard service requires torch")
guard = _load_module()


def _config(id2label):
    """Build a stand-in model config carrying the given mapping."""
    return SimpleNamespace(id2label=id2label)


class TestResolveLabels:
    """Where the index → label mapping comes from, and whether it is honest."""

    def test_named_labels_are_used(self) -> None:
        mapping, source = guard.resolve_labels(
            _config({0: "BENIGN", 1: "MALICIOUS"})
        )
        assert mapping == {0: "BENIGN", 1: "MALICIOUS"}
        assert source == "checkpoint config"

    def test_string_keys_are_normalised_to_int(self) -> None:
        # HF configs round-trip through JSON, so the keys arrive as strings.
        mapping, _ = guard.resolve_labels(_config({"0": "BENIGN", "1": "MALICIOUS"}))
        assert mapping == {0: "BENIGN", 1: "MALICIOUS"}

    def test_missing_mapping_falls_back(self) -> None:
        # This is the real upstream case: the base checkpoint declares nothing.
        mapping, source = guard.resolve_labels(_config(None))
        assert mapping == {0: "BENIGN", 1: "MALICIOUS"}
        assert source.startswith("fallback")

    def test_empty_mapping_falls_back(self) -> None:
        mapping, source = guard.resolve_labels(_config({}))
        assert mapping == {0: "BENIGN", 1: "MALICIOUS"}
        assert source.startswith("fallback")

    def test_synthesised_generic_labels_count_as_unnamed(self) -> None:
        # transformers invents LABEL_0/LABEL_1 when a checkpoint names nothing,
        # so treating them as a real mapping would propagate "LABEL_1" to the
        # client and lose the meaning entirely.
        mapping, source = guard.resolve_labels(
            _config({0: "LABEL_0", 1: "LABEL_1"})
        )
        assert mapping == {0: "BENIGN", 1: "MALICIOUS"}
        assert source.startswith("fallback")

    def test_inverted_fine_tune_is_respected_not_corrected(self) -> None:
        # If a fine-tune genuinely orders its labels the other way, that is the
        # truth about that checkpoint. Silently "fixing" it here would be the
        # inversion bug, just relocated into this service.
        mapping, source = guard.resolve_labels(
            _config({0: "MALICIOUS", 1: "BENIGN"})
        )
        assert mapping == {0: "MALICIOUS", 1: "BENIGN"}
        assert source == "checkpoint config"

    def test_provenance_distinguishes_the_two_paths(self) -> None:
        # The startup log is the only place a wrong assumption becomes visible
        # before production, so the two cases must not read the same.
        _, named = guard.resolve_labels(_config({0: "BENIGN", 1: "MALICIOUS"}))
        _, unnamed = guard.resolve_labels(_config(None))
        assert named != unnamed


class TestServiceContract:
    """The API shape the application-side adapter already depends on."""

    def test_fallback_order_matches_the_documented_convention(self) -> None:
        assert guard.FALLBACK_ID2LABEL == {0: "BENIGN", 1: "MALICIOUS"}

    def test_max_length_matches_the_model_context(self) -> None:
        # Llama-Prompt-Guard-2-86M has max_position_embeddings=512; a larger
        # value would silently truncate somewhere less predictable.
        assert guard.MAX_LENGTH <= 512
