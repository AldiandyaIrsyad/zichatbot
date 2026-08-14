"""Tests for the combined Subset A-D blind human audit."""

from __future__ import annotations

import csv
import importlib.util
import json
from collections import Counter
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rows(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_combined_builder_has_exact_counts_and_sealed_queue(tmp_path):
    builder = _load("blind_builder", "evals/blind_test/build_blind_test_ad_v2.py")
    queue = tmp_path / "queue.csv"
    key = tmp_path / "key.csv"
    meta = tmp_path / "meta.json"
    html = tmp_path / "review.html"
    builder.build(
        queue,
        key,
        meta,
        ROOT / "evals/blind_test/blind_test_ad_v2.template.html",
        html,
    )

    queue_rows = _rows(queue)
    key_rows = _rows(key)
    assert len(queue_rows) == len(key_rows) == 173
    assert not {
        "reference_label",
        "origin",
        "qc_family",
        "injected_wrong_label",
    }.intersection(queue_rows[0])
    assert Counter(
        row["subset"] for row in key_rows if row["origin"] == "audit_sample"
    ) == Counter({"subset_a": 30, "subset_b": 32, "subset_c": 40, "subset_d": 42})
    assert Counter(
        row["subset"] for row in key_rows if row["origin"] == "qc_control"
    ) == Counter({"subset_a": 6, "subset_b": 6, "subset_c": 8, "subset_d": 9})
    assert Counter(
        row["reference_label"]
        for row in key_rows
        if row["subset"] == "subset_d" and row["origin"] == "audit_sample"
    ) == Counter({"entailment": 17, "neutral": 15, "contradiction": 10})
    assert "__ITEMS_JSON__" not in html.read_text(encoding="utf-8")


def test_independent_scorer_recomputes_and_rejects_tampering(tmp_path):
    builder = _load("blind_builder_score", "evals/blind_test/build_blind_test_ad_v2.py")
    scorer = _load("blind_scorer", "evals/blind_test/score_blind_test_ad_v2.py")
    queue = tmp_path / "queue.csv"
    key = tmp_path / "key.csv"
    meta = tmp_path / "meta.json"
    html = tmp_path / "review.html"
    builder.build(
        queue,
        key,
        meta,
        ROOT / "evals/blind_test/blind_test_ad_v2.template.html",
        html,
    )

    metadata = json.loads(meta.read_text(encoding="utf-8"))
    key_rows = _rows(key)
    reviewed = tmp_path / "reviewed.csv"
    fields = [
        "item_id",
        "subset",
        "task",
        "human_label",
        "reference_label",
        "correct",
        "origin",
        "qc_family",
        "injected_wrong_label",
        "qc_detected",
        "source_row_id",
        "source_sha256",
        "config_sha256",
    ]
    with reviewed.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in key_rows:
            writer.writerow(
                {
                    **row,
                    "human_label": row["reference_label"],
                    "correct": "0",
                    "qc_detected": "0",
                    "config_sha256": metadata["config_sha256"],
                }
            )

    verified_csv = tmp_path / "verified.csv"
    verified_json = tmp_path / "verified.json"
    verified_md = tmp_path / "verified.md"
    scorer.score(reviewed, key, meta, verified_csv, verified_json, verified_md)
    summary = json.loads(verified_json.read_text(encoding="utf-8"))
    assert summary["pooled_authentic"]["correct"] == 144
    assert summary["pooled_qc"]["correct"] == 29
    assert "Tabel 4.6" in verified_md.read_text(encoding="utf-8")

    tampered = _rows(reviewed)
    tampered[0]["reference_label"] = "tampered"
    with reviewed.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(tampered)
    with pytest.raises(ValueError, match="Sealed field mismatch"):
        scorer.score(reviewed, key, meta, verified_csv, verified_json, verified_md)
