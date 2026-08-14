"""Tests for the Subset D build's fail-safe resume bookkeeping (C29).

The build's panel spend (5 models × hundreds of sentences) is now written per
row and resumable, so an abort never re-pays. These tests pin the pure
reconstruction logic — the counters/skip-sets phase 2 rebuilds from the rows
already on disk — plus the checkpoint CSV round-trip it relies on.
"""

from __future__ import annotations

from typing import Any, Dict, List

from evals._dataset_gen.build_subset_d import FIELDNAMES, core_resume_state
from evals._dataset_gen.checkpoint import IncrementalCSVWriter, resume_rows


def _core_row(qp_idx: int, family: str, parent_ref: str) -> Dict[str, Any]:
    return {
        "question_id": f"qp-{family}-{qp_idx:04d}",
        "question": "q",
        "full_response": "edited sentence.",
        "sentence_id": "0",
        "sentence_text": "edited sentence.",
        "retrieved_context": "ctx",
        "label": "not_supported",
        "verifier_note": "",
        "construction": "perturbed",
        "perturbation_family": family,
        "intended_label": "not_supported",
        "perturbation_of": parent_ref,
        "difficulty_band": "",
        "split": "core",
        "edit_note": "",
    }


class TestCoreResumeState:
    def test_empty_is_zeroed(self) -> None:
        kept, done_pairs, counter = core_resume_state([])
        assert kept == {}
        assert done_pairs == set()
        assert counter == 0

    def test_reconstructs_kept_counts_and_pairs_and_counter(self) -> None:
        rows = [
            _core_row(1, "factual_flip", "q-reweighted-001:2"),
            _core_row(2, "factual_flip", "q-reweighted-004:0"),
            _core_row(7, "evaluative_graft", "q-detail-002:1"),
        ]
        kept, done_pairs, counter = core_resume_state(rows)
        assert kept == {"factual_flip": 2, "evaluative_graft": 1}
        assert ("q-reweighted-001:2", "factual_flip") in done_pairs
        assert ("q-detail-002:1", "evaluative_graft") in done_pairs
        # highest qp index seen, so new ids won't collide
        assert counter == 7

    def test_done_pairs_prevent_re_perturbing_same_parent_family(self) -> None:
        rows = [_core_row(1, "scope_negation_flip", "q-crossdoc-003:1")]
        _, done_pairs, _ = core_resume_state(rows)
        # same parent, same family → skip; same parent, different family → allowed
        assert ("q-crossdoc-003:1", "scope_negation_flip") in done_pairs
        assert ("q-crossdoc-003:1", "far_paraphrase") not in done_pairs


class TestCheckpointRoundTrip:
    def test_incremental_write_then_resume_reads_back(self, tmp_path: Any) -> None:
        path = str(tmp_path / "subset_d.core.part.csv")
        written: List[Dict[str, Any]] = [
            _core_row(1, "factual_flip", "q-reweighted-001:2"),
            _core_row(2, "evaluative_graft", "q-detail-001:0"),
        ]
        with IncrementalCSVWriter(path, FIELDNAMES, resume=False) as w:
            for r in written:
                w.append(r)
        back = resume_rows(path, FIELDNAMES)
        assert len(back) == 2
        assert [r["question_id"] for r in back] == [r["question_id"] for r in written]
        # and the reconstruction agrees with what we wrote
        kept, _, counter = core_resume_state(back)
        assert kept == {"factual_flip": 1, "evaluative_graft": 1}
        assert counter == 2

    def test_resume_appends_rather_than_truncates(self, tmp_path: Any) -> None:
        path = str(tmp_path / "subset_d.core.part.csv")
        with IncrementalCSVWriter(path, FIELDNAMES, resume=False) as w:
            w.append(_core_row(1, "factual_flip", "q-reweighted-001:2"))
        with IncrementalCSVWriter(path, FIELDNAMES, resume=True) as w:
            w.append(_core_row(2, "far_paraphrase", "q-reweighted-002:0"))
        back = resume_rows(path, FIELDNAMES)
        assert len(back) == 2, "resume must append, not overwrite the paid-for first row"
