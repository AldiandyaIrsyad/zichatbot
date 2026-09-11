"""Tests for the hard Subset D rebuild's construction machinery.

Covers the pieces that make the rebuild honest without a running stack:
counterfactual generation + no-op rejection, the verification-gate contract,
adversarial difficulty banding (a tag, never a gate), the surface-feature
shortcut assertion, the class-balance floor, and the extended schema's
backward compatibility with the legacy 2026-07-19 files.
"""

from __future__ import annotations

import pytest

from evals._dataset_gen.perturbations import (
    BAND_EASY,
    BAND_HARD,
    BAND_MEDIUM,
    FAMILIES,
    FAMILY_BY_NAME,
    LABEL_TO_NLI,
    NOT_SUPPORTED,
    PARTIALLY_SUPPORTED,
    SUPPORTED,
    assert_class_balance,
    assert_no_shortcut_features,
    band_by_nli,
    generate_perturbation,
    nli_class_counts,
    _parse_perturbation,
)


class _FakeGenerator:
    """Minimal stand-in for DatasetGenerator with a scripted reply."""

    def __init__(self, reply, raises=False):
        self._reply = reply
        self._raises = raises
        self.calls = 0

    async def generate_single(self, prompt, system_prompt=None):  # noqa: D401
        self.calls += 1
        if self._raises:
            raise RuntimeError("upstream 500")
        return self._reply


class TestFamilies:
    def test_every_family_maps_to_a_scored_nli_class_or_is_partial(self) -> None:
        # Each family's intended 4-label must be one Exp3 knows how to map.
        for fam in FAMILIES:
            assert fam.intended_label in LABEL_TO_NLI
        # The set spans all three NLI classes so the core can be balanced.
        intended = {LABEL_TO_NLI[f.intended_label] for f in FAMILIES}
        assert intended == {"entailment", "neutral", "contradiction"}

    def test_family_lookup(self) -> None:
        assert FAMILY_BY_NAME["factual_flip"].intended_label == NOT_SUPPORTED
        assert FAMILY_BY_NAME["evaluative_graft"].intended_label == PARTIALLY_SUPPORTED
        assert FAMILY_BY_NAME["far_paraphrase"].intended_label == SUPPORTED


class TestParseJsonObject:
    """The shared parser that unblocked the Subset-D question generators."""

    def test_multiline_pretty_printed_json(self) -> None:
        # This is the exact shape that broke the old line-by-line JSONL parser.
        from evals._dataset_gen.generator import parse_json_object

        raw = '{\n  "question": "Berapa UKT prodi X?",\n  "note": "detail"\n}'
        obj = parse_json_object(raw)
        assert obj is not None and obj["question"] == "Berapa UKT prodi X?"

    def test_fenced_and_prose_wrapped(self) -> None:
        from evals._dataset_gen.generator import parse_json_object

        assert parse_json_object('```json\n{"question": "Q"}\n```')["question"] == "Q"
        assert parse_json_object('Sure: {"question": "Q"} done')["question"] == "Q"

    def test_garbage_is_none(self) -> None:
        from evals._dataset_gen.generator import parse_json_object

        assert parse_json_object("no json") is None
        assert parse_json_object("") is None
        assert parse_json_object("[1, 2, 3]") is None  # array, not object


class TestParsePerturbation:
    def test_bare_object(self) -> None:
        assert _parse_perturbation('{"perturbed":"X","edit_note":"n"}') == ("X", "n")

    def test_fenced_and_prose_wrapped(self) -> None:
        assert _parse_perturbation('```json\n{"perturbed":"Y","edit_note":"z"}\n```') == ("Y", "z")
        assert _parse_perturbation('Sure: {"perturbed":"Z","edit_note":"q"} ok') == ("Z", "q")

    def test_missing_field_or_garbage_is_none(self) -> None:
        assert _parse_perturbation('{"edit_note":"no perturbed field"}') is None
        assert _parse_perturbation("not json at all") is None
        assert _parse_perturbation("") is None


class TestGeneratePerturbation:
    @pytest.mark.asyncio
    async def test_success_returns_edit_and_note(self) -> None:
        gen = _FakeGenerator('{"perturbed":"UKT sepuluh juta","edit_note":"flipped 5->10"}')
        out = await generate_perturbation(gen, "UKT lima juta", "context", FAMILIES[0])
        assert out == ("UKT sepuluh juta", "flipped 5->10")

    @pytest.mark.asyncio
    async def test_noop_edit_is_rejected(self) -> None:
        # An edit identical to the input carries no label signal.
        gen = _FakeGenerator('{"perturbed":"UKT lima juta","edit_note":"none"}')
        out = await generate_perturbation(gen, "UKT lima juta", "context", FAMILIES[0])
        assert out is None

    @pytest.mark.asyncio
    async def test_generator_error_is_swallowed_as_none(self) -> None:
        gen = _FakeGenerator("", raises=True)
        out = await generate_perturbation(gen, "s", "c", FAMILIES[0])
        assert out is None


class TestVerificationGateContract:
    """The gate keeps a perturbed row iff the panel confirms the intended label.

    We model the gate directly (the builder wires the real panel via
    ``label_sentence``): keep when panel_label == intended, drop otherwise.
    """

    @staticmethod
    def _gate(panel_label, intended):
        return panel_label == intended

    def test_confirmed_label_is_kept(self) -> None:
        assert self._gate(NOT_SUPPORTED, NOT_SUPPORTED) is True

    def test_panel_disagreement_is_dropped(self) -> None:
        # Generator aimed for not_supported but the panel still reads it as
        # supported (the flipped number happened to appear in context): drop it.
        assert self._gate(SUPPORTED, NOT_SUPPORTED) is False


class TestBanding:
    def test_wrong_prediction_is_hard(self) -> None:
        assert band_by_nli("entailment", "contradiction") == BAND_HARD

    def test_correct_and_confident_is_easy(self) -> None:
        assert band_by_nli("entailment", "entailment", 0.95) == BAND_EASY

    def test_correct_low_confidence_is_medium(self) -> None:
        assert band_by_nli("neutral", "neutral", 0.4) == BAND_MEDIUM

    def test_correct_without_score_is_medium(self) -> None:
        assert band_by_nli("contradiction", "contradiction") == BAND_MEDIUM


class TestShortcutAssertion:
    _CTX = "mahasiswa wajib membayar ukt sebesar lima juta rupiah setiap semester berjalan"

    def test_minimal_edit_negatives_pass(self) -> None:
        # not_supported rows share the high-overlap band with supported ones,
        # because the edit is one word — so no surface feature is decisive.
        rows = []
        for _ in range(20):
            rows.append({"sentence_text": "mahasiswa wajib membayar ukt lima juta rupiah",
                         "retrieved_context": self._CTX, "label": SUPPORTED})
            rows.append({"sentence_text": "mahasiswa wajib membayar ukt sepuluh juta rupiah",
                         "retrieved_context": self._CTX, "label": NOT_SUPPORTED})
        report = assert_no_shortcut_features(rows)  # must not raise
        assert any(c["feature"] == "overlap" for c in report)

    def test_overlap_shortcut_trips(self) -> None:
        # Degenerate: every not_supported row has near-zero overlap (a different
        # topic) while supported rows are high overlap — overlap predicts label.
        rows = []
        for _ in range(20):
            rows.append({"sentence_text": "mahasiswa wajib membayar ukt lima juta rupiah",
                         "retrieved_context": self._CTX, "label": SUPPORTED})
            rows.append({"sentence_text": "cuaca hari ini sangat cerah dan panas sekali",
                         "retrieved_context": self._CTX, "label": NOT_SUPPORTED})
        with pytest.raises(AssertionError, match="surface feature"):
            assert_no_shortcut_features(rows)


class TestClassBalance:
    def _rows(self, entail, neutral, contra):
        rows = []
        rows += [{"label": SUPPORTED}] * entail
        rows += [{"label": PARTIALLY_SUPPORTED}] * neutral
        rows += [{"label": NOT_SUPPORTED}] * contra
        return rows

    def test_passes_above_floor(self) -> None:
        counts = assert_class_balance(self._rows(60, 60, 60), floor=60)
        assert counts == {"entailment": 60, "neutral": 60, "contradiction": 60}

    def test_trips_when_a_class_is_short(self) -> None:
        with pytest.raises(AssertionError, match="class balance floor"):
            assert_class_balance(self._rows(60, 5, 60), floor=60)

    def test_no_source_needed_is_not_counted(self) -> None:
        rows = self._rows(60, 60, 60) + [{"label": "no_source_needed"}] * 99
        counts = nli_class_counts(rows)
        assert "no_source_needed" not in counts and sum(counts.values()) == 180


class TestOrchestration:
    """Phase-2 gate + quota logic, with the panel and generator faked out."""

    @staticmethod
    def _parent(qid, text):
        # Mirrors a real phase-1 natural row (tagged with split='natural').
        return {
            "question_id": qid, "question": "Q", "full_response": text,
            "sentence_id": 0, "sentence_text": text, "retrieved_context": "ctx",
            "label": SUPPORTED, "verifier_note": "n", "construction": "natural",
            "split": "natural",
        }

    def test_family_quotas_ceil(self) -> None:
        from evals._dataset_gen.build_subset_d import _family_quotas

        q5 = _family_quotas(5)
        assert q5["factual_flip"] == 3 and q5["scope_negation_flip"] == 3
        assert q5["plausible_absent_detail"] == 3 and q5["evaluative_graft"] == 3
        assert q5["far_paraphrase"] == 2

        q6 = _family_quotas(6)
        assert q6["factual_flip"] == 3 and q6["scope_negation_flip"] == 3
        assert q6["plausible_absent_detail"] == 3 and q6["evaluative_graft"] == 3
        assert q6["far_paraphrase"] == 3

    def test_tag_fills_and_overrides(self) -> None:
        from evals._dataset_gen.build_subset_d import SLICE_FIELDS, _tag

        base = {"label": SUPPORTED, "sentence_text": "s"}
        out = _tag(base, construction="perturbed", split="core")
        assert all(k in out for k in SLICE_FIELDS)
        assert out["construction"] == "perturbed" and out["split"] == "core"
        assert out["difficulty_band"] == ""  # untouched slice field defaults empty

    @pytest.mark.asyncio
    async def test_gate_keeps_only_intended_and_caps_at_quota(self, monkeypatch) -> None:
        import evals._dataset_gen.build_subset_d as bd

        async def fake_perturb(generator, sentence, grounding, family):
            return (f"edited {sentence}", "note")

        async def fake_label(**kwargs):
            # The panel always reads the edit as not_supported, so only the two
            # not_supported families can pass the gate; evaluative/far/absent are dropped.
            row = {
                "question_id": kwargs["question_id"], "question": kwargs["question"],
                "full_response": kwargs["full_response"], "sentence_id": 0,
                "sentence_text": kwargs["sentence"], "retrieved_context": kwargs["retrieved_context"],
                "label": NOT_SUPPORTED, "verifier_note": "panel",
            }
            return row, "accepted", True

        monkeypatch.setattr(bd, "generate_perturbation", fake_perturb)
        monkeypatch.setattr(bd, "label_sentence", fake_label)

        naturals = [self._parent(f"q-{i:03d}", "mahasiswa wajib membayar ukt lima juta") for i in range(15)]
        chunks_by_qid = {r["question_id"]: [] for r in naturals}

        core_rows, kept, gate = await bd.phase2_perturb(
            panel=None, generator=None, natural_rows=naturals,
            chunks_by_qid=chunks_by_qid, core_floor=6, seed=0,
        )
        # Only not_supported families are kept, each capped at its quota (3).
        assert kept.get("factual_flip") == 3
        assert kept.get("scope_negation_flip") == 3
        assert kept.get("plausible_absent_detail", 0) == 0  # panel said not_supported != partial
        assert kept.get("evaluative_graft", 0) == 0  # panel said not_supported != partial
        assert kept.get("far_paraphrase", 0) == 0
        assert all(r["construction"] == "perturbed" and r["split"] == "core" for r in core_rows)
        assert all(r["perturbation_of"].endswith(":0") for r in core_rows)
        assert gate["verified"] == 6 and gate["rejected"] > 0

    def test_promote_balances_entailment(self) -> None:
        from evals._dataset_gen.build_subset_d import promote_supported_to_core

        naturals = [self._parent(f"q-{i:03d}", "kalimat pendukung alami") for i in range(50)]
        # core already holds 8 contradiction + 8 neutral, 0 entailment.
        core = [{"label": NOT_SUPPORTED, "split": "core"}] * 8 + [{"label": PARTIALLY_SUPPORTED, "split": "core"}] * 8
        promoted = promote_supported_to_core(naturals, core, core_floor=6, seed=1)
        # target = max(6, 8, 8) = 8 entailment needed, none present -> promote 8.
        assert promoted == 8
        assert sum(1 for r in naturals if r["split"] == "core") == 8


class TestExp3SliceReporting:
    """Exp3's dual-slice breakdown over split / difficulty_band."""

    def _rows(self):
        from evals._shared.dataset import SubsetDRow

        return [
            SubsetDRow("q", "q", "r", 0, "s", "c", "supported", "n", split="core", difficulty_band="hard"),
            SubsetDRow("q", "q", "r", 1, "s", "c", "not_supported", "n", split="core", difficulty_band="easy"),
            SubsetDRow("q", "q", "r", 2, "s", "c", "supported", "n", split="natural", difficulty_band="easy"),
        ]

    def test_groups_by_split(self) -> None:
        from evals.exp3_ram.run import slice_breakdown

        rows = self._rows()
        gts = ["entailment", "contradiction", "entailment"]
        out = slice_breakdown(rows, gts, gts, "split")
        by = {k: (n, acc) for k, n, acc, _ in out}
        assert by["core"] == (2, 1.0)
        assert by["natural"] == (1, 1.0)

    def test_legacy_rows_yield_no_slice(self) -> None:
        from evals._shared.dataset import SubsetDRow
        from evals.exp3_ram.run import slice_breakdown

        legacy = [SubsetDRow("q", "q", "r", 0, "s", "c", "supported", "n")]
        assert slice_breakdown(legacy, ["entailment"], ["entailment"], "split") == []


class TestSchemaBackCompat:
    def test_legacy_file_loads_with_empty_slice_fields(self, tmp_path) -> None:
        from evals._shared.dataset import load_subset_d

        csv_path = tmp_path / "legacy.csv"
        csv_path.write_text(
            "question_id,question,full_response,sentence_id,sentence_text,"
            "retrieved_context,label,verifier_note\n"
            "q-001,Q,R,0,S,C,supported,note\n",
            encoding="utf-8",
        )
        rows = load_subset_d(str(csv_path))
        assert rows[0].construction == "" and rows[0].split == ""
        assert rows[0].label == "supported"

    def test_extended_file_populates_slice_fields(self, tmp_path) -> None:
        from evals._shared.dataset import load_subset_d

        csv_path = tmp_path / "extended.csv"
        csv_path.write_text(
            "question_id,question,full_response,sentence_id,sentence_text,"
            "retrieved_context,label,verifier_note,construction,perturbation_family,"
            "intended_label,perturbation_of,difficulty_band,split,edit_note\n"
            "q-001,Q,R,0,S,C,not_supported,note,perturbed,factual_flip,"
            "not_supported,q-000:2,hard,core,flipped 5->10\n",
            encoding="utf-8",
        )
        rows = load_subset_d(str(csv_path))
        r = rows[0]
        assert r.construction == "perturbed"
        assert r.perturbation_family == "factual_flip"
        assert r.difficulty_band == "hard"
        assert r.split == "core"
        assert r.perturbation_of == "q-000:2"
