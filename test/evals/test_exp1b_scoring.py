"""Tests for Experiment 1b's measurement machinery.

Covers the three fixes that bring Exp1b to Exp1a's standard — repeated-run
self-agreement, error separation (an API outage is not an out-of-domain
verdict), and the offline centroid detector (Metode 4).
"""

from __future__ import annotations

import numpy as np
import pytest

from evals._shared.centroid import (
    COSINE,
    MAHALANOBIS,
    fit_centroid,
    is_in_domain,
    score,
)
from evals._shared.dataset import SubsetCRow
from evals._shared.repeats import repeat_passes, self_agreement
from evals.exp1b_relevance.run import (
    ERRORED,
    IN_DOMAIN,
    JDIH_ANCHORS,
    OUT_OF_DOMAIN,
    _title_tokens,
    derive_jdih_lexicon,
    run_centroid,
    run_keyword_overlap_baseline,
    scoreable,
)


class TestLexiconDerivation:
    """The KB-derived keyword-baseline lexicon (replaces the hand-authored list)."""

    _TITLES = [
        "2503-UN40-PT.03.03-2025 - Peraturan Rektor tentang Majelis Wali Amanat",
        "1210-UN40-KP.06.00-2025 - Keputusan Rektor tentang Peraturan Rektor",
        "8 Tahun 2024 - Peraturan Rektor tentang Majelis Wali Amanat Universitas",
        "47 Tahun 2022 - Peraturan Rektor tentang Mahasiswa Kerja Paruh Waktu",
    ]

    def test_title_tokens_strip_code_and_stopwords(self) -> None:
        toks = _title_tokens("2503-UN40-PT.03.03-2025 - Peraturan Rektor tentang Majelis Wali Amanat")
        assert "peraturan" in toks and "rektor" in toks and "majelis" in toks
        assert "tentang" not in toks  # stopword
        assert not any(t.isdigit() for t in toks)  # doc code stripped
        assert "un40" not in toks

    def test_derive_includes_anchors(self) -> None:
        lex = derive_jdih_lexicon(self._TITLES, min_doc_freq=2)
        assert JDIH_ANCHORS <= lex

    def test_derive_keeps_frequent_bigrams_excludes_unigrams(self) -> None:
        # "peraturan rektor" appears in all 4 titles; "majelis wali" in 2.
        lex = derive_jdih_lexicon(self._TITLES, min_doc_freq=3)
        assert "peraturan rektor" in lex
        assert "majelis wali" not in lex  # df=2 < 3
        # A unigram that is frequent must NOT enter (bigram-only design).
        assert "rektor" not in lex
        assert "peraturan" not in lex

    def test_min_doc_freq_is_respected(self) -> None:
        loose = derive_jdih_lexicon(self._TITLES, min_doc_freq=2)
        strict = derive_jdih_lexicon(self._TITLES, min_doc_freq=4)
        assert "majelis wali" in loose and "majelis wali" not in strict
        assert strict <= loose

    def test_baseline_uses_the_passed_lexicon(self) -> None:
        rows = [
            SubsetCRow(query="Apa isi peraturan rektor UPI?", label="in_domain", subtype="direct_upi"),
            SubsetCRow(query="Bagaimana cuaca hari ini?", label="out_of_domain", subtype="off_topic"),
        ]
        preds = run_keyword_overlap_baseline(rows, {"peraturan rektor"}, threshold=1)
        assert preds == [True, False]


class TestSelfAgreement:
    """Shared repeated-run agreement, used by both experiments."""

    def test_all_passes_identical_is_full_agreement(self) -> None:
        runs = [["a", "b", "a"], ["a", "b", "a"], ["a", "b", "a"]]
        agreement, distinct = self_agreement(runs)
        assert agreement == 1.0
        assert distinct == [1, 1, 1]

    def test_a_flipped_row_lowers_agreement(self) -> None:
        runs = [["a", "b"], ["a", "b"], ["a", "X"]]
        agreement, distinct = self_agreement(runs)
        assert agreement == 0.5
        assert distinct == [1, 2]

    def test_empty_is_zero(self) -> None:
        assert self_agreement([]) == (0.0, [])

    @pytest.mark.asyncio
    async def test_repeat_passes_runs_n_times(self) -> None:
        calls = {"n": 0}

        async def one():
            calls["n"] += 1
            return [calls["n"]]

        runs = await repeat_passes(one, 3, "test pass")
        assert calls["n"] == 3
        assert runs == [[1], [2], [3]]


class TestJudgeScoreable:
    """An outage must not be scored as an out-of-domain verdict."""

    def _rows(self):
        labels = ["in_domain", "out_of_domain", "in_domain", "out_of_domain"]
        return [SubsetCRow(query=f"q{i}", label=lab, subtype="s") for i, lab in enumerate(labels)]

    def test_errored_rows_are_dropped_and_counted(self) -> None:
        # rows: [in, out, in, out]; keep indices 0 and 2 (both in_domain).
        outcomes = [IN_DOMAIN, ERRORED, OUT_OF_DOMAIN, ERRORED]
        preds, truths, errored = scoreable(outcomes, self._rows())
        assert errored == 2
        assert preds == [True, False]  # judge said in, then out on the kept rows
        assert truths == [True, True]  # both kept rows are truly in_domain

    def test_positive_class_is_in_domain(self) -> None:
        outcomes = [IN_DOMAIN, OUT_OF_DOMAIN, IN_DOMAIN, OUT_OF_DOMAIN]
        preds, truths, errored = scoreable(outcomes, self._rows())
        assert errored == 0
        assert preds == [True, False, True, False]
        assert truths == [True, False, True, False]

    def test_an_outage_does_not_become_out_of_domain(self) -> None:
        # The whole point: all-errored yields no predictions, not a set of
        # out_of_domain calls that would look like a confident detector.
        outcomes = [ERRORED, ERRORED, ERRORED, ERRORED]
        preds, truths, errored = scoreable(outcomes, self._rows())
        assert preds == []
        assert errored == 4


class TestCentroidModel:
    """The offline OOD detector's numeric core (Metode 4)."""

    def _cloud(self, n=400, d=64, seed=0):
        rng = np.random.default_rng(seed)
        center = rng.normal(size=d)
        center /= np.linalg.norm(center)
        corpus = center + 0.15 * rng.normal(size=(n, d))
        return center, corpus, rng

    def test_in_domain_query_is_closer_than_out_of_domain(self) -> None:
        center, corpus, rng = self._cloud()
        model = fit_centroid(corpus, with_mahalanobis=True)
        q_in = center + 0.15 * rng.normal(size=center.shape)
        q_out = rng.normal(size=center.shape)

        assert score(model, q_in, COSINE) > score(model, q_out, COSINE)
        assert score(model, q_in, MAHALANOBIS) < score(model, q_out, MAHALANOBIS)

    def test_cosine_direction_is_higher_is_in_domain(self) -> None:
        assert is_in_domain(0.8, 0.5, COSINE) is True
        assert is_in_domain(0.3, 0.5, COSINE) is False

    def test_mahalanobis_direction_is_lower_is_in_domain(self) -> None:
        assert is_in_domain(2.0, 5.0, MAHALANOBIS) is True
        assert is_in_domain(9.0, 5.0, MAHALANOBIS) is False

    def test_mahalanobis_needs_a_precision_matrix(self) -> None:
        _, corpus, _ = self._cloud()
        model = fit_centroid(corpus, with_mahalanobis=False)
        assert model.precision is None
        with pytest.raises(ValueError, match="precision"):
            score(model, corpus[0], MAHALANOBIS)

    def test_fit_rejects_empty(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            fit_centroid(np.empty((0, 8)))

    def test_precision_exists_when_n_below_d(self) -> None:
        # Shrinkage must keep the inverse defined even when there are fewer
        # corpus vectors than dimensions — the bge-m3 regime.
        rng = np.random.default_rng(1)
        corpus = rng.normal(size=(30, 128))
        model = fit_centroid(corpus, with_mahalanobis=True, shrinkage=0.2)
        assert model.precision is not None
        assert np.all(np.isfinite(model.precision))

    @pytest.mark.asyncio
    async def test_run_centroid_thresholds_each_query(self) -> None:
        center, corpus, rng = self._cloud()
        q_in = center + 0.1 * rng.normal(size=center.shape)
        q_out = rng.normal(size=center.shape)
        queries = np.asarray([q_in, q_out])

        preds = await run_centroid(corpus, queries, threshold=0.3, metric=COSINE, shrinkage=0.1)
        assert preds == [True, False]
