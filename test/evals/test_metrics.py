"""Tests for the evaluation metrics module.

Verifies all 11 metric implementations against hand-computed values:
    - Accuracy, Precision, Recall, F1, FPR (BinaryMetrics)
    - HitRate@k, MRR (RetrievalMetrics)
    - Cohen's Kappa (MultiClassMetrics)
    - Faithfulness
    - Abstention Accuracy
    - Token Jaccard Similarity
    - Bootstrap CI / Wilson interval bounds
"""

from __future__ import annotations

import pytest

from evals._shared.metrics import (
    BinaryMetrics,
    CI,
    MultiClassMetrics,
    RetrievalMetrics,
    abstention_accuracy,
    bert_score_f1,
    bootstrap_binary_ci,
    bootstrap_ci,
    cohen_kappa,
    compute_binary_metrics,
    compute_multiclass_metrics,
    compute_retrieval_metrics,
    faithfulness,
    hit_rate_at_k,
    reciprocal_rank_at_k,
    token_jaccard_similarity,
    wilson_interval,
)


# ---------------------------------------------------------------------------
# Binary classification metrics
# ---------------------------------------------------------------------------


class TestBinaryMetrics:
    """Tests for BinaryMetrics and compute_binary_metrics()."""

    def test_perfect_classification(self) -> None:
        """All correct predictions → accuracy=1.0, precision=1.0, recall=1.0."""
        preds = [True, True, False, False]
        gts = [True, True, False, False]
        m = compute_binary_metrics(preds, gts)
        assert m.tp == 2
        assert m.fp == 0
        assert m.tn == 2
        assert m.fn == 0
        assert m.accuracy == 1.0
        assert m.precision == 1.0
        assert m.recall == 1.0
        assert m.f1 == 1.0
        assert m.fpr == 0.0

    def test_all_wrong(self) -> None:
        """All wrong predictions → accuracy=0.0."""
        preds = [True, True, False, False]
        gts = [False, False, True, True]
        m = compute_binary_metrics(preds, gts)
        assert m.tp == 0
        assert m.fp == 2
        assert m.tn == 0
        assert m.fn == 2
        assert m.accuracy == 0.0
        assert m.precision == 0.0
        assert m.recall == 0.0
        assert m.f1 == 0.0
        assert m.fpr == 1.0

    def test_mixed_predictions(self) -> None:
        """Hand-computed: tp=3, fp=1, tn=4, fn=2."""
        # tp: pred=T,gt=T (3); fp: pred=T,gt=F (1); fn: pred=F,gt=T (2); tn: pred=F,gt=F (4)
        preds =  [True, True, True, True, False, False, False, False, False, False]
        gts =    [True, True, True, False, True, True, False, False, False, False]
        m = compute_binary_metrics(preds, gts)
        assert m.tp == 3
        assert m.fp == 1
        assert m.tn == 4
        assert m.fn == 2
        assert m.total == 10
        assert m.accuracy == pytest.approx(7 / 10)
        assert m.precision == pytest.approx(3 / 4)
        assert m.recall == pytest.approx(3 / 5)
        assert m.f1 == pytest.approx(2 * 0.75 * 0.6 / (0.75 + 0.6))
        assert m.fpr == pytest.approx(1 / 5)

    def test_length_mismatch_raises(self) -> None:
        """Mismatched lengths should raise ValueError."""
        with pytest.raises(ValueError, match="Length mismatch"):
            compute_binary_metrics([True, False], [True])

    def test_empty_raises(self) -> None:
        """Empty sequences should raise ValueError."""
        with pytest.raises(ValueError, match="empty"):
            compute_binary_metrics([], [])

    def test_fpr_no_negatives(self) -> None:
        """FPR should be 0.0 when there are no negatives."""
        m = BinaryMetrics(tp=5, fp=0, tn=0, fn=3)
        assert m.fpr == 0.0


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------


class TestRetrievalMetrics:
    """Tests for HitRate@k, MRR, and compute_retrieval_metrics()."""

    def test_hit_rate_at_k_hit(self) -> None:
        """Relevant doc in top-k → 1.0."""
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = ["doc2"]
        assert hit_rate_at_k(retrieved, relevant, k=3) == 1.0

    def test_hit_rate_at_k_miss(self) -> None:
        """Relevant doc not in top-k → 0.0."""
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = ["doc4"]
        assert hit_rate_at_k(retrieved, relevant, k=3) == 0.0

    def test_hit_rate_at_k_cutoff(self) -> None:
        """Relevant doc at position k+1 → 0.0."""
        retrieved = ["doc1", "doc2", "doc3", "doc4"]
        relevant = ["doc4"]
        assert hit_rate_at_k(retrieved, relevant, k=3) == 0.0

    def test_hit_rate_empty_relevant(self) -> None:
        """Empty relevant list → 1.0 (vacuously true)."""
        assert hit_rate_at_k(["doc1"], [], k=3) == 1.0

    def test_reciprocal_rank_first_position(self) -> None:
        """Relevant doc at rank 1 → 1.0."""
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = ["doc1"]
        assert reciprocal_rank_at_k(retrieved, relevant, k=3) == 1.0

    def test_reciprocal_rank_third_position(self) -> None:
        """Relevant doc at rank 3 → 1/3."""
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = ["doc3"]
        assert reciprocal_rank_at_k(retrieved, relevant, k=3) == pytest.approx(1 / 3)

    def test_reciprocal_rank_no_hit(self) -> None:
        """No relevant doc in top-k → 0.0."""
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = ["doc4"]
        assert reciprocal_rank_at_k(retrieved, relevant, k=3) == 0.0

    def test_compute_retrieval_metrics_aggregation(self) -> None:
        """Aggregate over multiple queries."""
        results = [
            (["doc1", "doc2"], ["doc1"]),       # hit@1, mrr@1=1.0
            (["doc3", "doc1"], ["doc1"]),         # hit@1=0, hit@3=1, mrr@3=0.5
        ]
        m = compute_retrieval_metrics(results)
        assert m.query_count == 2
        assert m.hit_rate_at_1 == pytest.approx(0.5)
        assert m.hit_rate_at_3 == pytest.approx(1.0)
        assert m.mrr_at_1 == pytest.approx(0.5)
        assert m.mrr_at_3 == pytest.approx(0.75)

    def test_compute_retrieval_metrics_empty(self) -> None:
        """Empty results → all zeros."""
        m = compute_retrieval_metrics([])
        assert m.query_count == 0
        assert m.hit_rate_at_1 == 0.0


# ---------------------------------------------------------------------------
# Multi-class metrics & Cohen's Kappa
# ---------------------------------------------------------------------------


class TestMultiClassMetrics:
    """Tests for compute_multiclass_metrics() and cohen_kappa()."""

    def test_perfect_agreement(self) -> None:
        """Perfect agreement → kappa=1.0, accuracy=1.0."""
        preds = ["A", "B", "C", "A", "B"]
        gts = ["A", "B", "C", "A", "B"]
        m = compute_multiclass_metrics(preds, gts)
        assert m.accuracy == 1.0
        assert m.cohen_kappa == pytest.approx(1.0)

    def test_random_agreement(self) -> None:
        """ kappa should be near 0 for random agreement."""
        preds = ["A", "B", "A", "B"]
        gts = ["B", "A", "B", "A"]
        k = cohen_kappa(preds, gts, ["A", "B"])
        assert k <= 0.1  # Should be low

    def test_confusion_matrix(self) -> None:
        """Confusion matrix should be correctly built."""
        preds = ["A", "A", "B", "B"]
        gts = ["A", "B", "A", "B"]
        m = compute_multiclass_metrics(preds, gts, labels=["A", "B"])
        # Row = true, Col = predicted
        # A→A: 1, A→B: 1, B→A: 1, B→B: 1
        assert m.confusion[0][0] == 1  # true A, pred A
        assert m.confusion[0][1] == 1  # true A, pred B
        assert m.confusion[1][0] == 1  # true B, pred A
        assert m.confusion[1][1] == 1  # true B, pred B

    def test_per_class_metrics(self) -> None:
        """Per-class precision/recall/F1.

        preds = ["A", "A", "B", "B", "C"]
        gts   = ["A", "A", "B", "C", "C"]

        Confusion (rows=true, cols=pred):
            A  B  C
        A:  2  0  0
        B:  0  1  0
        C:  0  1  1

        Class A: tp=2, fp=0, fn=0 → P=1, R=1, F1=1
        Class B: tp=1, fp=1, fn=0 → P=0.5, R=1.0
        Class C: tp=1, fp=0, fn=1 → P=1, R=0.5
        """
        preds = ["A", "A", "B", "B", "C"]
        gts = ["A", "A", "B", "C", "C"]
        m = compute_multiclass_metrics(preds, gts, labels=["A", "B", "C"])
        # Class A: tp=2, fp=0, fn=0 → P=1, R=1, F1=1
        assert m.per_class["A"] == (1.0, 1.0, 1.0)
        # Class B: tp=1, fp=1 (one B pred has gt=C), fn=0
        # → P=0.5, R=1.0
        assert m.per_class["B"][0] == pytest.approx(0.5)
        assert m.per_class["B"][1] == pytest.approx(1.0)
        # Class C: tp=1, fp=0, fn=1 (one true C pred as B)
        # → P=1.0, R=0.5
        assert m.per_class["C"][0] == pytest.approx(1.0)
        assert m.per_class["C"][1] == pytest.approx(0.5)

    def test_macro_averages(self) -> None:
        """Macro-averaged metrics."""
        preds = ["A", "A", "B", "B"]
        gts = ["A", "A", "B", "B"]
        m = compute_multiclass_metrics(preds, gts, labels=["A", "B"])
        assert m.macro_precision == 1.0
        assert m.macro_recall == 1.0
        assert m.macro_f1 == 1.0

    def test_length_mismatch_raises(self) -> None:
        """Mismatched lengths should raise."""
        with pytest.raises(ValueError):
            compute_multiclass_metrics(["A"], ["A", "B"])

    def test_empty_raises(self) -> None:
        """Empty sequences should raise."""
        with pytest.raises(ValueError):
            compute_multiclass_metrics([], [])


# ---------------------------------------------------------------------------
# Faithfulness
# ---------------------------------------------------------------------------


class TestFaithfulness:
    """Tests for faithfulness()."""

    def test_all_supported(self) -> None:
        """All supported → 1.0."""
        labels = ["supported", "supported", "supported"]
        assert faithfulness(labels) == 1.0

    def test_all_not_supported(self) -> None:
        """All not_supported → 0.0."""
        labels = ["not_supported", "not_supported"]
        assert faithfulness(labels) == 0.0

    def test_mixed_with_no_source_needed(self) -> None:
        """no_source_needed excluded from denominator."""
        labels = ["supported", "not_supported", "no_source_needed"]
        # s_supported=1, s_verifiable=2 → 0.5
        assert faithfulness(labels) == pytest.approx(0.5)

    def test_all_no_source_needed(self) -> None:
        """All no_source_needed → 0.0 (no verifiable sentences)."""
        labels = ["no_source_needed", "no_source_needed"]
        assert faithfulness(labels) == 0.0

    def test_nli_labels(self) -> None:
        """NLI labels (entailment/contradiction/neutral) should work."""
        labels = ["entailment", "contradiction", "neutral", "no_source_needed"]
        # s_supported=1 (entailment), s_verifiable=3 → 1/3
        assert faithfulness(labels) == pytest.approx(1 / 3)

    def test_empty(self) -> None:
        """Empty labels → 0.0."""
        assert faithfulness([]) == 0.0


# ---------------------------------------------------------------------------
# Abstention Accuracy
# ---------------------------------------------------------------------------


class TestAbstentionAccuracy:
    """Tests for abstention_accuracy()."""

    def test_all_correct(self) -> None:
        """All abstained correctly → 1.0."""
        abstained = [True, True, True]
        assert abstention_accuracy(abstained, 3) == 1.0

    def test_none_correct(self) -> None:
        """None abstained → 0.0."""
        abstained = [False, False, False]
        assert abstention_accuracy(abstained, 3) == 0.0

    def test_partial(self) -> None:
        """2/3 correct → 2/3."""
        abstained = [True, True, False]
        assert abstention_accuracy(abstained, 3) == pytest.approx(2 / 3)

    def test_zero_total(self) -> None:
        """Zero out-of-domain → 0.0."""
        assert abstention_accuracy([], 0) == 0.0


# ---------------------------------------------------------------------------
# Token Jaccard Similarity
# ---------------------------------------------------------------------------


class TestTokenJaccard:
    """Tests for token_jaccard_similarity()."""

    def test_identical(self) -> None:
        """Identical texts → 1.0."""
        assert token_jaccard_similarity("hello world", "hello world") == 1.0

    def test_no_overlap(self) -> None:
        """No shared tokens → 0.0."""
        assert token_jaccard_similarity("hello world", "foo bar") == 0.0

    def test_partial_overlap(self) -> None:
        """Partial overlap → 1/3."""
        # tokens_a = {a, b, c}, tokens_b = {a, b, d}
        # intersection = {a, b} = 2, union = {a, b, c, d} = 4 → 0.5
        assert token_jaccard_similarity("a b c", "a b d") == pytest.approx(0.5)

    def test_case_insensitive(self) -> None:
        """Should be case-insensitive."""
        assert token_jaccard_similarity("Hello World", "hello world") == 1.0

    def test_both_empty(self) -> None:
        """Both empty → 1.0."""
        assert token_jaccard_similarity("", "") == 1.0


# ---------------------------------------------------------------------------
# Confidence Intervals
# ---------------------------------------------------------------------------


class TestConfidenceIntervals:
    """Tests for bootstrap_ci() and wilson_interval()."""

    def test_bootstrap_ci_point_estimate(self) -> None:
        """Bootstrap CI point estimate should equal the mean."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        ci = bootstrap_ci(values, n_resamples=500, seed=42)
        assert ci.point == pytest.approx(3.0)
        assert ci.lower <= ci.point <= ci.upper
        assert ci.method == "bootstrap"

    def test_bootstrap_ci_empty(self) -> None:
        """Empty values → CI(0, 0, 0)."""
        ci = bootstrap_ci([])
        assert ci.point == 0.0
        assert ci.lower == 0.0
        assert ci.upper == 0.0

    def test_bootstrap_ci_bounds(self) -> None:
        """CI bounds should contain the point estimate."""
        values = [0.1, 0.5, 0.9, 0.3, 0.7]
        ci = bootstrap_ci(values, n_resamples=1000, seed=42)
        assert ci.lower <= ci.point
        assert ci.upper >= ci.point

    def test_wilson_interval(self) -> None:
        """Wilson interval for 8/10 successes."""
        ci = wilson_interval(8, 10)
        assert ci.point == pytest.approx(0.8)
        assert ci.lower < 0.8
        assert ci.upper > 0.8
        assert ci.method == "wilson"

    def test_wilson_interval_zero_total(self) -> None:
        """Zero total → CI(0, 0, 0)."""
        ci = wilson_interval(0, 0)
        assert ci.point == 0.0

    def test_wilson_interval_all_success(self) -> None:
        """All successes → point=1.0, upper≈1.0."""
        ci = wilson_interval(10, 10)
        assert ci.point == 1.0
        assert ci.upper == pytest.approx(1.0)

    def test_bootstrap_binary_ci(self) -> None:
        """Bootstrap CI for binary accuracy."""
        preds = [True, True, True, False, False]
        gts = [True, True, False, False, False]
        ci = bootstrap_binary_ci(preds, gts, metric="accuracy", n_resamples=500, seed=42)
        # accuracy = 4/5 = 0.8
        assert ci.point == pytest.approx(0.8)
        assert ci.lower <= ci.point <= ci.upper


# ---------------------------------------------------------------------------
# BERTScore (fallback test)
# ---------------------------------------------------------------------------


class TestBertScore:
    """Tests for bert_score_f1() — only tests fallback behavior."""

    def test_empty_inputs(self) -> None:
        """Empty inputs → (0, 0, 0, [])."""
        p, r, f1, f1_per_example = bert_score_f1([], [])
        assert p == 0.0
        assert r == 0.0
        assert f1 == 0.0
        assert f1_per_example == []

    def test_length_mismatch_raises(self) -> None:
        """Mismatched lengths should raise."""
        with pytest.raises(ValueError):
            bert_score_f1(["a"], ["a", "b"])
