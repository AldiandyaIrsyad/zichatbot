"""Evaluation metrics for the thesis experiments.

Implements Accuracy, Precision, Recall, F1, False Positive Rate, Hit Rate@k,
Mean Reciprocal Rank, BERTScore, Faithfulness, Abstention Accuracy, and Cohen's
Kappa. Point estimates are paired with bootstrap confidence intervals
(resampling 1000×) or Wilson intervals for binary proportions.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Binary classification metrics (Exp 1a, 1b)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BinaryMetrics:
    """Confusion-matrix-derived metrics for binary classification (tp/fp/tn/fn)."""

    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0

    @property
    def total(self) -> int:
        """Total number of samples."""
        return self.tp + self.fp + self.tn + self.fn

    @property
    def accuracy(self) -> float:
        """Proportion of correct predictions."""
        n = self.total
        return (self.tp + self.tn) / n if n else 0.0

    @property
    def precision(self) -> float:
        """Proportion of positive predictions that are correct."""
        denom = self.tp + self.fp
        return self.tp / denom if denom else 0.0

    @property
    def recall(self) -> float:
        """Proportion of actual positives found."""
        denom = self.tp + self.fn
        return self.tp / denom if denom else 0.0

    @property
    def f1(self) -> float:
        """Harmonic mean of precision and recall."""
        p, r = self.precision, self.recall
        denom = p + r
        return 2 * p * r / denom if denom else 0.0

    @property
    def fpr(self) -> float:
        """False Positive Rate — proportion of negatives misclassified."""
        denom = self.fp + self.tn
        return self.fp / denom if denom else 0.0


def compute_binary_metrics(
    predictions: Sequence[bool],
    ground_truths: Sequence[bool],
) -> BinaryMetrics:
    """Compute binary classification metrics from prediction/ground-truth lists.

    Both sequences use True = positive class.

    Raises:
        ValueError: If lengths mismatch or lists are empty.
    """
    if len(predictions) != len(ground_truths):
        raise ValueError(
            f"Length mismatch: {len(predictions)} predictions vs "
            f"{len(ground_truths)} ground truths."
        )
    if not predictions:
        raise ValueError("Cannot compute metrics on empty sequences.")

    tp = fp = tn = fn = 0
    for pred, gt in zip(predictions, ground_truths):
        if pred and gt:
            tp += 1
        elif pred and not gt:
            fp += 1
        elif not pred and gt:
            fn += 1
        else:
            tn += 1
    return BinaryMetrics(tp=tp, fp=fp, tn=tn, fn=fn)


# ---------------------------------------------------------------------------
# Retrieval metrics (Exp 2)
# ---------------------------------------------------------------------------


def hit_rate_at_k(
    retrieved_doc_ids: Sequence[str],
    relevant_doc_ids: Sequence[str],
    k: int,
) -> float:
    """Compute Hit Rate@k for a single query.

    Returns 1.0 if any relevant doc appears in the top-``k`` of
    ``retrieved_doc_ids``, else 0.0.
    """
    if not relevant_doc_ids:
        return 1.0
    top_k = set(retrieved_doc_ids[:k])
    return 1.0 if any(rid in top_k for rid in relevant_doc_ids) else 0.0


def reciprocal_rank_at_k(
    retrieved_doc_ids: Sequence[str],
    relevant_doc_ids: Sequence[str],
    k: int,
) -> float:
    """Compute Reciprocal Rank@k for a single query.

    Returns 1/rank of the first relevant hit within the top-``k``, or 0.0.
    """
    if not relevant_doc_ids:
        return 1.0
    relevant_set = set(relevant_doc_ids)
    for rank, doc_id in enumerate(retrieved_doc_ids[:k], start=1):
        if doc_id in relevant_set:
            return 1.0 / rank
    return 0.0


@dataclass(frozen=True)
class RetrievalMetrics:
    """Aggregated retrieval metrics over all queries.

    Mean Hit Rate@k and MRR@k for k ∈ {1, 3, 5}, plus ``query_count``.
    """

    hit_rate_at_1: float
    hit_rate_at_3: float
    hit_rate_at_5: float
    mrr_at_1: float
    mrr_at_3: float
    mrr_at_5: float
    query_count: int


def compute_retrieval_metrics(
    results_per_query: Sequence[Tuple[Sequence[str], Sequence[str]]],
) -> RetrievalMetrics:
    """Compute aggregated Hit Rate@k and MRR@k for k ∈ {1, 3, 5}.

    ``results_per_query`` is a list of (retrieved_doc_ids, relevant_doc_ids)
    tuples.
    """
    if not results_per_query:
        return RetrievalMetrics(0, 0, 0, 0, 0, 0, 0)

    n = len(results_per_query)
    hr1 = hr3 = hr5 = 0.0
    mrr1 = mrr3 = mrr5 = 0.0

    for retrieved, relevant in results_per_query:
        hr1 += hit_rate_at_k(retrieved, relevant, 1)
        hr3 += hit_rate_at_k(retrieved, relevant, 3)
        hr5 += hit_rate_at_k(retrieved, relevant, 5)
        mrr1 += reciprocal_rank_at_k(retrieved, relevant, 1)
        mrr3 += reciprocal_rank_at_k(retrieved, relevant, 3)
        mrr5 += reciprocal_rank_at_k(retrieved, relevant, 5)

    return RetrievalMetrics(
        hit_rate_at_1=hr1 / n,
        hit_rate_at_3=hr3 / n,
        hit_rate_at_5=hr5 / n,
        mrr_at_1=mrr1 / n,
        mrr_at_3=mrr3 / n,
        mrr_at_5=mrr5 / n,
        query_count=n,
    )


# ---------------------------------------------------------------------------
# Multi-class metrics (Exp 3)
# ---------------------------------------------------------------------------


@dataclass
class MultiClassMetrics:
    """Per-class precision/recall/F1 + macro averages + Cohen's Kappa.

    ``per_class`` maps label → (precision, recall, f1); ``confusion`` is a 2D
    matrix with rows = true, cols = predicted.
    """

    labels: List[str]
    per_class: Dict[str, Tuple[float, float, float]] = field(default_factory=dict)
    confusion: List[List[int]] = field(default_factory=list)
    cohen_kappa: float = 0.0

    @property
    def macro_precision(self) -> float:
        """Macro-averaged precision."""
        if not self.per_class:
            return 0.0
        return sum(p[0] for p in self.per_class.values()) / len(self.per_class)

    @property
    def macro_recall(self) -> float:
        """Macro-averaged recall."""
        if not self.per_class:
            return 0.0
        return sum(p[1] for p in self.per_class.values()) / len(self.per_class)

    @property
    def macro_f1(self) -> float:
        """Macro-averaged F1."""
        if not self.per_class:
            return 0.0
        return sum(p[2] for p in self.per_class.values()) / len(self.per_class)

    @property
    def accuracy(self) -> float:
        """Overall accuracy from confusion matrix."""
        if not self.confusion:
            return 0.0
        total = sum(sum(row) for row in self.confusion)
        correct = sum(self.confusion[i][i] for i in range(len(self.confusion)))
        return correct / total if total else 0.0


def compute_multiclass_metrics(
    predictions: Sequence[str],
    ground_truths: Sequence[str],
    labels: Optional[Sequence[str]] = None,
) -> MultiClassMetrics:
    """Compute per-class P/R/F1, confusion matrix, and Cohen's Kappa.

    Args:
        labels: Optional fixed label ordering. If None, sorted unique labels.

    Raises:
        ValueError: If lengths mismatch or lists are empty.
    """
    if len(predictions) != len(ground_truths):
        raise ValueError("Length mismatch between predictions and ground truths.")
    if not predictions:
        raise ValueError("Cannot compute metrics on empty sequences.")

    if labels is None:
        labels = sorted(set(predictions) | set(ground_truths))
    label_list = list(labels)
    label_idx = {lbl: i for i, lbl in enumerate(label_list)}
    n_classes = len(label_list)

    confusion = [[0] * n_classes for _ in range(n_classes)]
    for pred, gt in zip(predictions, ground_truths):
        i = label_idx.get(gt, -1)
        j = label_idx.get(pred, -1)
        if i >= 0 and j >= 0:
            confusion[i][j] += 1

    per_class: Dict[str, Tuple[float, float, float]] = {}
    for c, lbl in enumerate(label_list):
        tp = confusion[c][c]
        fp = sum(confusion[r][c] for r in range(n_classes)) - tp
        fn = sum(confusion[c][r] for r in range(n_classes)) - tp
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        denom = precision + recall
        f1 = 2 * precision * recall / denom if denom else 0.0
        per_class[lbl] = (precision, recall, f1)

    kappa = cohen_kappa(predictions, ground_truths, label_list)

    return MultiClassMetrics(
        labels=label_list,
        per_class=per_class,
        confusion=confusion,
        cohen_kappa=kappa,
    )


def cohen_kappa(
    predictions: Sequence[str],
    ground_truths: Sequence[str],
    labels: Sequence[str],
) -> float:
    """Compute Cohen's Kappa agreement over ``labels``; value in [-1, 1]."""
    n = len(predictions)
    if n == 0:
        return 0.0

    label_idx = {lbl: i for i, lbl in enumerate(labels)}
    n_classes = len(labels)

    confusion = [[0] * n_classes for _ in range(n_classes)]
    for pred, gt in zip(predictions, ground_truths):
        i = label_idx.get(gt)
        j = label_idx.get(pred)
        if i is not None and j is not None:
            confusion[i][j] += 1

    po = sum(confusion[i][i] for i in range(n_classes)) / n

    row_sums = [sum(confusion[i]) / n for i in range(n_classes)]
    col_sums = [sum(confusion[r][c] for r in range(n_classes)) / n for c in range(n_classes)]
    pe = sum(row_sums[i] * col_sums[i] for i in range(n_classes))

    denom = 1.0 - pe
    return (po - pe) / denom if denom else 0.0


# ---------------------------------------------------------------------------
# End-to-end metrics (Exp 4)
# ---------------------------------------------------------------------------


def faithfulness(
    sentence_labels: Sequence[str],
) -> float:
    """Compute Faithfulness — proportion of supported sentences.

    Maps 4-label Subset D annotations to S_supported / S_verifiable, where
    S_verifiable excludes 'no_source_needed'. Accepts the annotation labels
    ('supported', 'partially_supported', 'not_supported', 'no_source_needed')
    and the NLI labels ('entailment', 'neutral', 'contradiction'). Returns a
    score in [0, 1].
    """
    supported_map = {
        "supported": True,
        "entailment": True,
        "not_supported": False,
        "contradiction": False,
        "partially_supported": False,
        "neutral": False,
        "no_source_needed": None,
    }

    s_supported = 0
    s_verifiable = 0
    for label in sentence_labels:
        key = label.strip().lower()
        status = supported_map.get(key)
        if status is None:
            continue  # no_source_needed — excluded
        s_verifiable += 1
        if status:
            s_supported += 1

    return s_supported / s_verifiable if s_verifiable else 0.0


def abstention_accuracy(
    abstained: Sequence[bool],
    out_of_domain_total: int,
) -> float:
    """Compute Abstention Accuracy.

    ``abstained`` marks, per query, whether the system correctly refused/warned;
    returns the share of the ``out_of_domain_total`` queries correctly
    abstained.
    """
    if out_of_domain_total == 0:
        return 0.0
    return sum(1 for a in abstained if a) / out_of_domain_total


def token_jaccard_similarity(text_a: str, text_b: str) -> float:
    """Compute token-level Jaccard similarity between two strings, in [0, 1].

    Not used as Experiment 3's baseline — see ``token_containment_similarity``.
    Kept for other callers/tests.
    """
    tokens_a = set(text_a.lower().split())
    tokens_b = set(text_b.lower().split())
    if not tokens_a and not tokens_b:
        return 1.0
    union = tokens_a | tokens_b
    if not union:
        return 0.0
    return len(tokens_a & tokens_b) / len(union)


def token_containment_similarity(text_a: str, text_b: str) -> float:
    """Compute token-level containment of ``text_a`` in ``text_b``.

    ``|tokens(text_a) ∩ tokens(text_b)| / |tokens(text_a)|`` — unlike
    Jaccard, the denominator is only ``text_a``'s own vocabulary, not the
    union with ``text_b``. This matters for Experiment 3's baseline, where
    ``text_a`` is a ~20-token sentence and ``text_b`` is a 4,000-22,000 token
    retrieved context: with Jaccard the union term is dominated by the huge
    premise almost regardless of overlap, so similarity values for this pairing
    are compressed into roughly [0, 0.1] — below any reachable threshold,
    degenerating the Jaccard baseline into a constant classifier. Containment
    instead asks how much of the sentence's own vocabulary appears in the
    context, which stays meaningful regardless of the size mismatch.

    ``text_a`` is the shorter text whose coverage is measured (the
    sentence/hypothesis); ``text_b`` is the reference it's checked against (the
    retrieved context/premise). Returns 1.0 if ``text_a`` is empty (vacuously
    contained), matching ``token_jaccard_similarity``'s empty/empty convention.
    """
    tokens_a = set(text_a.lower().split())
    if not tokens_a:
        return 1.0
    tokens_b = set(text_b.lower().split())
    if not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a)


def bert_score_f1(
    candidates: Sequence[str],
    references: Sequence[str],
    model_type: str = "indobenchmark/indobert-base-p1",
) -> Tuple[float, float, float, List[float]]:
    """Compute BERTScore F1 between candidate and reference texts.

    Uses the ``bert-score`` Python package. ``model_type`` defaults to an
    IndoBERT checkpoint that isn't in bert-score's built-in ``model2layers``
    registry (the checkpoints it has official per-layer calibration for) —
    calling ``score()`` with an unlisted model_type and no explicit
    ``num_layers`` raises a bare ``KeyError`` inside the package rather than
    falling back, so this looks the layer count up from the model's own config
    when it's not in that registry.

    Returns (precision, recall, f1, f1_per_example): the first three are corpus
    means; ``f1_per_example`` is the per-candidate F1 list (same order as
    ``candidates``), so callers needing a bootstrap CI can resample this array
    directly instead of re-invoking the model per resample (see
    exp4_end_to_end/run.py's ``compute_e2e_metrics``). Returns
    ``(0.0, 0.0, 0.0, [])`` only if the ``bert-score`` package isn't installed
    (``pip install bert-score``) — a zero here means the environment is missing
    the dependency, not that the generated text scored zero.
    """
    if len(candidates) != len(references):
        raise ValueError("Length mismatch between candidates and references.")
    if not candidates:
        return 0.0, 0.0, 0.0, []

    try:
        import bert_score.utils as bs_utils  # type: ignore[import-untyped]
        from bert_score import score as bs_score  # type: ignore[import-untyped]
        from transformers import AutoConfig, AutoTokenizer

        config = AutoConfig.from_pretrained(model_type)
        num_layers = bs_utils.model2layers.get(model_type, config.num_hidden_layers)

        # indobert-base-p1's tokenizer_config.json never sets model_max_length,
        # so HF defaults it to a ~10^30 sentinel. bert-score's own
        # get_tokenizer() loads the tokenizer fresh via
        # transformers.AutoTokenizer with no override, and the current
        # transformers Rust tokenizer backend can't convert that sentinel to a C
        # int when truncation is enabled, crashing with "OverflowError: int too
        # big to convert". Patch AutoTokenizer.from_pretrained *inside
        # bert_score.utils's own namespace* (the only place it's called from)
        # for the duration of this call, clamping model_max_length to this
        # model's real position-embedding limit — bert-score exposes no hook to
        # pass a pre-configured tokenizer in.
        max_position = getattr(config, "max_position_embeddings", 512)
        _orig_from_pretrained = AutoTokenizer.from_pretrained

        def _patched_from_pretrained(*args, **kwargs):
            tok = _orig_from_pretrained(*args, **kwargs)
            if tok.model_max_length > max_position:
                tok.model_max_length = max_position
            return tok

        bs_utils.AutoTokenizer.from_pretrained = _patched_from_pretrained
        try:
            results = bs_score(
                list(candidates),
                list(references),
                model_type=model_type,
                num_layers=num_layers,
                lang="id",
                verbose=False,
            )
        finally:
            bs_utils.AutoTokenizer.from_pretrained = _orig_from_pretrained

        p = results[0].mean().item()
        r = results[1].mean().item()
        f1_per_example = results[2].tolist()
        f1 = results[2].mean().item()
        return p, r, f1, f1_per_example
    except ImportError:
        return 0.0, 0.0, 0.0, []


# ---------------------------------------------------------------------------
# Confidence intervals
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CI:
    """Confidence interval for a point estimate.

    ``method`` is 'bootstrap' or 'wilson'.
    """

    point: float
    lower: float
    upper: float
    method: str = "bootstrap"


def bootstrap_ci(
    values: Sequence[float],
    n_resamples: int = 1000,
    confidence: float = 0.95,
    statistic: str = "mean",
    seed: Optional[int] = 42,
) -> CI:
    """Compute bootstrap confidence interval for a statistic.

    Resamples with replacement ``n_resamples`` times and computes the
    statistic ('mean' or 'median') on each resample; returns the percentile
    interval.
    """
    if not values:
        return CI(0.0, 0.0, 0.0)

    rng = random.Random(seed)
    n = len(values)
    values_list = list(values)

    def _stat(sample: List[float]) -> float:
        if statistic == "median":
            s = sorted(sample)
            mid = n // 2
            return s[mid] if n % 2 == 1 else (s[mid - 1] + s[mid]) / 2
        return sum(sample) / len(sample) if sample else 0.0

    point = _stat(values_list)
    boot_stats: List[float] = []
    for _ in range(n_resamples):
        sample = [values_list[rng.randrange(n)] for _ in range(n)]
        boot_stats.append(_stat(sample))

    boot_stats.sort()
    alpha = 1.0 - confidence
    lower_idx = int((alpha / 2) * n_resamples)
    upper_idx = int((1 - alpha / 2) * n_resamples)
    upper_idx = min(upper_idx, n_resamples - 1)

    return CI(
        point=point,
        lower=boot_stats[lower_idx],
        upper=boot_stats[upper_idx],
        method="bootstrap",
    )


def wilson_interval(
    successes: int,
    total: int,
    confidence: float = 0.95,
) -> CI:
    """Compute Wilson score interval for a binomial proportion."""
    if total == 0:
        return CI(0.0, 0.0, 0.0, method="wilson")

    p = successes / total
    z = 1.959963984540054  # z for 95% CI
    if confidence != 0.95:
        # Approximate z for other confidence levels
        alpha = 1.0 - confidence
        z = _z_from_alpha(alpha)

    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom

    return CI(
        point=p,
        lower=max(0.0, center - margin),
        upper=min(1.0, center + margin),
        method="wilson",
    )


def _z_from_alpha(alpha: float) -> float:
    """Approximate z-score from alpha (significance level, 1 - confidence)."""
    table = {0.10: 1.6449, 0.05: 1.9600, 0.01: 2.5758, 0.001: 3.2905}
    return table.get(round(alpha, 4), 1.96)


def bootstrap_binary_ci(
    predictions: Sequence[bool],
    ground_truths: Sequence[bool],
    metric: str = "accuracy",
    n_resamples: int = 1000,
    seed: Optional[int] = 42,
) -> CI:
    """Bootstrap CI for a binary classification metric.

    ``metric`` is one of 'accuracy', 'precision', 'recall', 'f1', 'fpr'.
    """
    if not predictions:
        return CI(0.0, 0.0, 0.0)

    rng = random.Random(seed)
    n = len(predictions)
    pred_list = list(predictions)
    gt_list = list(ground_truths)

    def _metric(preds: List[bool], gts: List[bool]) -> float:
        m = compute_binary_metrics(preds, gts)
        return {
            "accuracy": m.accuracy,
            "precision": m.precision,
            "recall": m.recall,
            "f1": m.f1,
            "fpr": m.fpr,
        }.get(metric, m.accuracy)

    point = _metric(pred_list, gt_list)
    boot_stats: List[float] = []
    for _ in range(n_resamples):
        indices = [rng.randrange(n) for _ in range(n)]
        preds = [pred_list[i] for i in indices]
        gts = [gt_list[i] for i in indices]
        boot_stats.append(_metric(preds, gts))

    boot_stats.sort()
    alpha = 0.05
    lower_idx = int((alpha / 2) * n_resamples)
    upper_idx = int((1 - alpha / 2) * n_resamples)
    upper_idx = min(upper_idx, n_resamples - 1)

    return CI(point=point, lower=boot_stats[lower_idx], upper=boot_stats[upper_idx])
