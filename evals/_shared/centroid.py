"""Corpus-centroid out-of-domain detector (Exp1b).

A cheap, LLM-free relevance signal: embed the query with the same model the
corpus was embedded with, and measure how far it sits from the corpus. An
in-domain query lands near the cloud of document embeddings; an out-of-domain
one lands outside it. Two distances are offered:

- **cosine-to-centroid** (primary): cosine similarity between the query and the
  mean corpus vector. Robust, parameter-free, and the natural metric because
  the store indexes bge-m3 vectors under cosine distance.
- **Mahalanobis** (optional): accounts for the corpus's covariance, so a query
  that is off-centre along a low-variance direction is penalised more than one
  off-centre along a high-variance direction. bge-m3 is 1024-dimensional, so the
  raw sample covariance is ill-conditioned; a shrinkage estimator toward a
  scaled identity keeps the inverse stable.

The numeric core here is pure (NumPy in, floats out) so it is unit-testable on
synthetic clouds without a running Qdrant or embedder. The harness supplies the
corpus vectors (scrolled from Qdrant) and the query vectors (from Infinity).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

COSINE = "cosine"
MAHALANOBIS = "mahalanobis"


@dataclass
class CentroidModel:
    """A fitted corpus centroid and (optionally) a whitening matrix.

    ``centroid`` is the mean corpus vector, shape (d,); ``precision`` is the
    inverse shrinkage-covariance for Mahalanobis, shape (d, d), or None when
    only cosine scoring is needed; ``n`` is the number of corpus vectors fitted.
    """

    centroid: np.ndarray
    precision: Optional[np.ndarray]
    n: int


def fit_centroid(
    vectors: np.ndarray,
    with_mahalanobis: bool = True,
    shrinkage: float = 0.1,
) -> CentroidModel:
    """Fit the corpus centroid and, optionally, a shrinkage precision matrix.

    Args:
        vectors: Corpus embeddings, shape (n, d).
        with_mahalanobis: Whether to compute the precision matrix.
        shrinkage: Weight (0..1) toward the scaled-identity target when
            regularising the covariance. Higher = more regularised. bge-m3's
            covariance is rank-deficient unless n >> d, so a non-zero value is
            required for the inverse to exist.

    Raises:
        ValueError: If ``vectors`` is empty.
    """
    if vectors.ndim != 2 or vectors.shape[0] == 0:
        raise ValueError("fit_centroid needs a non-empty (n, d) array")

    centroid = vectors.mean(axis=0)
    precision: Optional[np.ndarray] = None

    if with_mahalanobis:
        d = vectors.shape[1]
        cov = np.cov(vectors, rowvar=False)
        cov = np.atleast_2d(cov)
        # Ledoit-Wolf-style diagonal-target shrinkage: pull the sample
        # covariance toward a scaled identity so the inverse is well-defined
        # even when n < d.
        target = (np.trace(cov) / d) * np.eye(d)
        regularised = (1.0 - shrinkage) * cov + shrinkage * target
        precision = np.linalg.pinv(regularised)

    return CentroidModel(centroid=centroid, precision=precision, n=int(vectors.shape[0]))


def score(model: CentroidModel, query: np.ndarray, metric: str = COSINE) -> float:
    """Score one query embedding (shape (d,)) against the corpus centre.

    ``metric`` is ``COSINE`` (higher = more in-domain) or ``MAHALANOBIS``
    (lower = more in-domain).

    Raises:
        ValueError: For an unknown metric, or Mahalanobis without a precision
            matrix.
    """
    query = np.asarray(query, dtype=float)
    if metric == COSINE:
        denom = np.linalg.norm(query) * np.linalg.norm(model.centroid)
        if denom == 0:
            return 0.0
        return float(np.dot(query, model.centroid) / denom)
    if metric == MAHALANOBIS:
        if model.precision is None:
            raise ValueError("model was fitted without a precision matrix")
        delta = query - model.centroid
        return float(np.sqrt(max(delta @ model.precision @ delta, 0.0)))
    raise ValueError(f"unknown metric: {metric!r}")


def is_in_domain(value: float, threshold: float, metric: str = COSINE) -> bool:
    """Turn a score from :func:`score` into an in-domain decision.

    ``metric`` sets the comparison direction — cosine is in-domain **at or
    above** the threshold; Mahalanobis is in-domain **at or below** it.
    """
    if metric == COSINE:
        return value >= threshold
    if metric == MAHALANOBIS:
        return value <= threshold
    raise ValueError(f"unknown metric: {metric!r}")
