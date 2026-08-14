"""Unit tests for SimilarityThresholdRelevanceChecker and
NliEntailmentRelevanceChecker (app/thesis/ivm/checkers.py)."""
from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from app.guardrails.ram.interfaces import NLIResult
from app.guardrails.ivm.checkers import (
    NliEntailmentRelevanceChecker,
    SimilarityThresholdRelevanceChecker,
)


class TestSimilarityThresholdRelevanceChecker:
    @pytest.mark.asyncio
    async def test_accepts_when_top_score_meets_threshold(self) -> None:
        checker = SimilarityThresholdRelevanceChecker(threshold=0.02)
        result = await checker.check_query("q", ["ctx"], [0.01, 0.03])
        assert result is True

    @pytest.mark.asyncio
    async def test_rejects_when_top_score_below_threshold(self) -> None:
        checker = SimilarityThresholdRelevanceChecker(threshold=0.05)
        result = await checker.check_query("q", ["ctx"], [0.01, 0.03])
        assert result is False

    @pytest.mark.asyncio
    async def test_fails_closed_when_no_scores_given(self) -> None:
        checker = SimilarityThresholdRelevanceChecker(threshold=0.02)
        result = await checker.check_query("q", ["ctx"], [])
        assert result is False

    @pytest.mark.asyncio
    async def test_uses_max_not_first_score(self) -> None:
        # Order shouldn't matter — must consider the max, not scores[0].
        checker = SimilarityThresholdRelevanceChecker(threshold=0.02)
        result = await checker.check_query("q", ["c1", "c2"], [0.001, 0.05])
        assert result is True


class TestNliEntailmentRelevanceChecker:
    @pytest.mark.asyncio
    async def test_accepts_when_entailment_score_meets_threshold(self) -> None:
        nli = AsyncMock()
        nli.check = AsyncMock(
            return_value=NLIResult(label="entailment", entailment_score=0.8)
        )
        checker = NliEntailmentRelevanceChecker(nli_model=nli, threshold=0.5)

        result = await checker.check_query("q", ["ctx"], [])

        assert result is True
        nli.check.assert_awaited_once_with(premise="ctx", hypothesis="q")

    @pytest.mark.asyncio
    async def test_rejects_low_confidence_neutral_even_without_contradiction(self) -> None:
        # Regression case motivating the threshold-not-label design decision.
        nli = AsyncMock()
        nli.check = AsyncMock(
            return_value=NLIResult(label="neutral", entailment_score=0.2, neutral_score=0.7)
        )
        checker = NliEntailmentRelevanceChecker(nli_model=nli, threshold=0.5)

        result = await checker.check_query("q", ["ctx"], [])

        assert result is False

    @pytest.mark.asyncio
    async def test_joins_multiple_context_chunks_as_single_premise(self) -> None:
        nli = AsyncMock()
        nli.check = AsyncMock(return_value=NLIResult(label="entailment", entailment_score=0.9))
        checker = NliEntailmentRelevanceChecker(nli_model=nli, threshold=0.5)

        await checker.check_query("q", ["ctx1", "ctx2"], [])

        premise_used = nli.check.await_args.kwargs["premise"]
        assert premise_used == "ctx1\nctx2"

    @pytest.mark.asyncio
    async def test_fails_closed_when_no_context_chunks(self) -> None:
        nli = AsyncMock()
        checker = NliEntailmentRelevanceChecker(nli_model=nli, threshold=0.5)

        result = await checker.check_query("q", [], [])

        assert result is False
        nli.check.assert_not_awaited()
