"""Tests for the Experiment 4 harness additions.

Covers the pieces added so Exp4 speaks to RQ4 ("performa keseluruhan ...
berguardrail ganda") rather than only to "guardrails on vs off":

    - latency aggregation (the cost side of the trade-off), which must ignore
      errored rows so a timeout's 120s doesn't become the reported number;
    - the per-category breakdown, since Subset A labels questions
      factual/procedural/multi-hop but only the aggregate was ever reported;
    - the guardrail condition switches reaching the API as query params.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from evals._shared.dataset import SubsetARow
from evals.exp4_end_to_end.run import (
    PipelineResult,
    compute_e2e_metrics,
    compute_per_category,
    run_pipeline,
    stratified_sample,
)


def _result(**kwargs: Any) -> PipelineResult:
    defaults: Dict[str, Any] = {
        "question": "Q?",
        "response": "Jawaban lengkap tentang Statuta UPI.",
        "category": "factual",
        "ground_truth": "Jawaban acuan.",
        "latency_s": 1.0,
    }
    defaults.update(kwargs)
    return PipelineResult(**defaults)


class TestLatency:
    def test_errored_rows_excluded(self) -> None:
        """A timeout says nothing about how long the pipeline takes when it works."""
        results = [
            _result(question="a", latency_s=1.0),
            _result(question="b", latency_s=3.0),
            _result(question="c", latency_s=120.0, errored=True, response=""),
        ]
        metrics, _ = compute_e2e_metrics(results)
        assert metrics.latency_mean_s == pytest.approx(2.0)

    def test_percentiles_reported(self) -> None:
        results = [_result(question=str(i), latency_s=float(i)) for i in range(1, 11)]
        metrics, _ = compute_e2e_metrics(results)
        assert metrics.latency_p50_s == pytest.approx(6.0)
        assert metrics.latency_p95_s >= metrics.latency_p50_s

    def test_no_latencies_leaves_zero(self) -> None:
        metrics, _ = compute_e2e_metrics([_result(latency_s=0.0)])
        assert metrics.latency_mean_s == 0.0


class TestPerCategory:
    def test_groups_in_domain_rows_by_category(self) -> None:
        results = [
            _result(question="f1", category="factual"),
            _result(question="f2", category="factual"),
            _result(question="m1", category="multi-hop"),
            _result(question="o1", category="out-of-domain"),
        ]
        per_row = {
            "f1": {"bertscore_f1": 0.4, "faithfulness_score": 0.8},
            "f2": {"bertscore_f1": 0.6, "faithfulness_score": 0.6},
            "m1": {"bertscore_f1": 0.2, "faithfulness_score": 0.4},
        }
        out = compute_per_category(results, per_row)

        assert set(out) == {"factual", "multi-hop"}, "out-of-domain must be excluded"
        assert out["factual"]["n"] == 2
        assert out["factual"]["bertscore_f1"] == pytest.approx(0.5)
        assert out["factual"]["faithfulness_score"] == pytest.approx(0.7)
        assert out["multi-hop"]["n"] == 1

    def test_missing_scores_yield_none_not_zero(self) -> None:
        """A category with no scored rows must read as absent, not as 0.0."""
        results = [_result(question="f1", category="factual")]
        out = compute_per_category(results, {"f1": {}})
        assert out["factual"]["bertscore_f1"] is None
        assert out["factual"]["faithfulness_score"] is None


class TestConditionSwitches:
    """The switches must reach the API, or the ablation silently measures one cell."""

    @staticmethod
    async def _capture_url(**switches: Any) -> str:
        captured: List[str] = []

        class FakeResponse:
            status_code = 200
            headers = {"content-type": "application/x-ndjson"}
            text = '{"type": "chunk", "content": "Jawaban."}\n{"type": "done"}'

            def json(self) -> Dict[str, Any]:
                return {}

        class FakeClient:
            async def __aenter__(self) -> "FakeClient":
                return self

            async def __aexit__(self, *args: Any) -> None:
                return None

            async def post(self, url: str, **kwargs: Any) -> FakeResponse:
                captured.append(url)
                return FakeResponse()

        row = SubsetARow(
            question="Apa isi Statuta UPI?",
            category="factual",
            ground_truth_answer="Jawaban.",
            source_doc_id="doc-1",
            source_context="Konteks.",
        )
        with patch("evals.exp4_end_to_end.run.httpx.AsyncClient", lambda **kw: FakeClient()):
            await run_pipeline("http://localhost:8000", row, session_id="s1", **switches)
        return captured[-1]

    @pytest.mark.asyncio
    async def test_full_condition_sends_no_skip_params(self) -> None:
        url = await self._capture_url(skip_ivm=False, skip_ram=False)
        assert "skip_ivm=false" in url
        assert "skip_ram=false" in url

    @pytest.mark.asyncio
    async def test_ivm_only_skips_ram(self) -> None:
        url = await self._capture_url(skip_ivm=False, skip_ram=True)
        assert "skip_ivm=false" in url
        assert "skip_ram=true" in url

    @pytest.mark.asyncio
    async def test_ram_only_skips_ivm(self) -> None:
        url = await self._capture_url(skip_ivm=True, skip_ram=False)
        assert "skip_ivm=true" in url
        assert "skip_ram=false" in url

    @pytest.mark.asyncio
    async def test_legacy_skip_guardrails_still_works(self) -> None:
        url = await self._capture_url(skip_guardrails=True)
        assert "skip_guardrails=true" in url

    @pytest.mark.asyncio
    async def test_latency_recorded(self) -> None:
        row = SubsetARow("Q?", "factual", "A", "doc-1", "C")

        class FakeResponse:
            status_code = 200
            headers = {"content-type": "application/x-ndjson"}
            text = '{"type": "chunk", "content": "Jawaban."}\n{"type": "done"}'

            def json(self) -> Dict[str, Any]:
                return {}

        class FakeClient:
            async def __aenter__(self) -> "FakeClient":
                return self

            async def __aexit__(self, *args: Any) -> None:
                return None

            async def post(self, url: str, **kwargs: Any) -> FakeResponse:
                return FakeResponse()

        with patch("evals.exp4_end_to_end.run.httpx.AsyncClient", lambda **kw: FakeClient()):
            result = await run_pipeline("http://localhost:8000", row, session_id="s1")

        assert result.latency_s > 0.0


class TestStratifiedSample:
    """The lean RQ4 --limit run must keep every category, deterministically."""

    @staticmethod
    def _rows() -> List[SubsetARow]:
        # 10 factual, 6 procedural, 3 multi-hop, 2 out-of-domain = 21 rows.
        rows: List[SubsetARow] = []
        for cat, n in [("factual", 10), ("procedural", 6), ("multi-hop", 3), ("out-of-domain", 2)]:
            for i in range(n):
                rows.append(SubsetARow(f"{cat}-{i}", cat, "A", "doc-1", "C"))
        return rows

    def test_limit_zero_or_larger_returns_full(self) -> None:
        rows = self._rows()
        assert stratified_sample(rows, 0, 42) == rows
        assert stratified_sample(rows, len(rows) + 5, 42) == rows

    def test_respects_limit_and_keeps_every_category(self) -> None:
        rows = self._rows()
        sample = stratified_sample(rows, 8, 42)
        assert len(sample) == 8
        # out-of-domain has only 2 rows but must not be starved — the
        # abstention metric depends on it being present.
        cats = {r.category for r in sample}
        assert cats == {"factual", "procedural", "multi-hop", "out-of-domain"}

    def test_even_spread_when_divisible(self) -> None:
        rows = self._rows()
        sample = stratified_sample(rows, 4, 42)
        counts = Counter(r.category for r in sample)
        # one per category — round-robin, not head-N (which would be all factual)
        assert counts == Counter(
            {"factual": 1, "procedural": 1, "multi-hop": 1, "out-of-domain": 1}
        )

    def test_deterministic_for_a_seed(self) -> None:
        rows = self._rows()
        a = [r.question for r in stratified_sample(rows, 9, 7)]
        b = [r.question for r in stratified_sample(rows, 9, 7)]
        assert a == b

    def test_no_duplicate_rows(self) -> None:
        rows = self._rows()
        sample = stratified_sample(rows, 15, 42)
        assert len({r.question for r in sample}) == len(sample)
