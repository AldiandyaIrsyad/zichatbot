"""Tests for evidence-window selection inside a cited chunk.

Citation targeting is the invariant: only chunks the LLM cited are consulted.
These tests cover the layer below it — picking which *span* of that chunk
becomes the NLI premise, instead of feeding the whole parent page (1200+
tokens, far outside the NLI model's training distribution).
"""
from __future__ import annotations

from typing import List, Optional
from unittest.mock import AsyncMock

import pytest

from app.guardrails.ram.interfaces import NLIResult, RerankResult, RetrievedContext
from app.guardrails.ram.service import RAMService

LONG_TEXT = " ".join(f"Kalimat nomor {i} berisi ketentuan tentang tarif UKT." for i in range(1, 13))

TABLE_TEXT = (
    "| Kode | Kemampuan Ekonomi | Kelompok |\n"
    "| --- | --- | --- |\n"
    "| a. | Penghasilan <= Rp500.000 | Kelompok I |\n"
    "| b. | Penghasilan Rp500.001-Rp1.000.000 | Kelompok II |\n"
    "| c. | Penghasilan Rp1.000.001-Rp2.000.000 | Kelompok III |\n"
    "| d. | Penghasilan > Rp9.000.001 | Kelompok VIII |\n"
)


def _result(label: str, entailment: float = 0.0, contradiction: float = 0.0) -> NLIResult:
    return NLIResult(
        label=label,
        entailment_score=entailment,
        neutral_score=max(0.0, 1.0 - entailment - contradiction),
        contradiction_score=contradiction,
    )


class _Reranker:
    """Returns the candidate whose text contains ``needle`` first."""

    def __init__(self, needle: str) -> None:
        self.needle = needle
        self.seen: List[str] = []

    async def rerank(
        self, query: str, documents: List[str], top_k: Optional[int] = None
    ) -> List[RerankResult]:
        self.seen = documents
        order = sorted(
            range(len(documents)),
            key=lambda i: (self.needle not in documents[i], i),
        )
        return [RerankResult(index=i, score=1.0) for i in order[: top_k or len(documents)]]


class TestPremiseWindows:
    def test_child_text_leads(self) -> None:
        ctx = RetrievedContext(text=LONG_TEXT, source_title="Doc", child_text="Kalimat penting.")
        assert RAMService._premise_windows(ctx)[0] == "Kalimat penting."

    def test_long_prose_is_windowed_not_returned_whole(self) -> None:
        ctx = RetrievedContext(text=LONG_TEXT, source_title="Doc")
        windows = RAMService._premise_windows(ctx)
        assert len(windows) > 1
        assert all(len(w) < len(LONG_TEXT) for w in windows)

    def test_table_windows_keep_the_header(self) -> None:
        ctx = RetrievedContext(text=TABLE_TEXT, source_title="Doc", content_type="table")
        windows = RAMService._premise_windows(ctx)
        assert len(windows) > 1
        assert all(w.startswith("| Kode | Kemampuan Ekonomi | Kelompok |") for w in windows)

    def test_windows_are_deduplicated(self) -> None:
        ctx = RetrievedContext(text="Satu kalimat saja.", source_title="Doc",
                               child_text="Satu kalimat saja.")
        assert RAMService._premise_windows(ctx) == ["Satu kalimat saja."]

    def test_empty_chunk_yields_nothing(self) -> None:
        assert RAMService._premise_windows(RetrievedContext(text="", source_title="Doc")) == []


class TestSelectPremises:
    @pytest.mark.asyncio
    async def test_reranker_picks_the_matching_window(self) -> None:
        reranker = _Reranker("Kalimat nomor 9")
        service = RAMService(nli_model=AsyncMock(), reranker_model=reranker)
        ctx = RetrievedContext(text=LONG_TEXT, source_title="Doc")

        premises = await service._select_premises("Apa isi kalimat 9?", ctx)

        assert "Kalimat nomor 9" in premises[0]
        assert len(premises) <= 2

    @pytest.mark.asyncio
    async def test_no_reranker_falls_back_to_leading_windows(self) -> None:
        service = RAMService(nli_model=AsyncMock(), reranker_model=None)
        ctx = RetrievedContext(text=LONG_TEXT, source_title="Doc", child_text="Anchor.")

        premises = await service._select_premises("klaim", ctx)

        assert premises[0] == "Anchor."
        assert len(premises) <= 2

    @pytest.mark.asyncio
    async def test_reranker_failure_degrades_instead_of_raising(self) -> None:
        failing = AsyncMock()
        failing.rerank = AsyncMock(side_effect=RuntimeError("reranker down"))
        service = RAMService(nli_model=AsyncMock(), reranker_model=failing)
        ctx = RetrievedContext(text=LONG_TEXT, source_title="Doc")

        premises = await service._select_premises("klaim", ctx)

        assert premises  # still usable
        assert len(premises) <= 2


class TestCheckWindows:
    @pytest.mark.asyncio
    async def test_confident_entailment_short_circuits(self) -> None:
        nli = AsyncMock()
        nli.check = AsyncMock(side_effect=[_result("entailment", 0.9), _result("neutral")])
        service = RAMService(nli_model=nli)

        result, premise = await service._check_windows("klaim", ["window A", "window B"])

        assert result.label == "entailment"
        assert premise == "window A"
        assert nli.check.await_count == 1  # second window never spent

    @pytest.mark.asyncio
    async def test_entailment_beats_an_earlier_contradiction(self) -> None:
        # A window carrying an exception clause must not pre-empt the rule it
        # qualifies in a lower-ranked window.
        nli = AsyncMock()
        nli.check = AsyncMock(side_effect=[
            _result("contradiction", contradiction=0.95),
            _result("entailment", 0.88),
        ])
        service = RAMService(nli_model=nli)

        result, premise = await service._check_windows("klaim", ["exception", "rule"])

        assert result.label == "entailment"
        assert premise == "rule"

    @pytest.mark.asyncio
    async def test_falls_back_to_first_result_when_nothing_is_confident(self) -> None:
        nli = AsyncMock()
        nli.check = AsyncMock(side_effect=[_result("neutral", 0.3), _result("neutral", 0.1)])
        service = RAMService(nli_model=nli)

        result, premise = await service._check_windows("klaim", ["a", "b"])

        assert result.label == "neutral"
        assert premise == "a"

    @pytest.mark.asyncio
    async def test_all_calls_failing_returns_none(self) -> None:
        nli = AsyncMock()
        nli.check = AsyncMock(side_effect=RuntimeError("nli down"))
        service = RAMService(nli_model=nli)

        assert await service._check_windows("klaim", ["a", "b"]) is None


class TestAssessClaimStillCitationScoped:
    @pytest.mark.asyncio
    async def test_only_cited_chunks_are_consulted(self) -> None:
        nli = AsyncMock()
        nli.check = AsyncMock(return_value=_result("entailment", 0.9))
        service = RAMService(nli_model=nli)
        cited = RetrievedContext(text="Isi yang dikutip.", source_title="Cited")
        other = RetrievedContext(text="Isi yang tidak dikutip.", source_title="Uncited")

        result = await service.assess_claim("klaim", [cited, other], (1,))

        assert result.source_title == "Cited"
        premises = [c.kwargs["premise"] for c in nli.check.await_args_list]
        assert all("tidak dikutip" not in p for p in premises)

    @pytest.mark.asyncio
    async def test_evidence_snippet_is_the_window_that_was_checked(self) -> None:
        nli = AsyncMock()
        nli.check = AsyncMock(return_value=_result("entailment", 0.9))
        service = RAMService(nli_model=nli, reranker_model=_Reranker("Kalimat nomor 9"))
        ctx = RetrievedContext(text=LONG_TEXT, source_title="Doc")

        result = await service.assess_claim("Apa isi kalimat 9?", [ctx], (1,))

        assert "Kalimat nomor 9" in result.evidence_snippet
        assert len(result.evidence_snippet) < len(LONG_TEXT)
