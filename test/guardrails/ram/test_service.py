"""Tests for RAMService's evidence-snippet sanitization helper and
per-sentence assessment logic."""
from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from app.guardrails.ram.interfaces import NLIResult, RerankResult, RetrievedContext
from app.guardrails.ram.service import (
    LABEL_CONTRADICTION,
    LABEL_ENTAILMENT,
    LABEL_NEUTRAL,
    NLI_CONTRADICTION_THRESHOLD,
    RAMService,
    _sanitize_snippet,
    EVIDENCE_SNIPPET_MAX_CHARS,
)


class TestSanitizeSnippet:
    def test_collapses_whitespace(self) -> None:
        assert _sanitize_snippet("Ada   banyak\nspasi.") == "Ada banyak spasi."

    def test_strips_characters_that_break_the_citation_marker_grammar(self) -> None:
        result = _sanitize_snippet('Isi (a); (b) * penting "kutipan"')
        assert ";" not in result
        assert ")" not in result
        assert "*" not in result
        assert '"' not in result

    def test_truncates_long_text_with_ellipsis(self) -> None:
        text = "kata " * 100
        result = _sanitize_snippet(text)
        assert len(result) <= EVIDENCE_SNIPPET_MAX_CHARS
        assert result.endswith("…")

    def test_short_text_is_unchanged_besides_whitespace(self) -> None:
        assert _sanitize_snippet("Teks singkat.") == "Teks singkat."


def _make_table_context(num_data_rows: int, content_type: str = "table") -> RetrievedContext:
    header = "| Nama | Nilai |"
    separator = "| --- | --- |"
    rows = [f"| Baris{i} | {i} |" for i in range(1, num_data_rows + 1)]
    text = "\n".join([header, separator] + rows)
    return RetrievedContext(text=text, source_title="Doc A", page=1, content_type=content_type)


class TestAssessSentence:
    @pytest.mark.asyncio
    async def test_table_context_window_carries_header_for_later_rows(self) -> None:
        # Regression test for the reported bug: the matching fact lives in
        # a later row-group (rows 4-6), which the old prose-based windowing
        # would have windowed without the header.
        ctx = _make_table_context(7)
        reranker = AsyncMock()
        reranker.rerank = AsyncMock(return_value=[RerankResult(index=0, score=0.9)])
        nli = AsyncMock()
        nli.check = AsyncMock(
            return_value=NLIResult(label=LABEL_ENTAILMENT, entailment_score=0.95, contradiction_score=0.0)
        )
        service = RAMService(nli_model=nli, reranker_model=reranker)

        await service.assess_sentence("Baris5 bernilai 5.", premise="", contexts=[ctx])

        assert nli.check.await_count >= 1
        premise_used = nli.check.await_args.kwargs["premise"]
        assert "| Nama | Nilai |" in premise_used

    @pytest.mark.asyncio
    async def test_confident_top1_short_circuits_after_one_nli_call(self) -> None:
        ctx = _make_table_context(5)  # 5 rows -> >1 window at rows_per_window=3/step=2
        reranker = AsyncMock()
        reranker.rerank = AsyncMock(
            return_value=[RerankResult(index=0, score=0.9), RerankResult(index=1, score=0.5)]
        )
        nli = AsyncMock()
        nli.check = AsyncMock(
            return_value=NLIResult(label=LABEL_ENTAILMENT, entailment_score=0.9, contradiction_score=0.0)
        )
        service = RAMService(nli_model=nli, reranker_model=reranker)

        result = await service.assess_sentence("Baris1 bernilai 1.", premise="", contexts=[ctx])

        assert nli.check.await_count == 1
        assert result.label == LABEL_ENTAILMENT

    @pytest.mark.asyncio
    async def test_neutral_top1_tries_second_candidate(self) -> None:
        ctx = _make_table_context(5)  # 5 rows -> >1 window at rows_per_window=3/step=2
        reranker = AsyncMock()
        reranker.rerank = AsyncMock(
            return_value=[RerankResult(index=0, score=0.9), RerankResult(index=1, score=0.5)]
        )
        nli = AsyncMock()
        nli.check = AsyncMock(
            side_effect=[
                NLIResult(label=LABEL_NEUTRAL, entailment_score=0.3, contradiction_score=0.1),
                NLIResult(label=LABEL_ENTAILMENT, entailment_score=0.85, contradiction_score=0.0),
            ]
        )
        service = RAMService(nli_model=nli, reranker_model=reranker)

        result = await service.assess_sentence("Baris1 bernilai 1.", premise="", contexts=[ctx])

        assert nli.check.await_count == 2
        assert result.label == LABEL_ENTAILMENT

    @pytest.mark.asyncio
    async def test_all_candidates_neutral_falls_back_to_first_instead_of_dropping(self) -> None:
        ctx = _make_table_context(5)  # 5 rows -> >1 window at rows_per_window=3/step=2
        reranker = AsyncMock()
        reranker.rerank = AsyncMock(
            return_value=[RerankResult(index=0, score=0.9), RerankResult(index=1, score=0.5)]
        )
        nli = AsyncMock()
        nli.check = AsyncMock(
            return_value=NLIResult(label=LABEL_NEUTRAL, entailment_score=0.4, contradiction_score=0.1)
        )
        service = RAMService(nli_model=nli, reranker_model=reranker)

        result = await service.assess_sentence("Baris1 bernilai 1.", premise="", contexts=[ctx])

        assert nli.check.await_count == 2
        assert result.label == LABEL_NEUTRAL
        # Fallback carries metadata from the first candidate, not a bare drop.
        assert result.source_title == "Doc A"

    @pytest.mark.asyncio
    async def test_confident_contradiction_does_not_preempt_later_entailment(self) -> None:
        # Regression test: a spurious confident contradiction from one
        # window (e.g. an exception clause) must not short-circuit before a
        # later, valid entailment from another window (e.g. the general
        # rule it qualifies) gets a chance to be checked.
        ctx = _make_table_context(5)  # 5 rows -> >1 window at rows_per_window=3/step=2
        reranker = AsyncMock()
        reranker.rerank = AsyncMock(
            return_value=[RerankResult(index=0, score=0.9), RerankResult(index=1, score=0.5)]
        )
        nli = AsyncMock()
        nli.check = AsyncMock(
            side_effect=[
                NLIResult(label=LABEL_CONTRADICTION, entailment_score=0.0, contradiction_score=0.9),
                NLIResult(label=LABEL_ENTAILMENT, entailment_score=0.85, contradiction_score=0.0),
            ]
        )
        service = RAMService(nli_model=nli, reranker_model=reranker)

        result = await service.assess_sentence("Baris1 bernilai 1.", premise="", contexts=[ctx])

        assert nli.check.await_count == 2
        assert result.label == LABEL_ENTAILMENT

    @pytest.mark.asyncio
    async def test_moderate_confidence_contradiction_below_raised_threshold_is_suppressed(self) -> None:
        ctx = _make_table_context(5)
        reranker = AsyncMock()
        reranker.rerank = AsyncMock(
            return_value=[RerankResult(index=0, score=0.9), RerankResult(index=1, score=0.5)]
        )
        moderate_score = NLI_CONTRADICTION_THRESHOLD - 0.1
        nli = AsyncMock()
        nli.check = AsyncMock(
            return_value=NLIResult(
                label=LABEL_CONTRADICTION, entailment_score=0.1, contradiction_score=moderate_score
            )
        )
        service = RAMService(nli_model=nli, reranker_model=reranker)

        result = await service.assess_sentence("Baris1 bernilai 1.", premise="", contexts=[ctx])

        # Below NLI_CONTRADICTION_THRESHOLD, so the raw "contradiction"
        # label from the model is downgraded to neutral rather than
        # surfaced as a citation badge (ChatService._format_citation
        # renders any "contradiction" label unconditionally, so filtering
        # must happen here, not at display time).
        assert result.label == LABEL_NEUTRAL

    @pytest.mark.asyncio
    async def test_high_confidence_contradiction_still_surfaced_when_no_entailment_found(self) -> None:
        ctx = _make_table_context(5)
        reranker = AsyncMock()
        reranker.rerank = AsyncMock(
            return_value=[RerankResult(index=0, score=0.9), RerankResult(index=1, score=0.5)]
        )
        nli = AsyncMock()
        nli.check = AsyncMock(
            side_effect=[
                NLIResult(
                    label=LABEL_CONTRADICTION,
                    entailment_score=0.0,
                    contradiction_score=NLI_CONTRADICTION_THRESHOLD + 0.1,
                ),
                NLIResult(label=LABEL_NEUTRAL, entailment_score=0.3, contradiction_score=0.2),
            ]
        )
        service = RAMService(nli_model=nli, reranker_model=reranker)

        result = await service.assess_sentence("Baris1 bernilai 1.", premise="", contexts=[ctx])

        assert nli.check.await_count == 2
        assert result.label == LABEL_CONTRADICTION
        assert result.contradiction_score == NLI_CONTRADICTION_THRESHOLD + 0.1

    @pytest.mark.asyncio
    async def test_non_markdown_table_content_degrades_to_prose_path(self) -> None:
        ctx = RetrievedContext(
            text="<table><tr><td>Isi tabel HTML mentah.</td></tr></table>",
            source_title="Doc B",
            content_type="table",
        )
        reranker = AsyncMock()
        reranker.rerank = AsyncMock(return_value=[RerankResult(index=0, score=0.8)])
        nli = AsyncMock()
        nli.check = AsyncMock(
            return_value=NLIResult(label=LABEL_ENTAILMENT, entailment_score=0.7, contradiction_score=0.0)
        )
        service = RAMService(nli_model=nli, reranker_model=reranker)

        result = await service.assess_sentence("Pernyataan apa saja.", premise="", contexts=[ctx])

        assert result.label == LABEL_ENTAILMENT
