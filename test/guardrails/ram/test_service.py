"""Tests for RAMService's evidence-snippet sanitization and citation-local
claim assessment."""
from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from app.guardrails.ram.interfaces import NLIResult, RetrievedContext
from app.guardrails.ram.service import (
    LABEL_CONTRADICTION,
    LABEL_ENTAILMENT,
    LABEL_NEUTRAL,
    EVIDENCE_SNIPPET_MAX_CHARS,
    RAMService,
    _sanitize_snippet,
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


def _make_text_context(text: str = "Isi peraturan terkait.") -> RetrievedContext:
    return RetrievedContext(text=text, source_title="Doc A", page=1, doc_id="doc-1")


class TestAssessClaim:
    @pytest.mark.asyncio
    async def test_uses_cited_chunk_as_premise(self) -> None:
        ctx = _make_text_context("Kalimat bukti yang mendukung.")
        nli = AsyncMock()
        nli.check = AsyncMock(
            return_value=NLIResult(
                label=LABEL_ENTAILMENT, entailment_score=0.95, contradiction_score=0.0
            )
        )
        service = RAMService(nli_model=nli)

        result = await service.assess_claim("Klaim yang diuji.", [ctx], (1,))

        assert nli.check.await_count == 1
        assert nli.check.await_args.kwargs["premise"] == "Kalimat bukti yang mendukung."
        assert nli.check.await_args.kwargs["hypothesis"] == "Klaim yang diuji."
        assert result.label == LABEL_ENTAILMENT
        assert result.source_title == "Doc A"
        assert result.page == 1
        assert result.doc_id == "doc-1"

    @pytest.mark.asyncio
    async def test_table_context_prefers_child_text(self) -> None:
        ctx = RetrievedContext(
            text="| Nama | Nilai |\n| --- | --- |\n| A | 1 |\n| B | 2 |",
            source_title="Doc A",
            content_type="table",
            child_text="| Nama | Nilai |\n| --- | --- |\n| A | 1 |",
        )
        nli = AsyncMock()
        nli.check = AsyncMock(
            return_value=NLIResult(
                label=LABEL_ENTAILMENT, entailment_score=0.8, contradiction_score=0.0
            )
        )
        service = RAMService(nli_model=nli)

        await service.assess_claim("A bernilai 1.", [ctx], (1,))

        assert nli.check.await_args.kwargs["premise"] == ctx.child_text

    @pytest.mark.asyncio
    async def test_multiple_citations_prefer_highest_entailment(self) -> None:
        ctx_a = _make_text_context("Bukti A.")
        ctx_b = _make_text_context("Bukti B yang lebih mendukung.")
        nli = AsyncMock()
        nli.check = AsyncMock(
            side_effect=[
                NLIResult(label=LABEL_NEUTRAL, entailment_score=0.3, contradiction_score=0.1),
                NLIResult(label=LABEL_ENTAILMENT, entailment_score=0.9, contradiction_score=0.0),
            ]
        )
        service = RAMService(nli_model=nli)

        result = await service.assess_claim("Klaim.", [ctx_a, ctx_b], (1, 2))

        assert nli.check.await_count == 2
        assert result.label == LABEL_ENTAILMENT
        assert result.entailment_score == 0.9

    @pytest.mark.asyncio
    async def test_contradiction_chosen_when_no_entailment(self) -> None:
        ctx = _make_text_context("Bukti yang bertentangan.")
        nli = AsyncMock()
        nli.check = AsyncMock(
            return_value=NLIResult(
                label=LABEL_CONTRADICTION, entailment_score=0.0, contradiction_score=0.85
            )
        )
        service = RAMService(nli_model=nli)

        result = await service.assess_claim("Klaim.", [ctx], (1,))

        assert result.label == LABEL_CONTRADICTION
        assert result.contradiction_score == 0.85

    @pytest.mark.asyncio
    async def test_out_of_range_citation_returns_neutral(self) -> None:
        ctx = _make_text_context()
        nli = AsyncMock()
        service = RAMService(nli_model=nli)

        result = await service.assess_claim("Klaim.", [ctx], (5,))

        assert nli.check.await_count == 0
        assert result.label == LABEL_NEUTRAL

    @pytest.mark.asyncio
    async def test_disabled_service_short_circuits(self) -> None:
        nli = AsyncMock()
        service = RAMService(nli_model=nli, enabled=False)

        result = await service.assess_claim("Klaim.", [_make_text_context()], (1,))

        assert nli.check.await_count == 0
        assert result.label == LABEL_NEUTRAL
