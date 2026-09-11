"""Tests for the citation badge helper in ChatService.

Verifies that ``_format_citation()`` produces the three-score badge grammar::

    *(Supported:0.92; Neutral:0.03; Contradicted:0.05; SOURCE; Page N; DocID:ID; Evidence:"...")*

and that ``_format_unverified()`` produces ``*(Unverified)*``.
"""

from __future__ import annotations

from typing import Optional

from app.chat.application.chat_service import ChatService
from app.guardrails.ram.interfaces import NLIResult


def _make_nli_result(
    entailment_score: float = 0.92,
    contradiction_score: float = 0.05,
    neutral_score: float = 0.03,
    source_title: str = "Pedoman Rektor UPI",
    page: Optional[int] = 12,
    doc_id: str = "doc-123",
    evidence_snippet: str = "",
) -> NLIResult:
    """Create an NLIResult for testing."""
    return NLIResult(
        label="entailment",
        entailment_score=entailment_score,
        contradiction_score=contradiction_score,
        neutral_score=neutral_score,
        source_title=source_title,
        page=page,
        doc_id=doc_id,
        evidence_snippet=evidence_snippet,
    )


class TestFormatCitation:
    """Tests for ChatService._format_citation()."""

    def test_includes_all_three_scores(self) -> None:
        result = _make_nli_result(
            entailment_score=0.92, neutral_score=0.03, contradiction_score=0.05
        )
        citation = ChatService._format_citation(result)
        assert "Supported:0.92" in citation
        assert "Neutral:0.03" in citation
        assert "Contradicted:0.05" in citation

    def test_includes_source_page_docid(self) -> None:
        result = _make_nli_result(
            source_title="Pedoman Rektor UPI", page=12, doc_id="abc-123"
        )
        citation = ChatService._format_citation(result)
        assert "Pedoman Rektor UPI" in citation
        assert "Page 12" in citation
        assert "DocID:abc-123" in citation

    def test_none_result_returns_empty(self) -> None:
        assert ChatService._format_citation(None) == ""

    def test_wrapped_in_asterisk_parens(self) -> None:
        citation = ChatService._format_citation(_make_nli_result())
        assert citation.startswith(" *(")
        assert citation.endswith(")*")

    def test_scores_two_decimals(self) -> None:
        citation = ChatService._format_citation(
            _make_nli_result(entailment_score=0.123456)
        )
        assert "Supported:0.12" in citation

    def test_evidence_after_docid(self) -> None:
        result = _make_nli_result(doc_id="abc-123", evidence_snippet="Permohonan diajukan.")
        citation = ChatService._format_citation(result)
        assert citation.index("DocID:") < citation.index("Evidence:")

    def test_unverified_badge(self) -> None:
        assert ChatService._format_unverified() == " *(Unverified)*"
