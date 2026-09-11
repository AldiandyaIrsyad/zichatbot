"""Tests for the NLI bounded context (domain, selector, adapters' parsing)."""

from __future__ import annotations

import pytest

from app.guardrails.nli.domain.models import (
    LABEL_CONTRADICTION,
    LABEL_ENTAILMENT,
    LABEL_NEUTRAL,
    NLI_LABELS,
    LabelSpace,
    NLIModelKind,
    NLIModelSpec,
    NLIResult,
)


class TestDomainModels:
    def test_nli_labels_order_is_pinned(self) -> None:
        assert NLI_LABELS == ("entailment", "neutral", "contradiction")

    def test_nli_result_defaults(self) -> None:
        r = NLIResult(label="entailment")
        assert r.entailment_score == 0.0
        assert r.contradiction_score == 0.0
        assert r.neutral_score == 0.0
        assert r.doc_id == ""

    def test_spec_defaults_to_three_way(self) -> None:
        spec = NLIModelSpec(kind=NLIModelKind.INDO_ROBERTA)
        assert spec.label_space == LabelSpace.THREE_WAY
        assert spec.max_concurrency == 8


class TestZeroshotParseResponse:
    def test_entailment_wins(self) -> None:
        from app.guardrails.nli.infra.zeroshot_client import ZeroshotNLIClient

        data = {
            "data": [
                [
                    {"label": "entailment", "score": 0.8},
                    {"label": "not_entailment", "score": 0.2},
                ]
            ]
        }
        result = ZeroshotNLIClient._parse_response(data)
        assert result.label == LABEL_ENTAILMENT
        assert result.entailment_score == 0.8
        assert result.neutral_score == 0.2
        assert result.contradiction_score == 0.0

    def test_not_entailment_maps_to_neutral(self) -> None:
        from app.guardrails.nli.infra.zeroshot_client import ZeroshotNLIClient

        data = {
            "data": [
                [
                    {"label": "entailment", "score": 0.3},
                    {"label": "not_entailment", "score": 0.7},
                ]
            ]
        }
        result = ZeroshotNLIClient._parse_response(data)
        assert result.label == LABEL_NEUTRAL
        assert result.contradiction_score == 0.0

    def test_label_0_1_fallback(self) -> None:
        from app.guardrails.nli.infra.zeroshot_client import ZeroshotNLIClient

        data = {
            "data": [
                [
                    {"label": "label_0", "score": 0.6},
                    {"label": "label_1", "score": 0.4},
                ]
            ]
        }
        result = ZeroshotNLIClient._parse_response(data)
        assert result.label == LABEL_ENTAILMENT
        assert result.entailment_score == 0.6


class TestLocalToResult:
    def test_three_way_argmax(self) -> None:
        from app.guardrails.nli.infra.local_client import LocalNLIClient

        client = LocalNLIClient(model_id="x", label_space=LabelSpace.THREE_WAY)
        result = client._to_result(
            [0.7, 0.2, 0.1],
            {0: "entailment", 1: "neutral", 2: "contradiction"},
        )
        assert result.label == LABEL_ENTAILMENT
        assert result.entailment_score == 0.7
        assert result.neutral_score == 0.2
        assert result.contradiction_score == 0.1

    def test_binary_maps_not_entailment_to_neutral(self) -> None:
        from app.guardrails.nli.infra.local_client import LocalNLIClient

        client = LocalNLIClient(model_id="x", label_space=LabelSpace.BINARY)
        result = client._to_result(
            [0.4, 0.6],
            {0: "entailment", 1: "not_entailment"},
        )
        assert result.label == LABEL_NEUTRAL
        assert result.contradiction_score == 0.0
        assert result.neutral_score == 0.6

    def test_binary_entailment(self) -> None:
        from app.guardrails.nli.infra.local_client import LocalNLIClient

        client = LocalNLIClient(model_id="x", label_space=LabelSpace.BINARY)
        result = client._to_result(
            [0.9, 0.1],
            {0: "entailment", 1: "not_entailment"},
        )
        assert result.label == LABEL_ENTAILMENT
        assert result.entailment_score == 0.9


class TestRamReexport:
    def test_ram_interfaces_reexports_nli_types(self) -> None:
        # Backward compat: existing import sites keep working.
        from app.guardrails.ram.interfaces import INLIModel, NLIResult as R  # noqa: F401
        from app.guardrails.nli.domain.interfaces import INLIModel as NLI_INLIModel

        assert INLIModel is NLI_INLIModel
        assert R is NLIResult
