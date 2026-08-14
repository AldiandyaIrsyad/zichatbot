"""Unit tests for get_relevance_checker's ood_method branching
(app/chat/dependency.py)."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

from pydantic import SecretStr

import app.chat.dependency as dependency
from app.guardrails.ivm.checkers import (
    LLMJudgeRelevanceChecker,
    NliEntailmentRelevanceChecker,
    SimilarityThresholdRelevanceChecker,
)


def _fake_config(**overrides):
    base = dict(
        ood_method="llm_judge",
        ood_similarity_threshold=0.02,
        ood_nli_entailment_threshold=0.5,
        llm_base_url="http://x",
        llm_api_key=SecretStr("k"),
        llm_model="m",
        relevance_judge_prompt="p",
        relevance_judge_user_template="Context:\n{context}\n\nQuery: {query}\n\nIs this relevant?",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


class TestGetRelevanceChecker:
    def test_default_returns_llm_judge_checker(self, monkeypatch) -> None:
        monkeypatch.setattr(dependency, "get_chat_config", lambda: _fake_config())
        checker = dependency.get_relevance_checker(nli_client=AsyncMock())
        assert isinstance(checker, LLMJudgeRelevanceChecker)

    def test_similarity_threshold_method_selected(self, monkeypatch) -> None:
        monkeypatch.setattr(
            dependency,
            "get_chat_config",
            lambda: _fake_config(ood_method="similarity_threshold", ood_similarity_threshold=0.07),
        )
        checker = dependency.get_relevance_checker(nli_client=AsyncMock())
        assert isinstance(checker, SimilarityThresholdRelevanceChecker)
        assert checker.threshold == 0.07

    def test_nli_entailment_method_selected_and_injects_nli_client(self, monkeypatch) -> None:
        monkeypatch.setattr(
            dependency,
            "get_chat_config",
            lambda: _fake_config(ood_method="nli_entailment", ood_nli_entailment_threshold=0.6),
        )
        fake_nli = AsyncMock()
        checker = dependency.get_relevance_checker(nli_client=fake_nli)
        assert isinstance(checker, NliEntailmentRelevanceChecker)
        assert checker.nli_model is fake_nli
        assert checker.threshold == 0.6
