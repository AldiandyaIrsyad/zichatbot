"""Tests for LLMConnection._suppress_thinking (Qwen /no_think soft switch).

Measured: OpenRouter's ``reasoning={"enabled": False}`` is silently ignored for
qwen3, so ``content`` comes back empty and the whole English chain-of-thought
lands in the ``reasoning`` field — which the content-empty fallback then streams
as the answer (Exp4 measured 63-93% contamination). Qwen's ``/no_think`` token is
the mechanism that actually disables thinking; these tests pin that it's applied
to the right turn, only for Qwen, and never duplicated.
"""

from __future__ import annotations

from typing import Dict, List

from app.chat.infra.llm_connection import LLMConnection


def _msgs() -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": "Kamu asisten JDIH."},
        {"role": "user", "content": "Pertanyaan pertama?"},
        {"role": "assistant", "content": "Jawaban pertama."},
        {"role": "user", "content": "Pertanyaan kedua?"},
    ]


class TestSuppressThinking:
    def test_appends_no_think_to_last_user_turn_for_qwen(self) -> None:
        out = LLMConnection._suppress_thinking("qwen/qwen3-14b", _msgs())
        assert out[-1]["content"] == "Pertanyaan kedua? /no_think"
        # earlier user turn untouched
        assert out[1]["content"] == "Pertanyaan pertama?"
        # system turn untouched
        assert out[0]["content"] == "Kamu asisten JDIH."

    def test_noop_for_non_qwen_models(self) -> None:
        msgs = _msgs()
        out = LLMConnection._suppress_thinking("openai/gpt-5.6", msgs)
        assert out == msgs
        assert all("/no_think" not in m["content"] for m in out)

    def test_case_insensitive_and_size_agnostic(self) -> None:
        for model in ("Qwen/Qwen3-8B", "qwen/qwen3-32b", "qwen/qwen3-14b"):
            out = LLMConnection._suppress_thinking(model, _msgs())
            assert out[-1]["content"].endswith("/no_think")

    def test_not_duplicated_if_already_present(self) -> None:
        msgs = [{"role": "user", "content": "Sudah ada /no_think"}]
        out = LLMConnection._suppress_thinking("qwen/qwen3-14b", msgs)
        assert out[0]["content"].count("/no_think") == 1

    def test_does_not_mutate_input(self) -> None:
        msgs = _msgs()
        LLMConnection._suppress_thinking("qwen/qwen3-14b", msgs)
        assert msgs[-1]["content"] == "Pertanyaan kedua?", "input list must be untouched"

    def test_no_user_turn_is_safe(self) -> None:
        msgs = [{"role": "system", "content": "system only"}]
        out = LLMConnection._suppress_thinking("qwen/qwen3-14b", msgs)
        assert out == msgs
