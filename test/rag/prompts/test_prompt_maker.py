"""Tests for the prompt maker (Indonesian prompts + injection-defense delimiter)."""
from __future__ import annotations

import re

from app.rag.prompts import (
    DEFAULT_SYSTEM_PROMPT_ID,
    build_context_block,
    build_prompt,
    build_system_prompt,
    build_user_turn,
    make_user_delimiter,
)
from app.guardrails.ram.interfaces import RetrievedContext


def _make_context(text: str = "Some SOP text.", breadcrumbs=(), page=None) -> RetrievedContext:
    return RetrievedContext(text=text, source_title="Doc", breadcrumbs=list(breadcrumbs), page=page)


class TestMakeUserDelimiter:
    def test_returns_hex_string(self) -> None:
        nonce = make_user_delimiter()
        assert re.fullmatch(r"[0-9a-f]+", nonce)

    def test_nonces_are_unique_across_calls(self) -> None:
        nonces = {make_user_delimiter() for _ in range(20)}
        assert len(nonces) == 20


class TestBuildContextBlock:
    def test_empty_contexts_returns_empty_string(self) -> None:
        assert build_context_block([]) == ""

    def test_includes_breadcrumbs_when_present(self) -> None:
        ctx = _make_context(text="Isi pasal.", breadcrumbs=["BAB I", "Pasal 5"])
        block = build_context_block([ctx])
        assert "BAB I > Pasal 5" in block
        assert "Isi pasal." in block

    def test_numbers_multiple_contexts(self) -> None:
        block = build_context_block([_make_context("A"), _make_context("B")])
        assert "Sumber 1" in block
        assert "Sumber 2" in block

    def test_omits_breadcrumb_brackets_when_absent(self) -> None:
        block = build_context_block([_make_context("A", breadcrumbs=[])])
        assert block == "[Sumber 1]\nA"

    def test_includes_page_when_present(self) -> None:
        ctx = _make_context("Isi pasal.", page=12)
        block = build_context_block([ctx])
        assert "Halaman 12" in block
        assert "Isi pasal." in block

    def test_pairs_page_and_content_in_one_block(self) -> None:
        ctx = _make_context("Isi pasal.", breadcrumbs=["BAB I"], page=12)
        block = build_context_block([ctx])
        assert block == "[Sumber 1, Halaman 12, BAB I]\nIsi pasal."


class TestBuildUserTurn:
    def test_wraps_user_message_in_delimiter_tags(self) -> None:
        nonce = "deadbeef"
        turn = build_user_turn("Apa itu SOP?", "", nonce)
        assert f"<user_input_{nonce}>" in turn
        assert f"</user_input_{nonce}>" in turn
        assert "Apa itu SOP?" in turn

    def test_delimiter_always_applied_even_without_context(self) -> None:
        nonce = "abc123"
        turn = build_user_turn("Halo", "", nonce)
        assert f"<user_input_{nonce}>" in turn
        assert "Konteks:" not in turn

    def test_includes_context_block_when_present(self) -> None:
        nonce = "abc123"
        turn = build_user_turn("Halo", "[1] Some context", nonce)
        assert "Konteks:" in turn
        assert "[1] Some context" in turn

    def test_user_message_is_between_the_delimiter_tags(self) -> None:
        nonce = "feedface"
        turn = build_user_turn("Instruksi jahat", "", nonce)
        start = turn.index(f"<user_input_{nonce}>")
        end = turn.index(f"</user_input_{nonce}>")
        message_pos = turn.index("Instruksi jahat")
        assert start < message_pos < end


class TestBuildSystemPrompt:
    def test_references_the_same_nonce(self) -> None:
        nonce = "cafebabe"
        prompt = build_system_prompt("Base prompt.", nonce)
        assert f"<user_input_{nonce}>" in prompt
        assert f"</user_input_{nonce}>" in prompt

    def test_preserves_base_prompt(self) -> None:
        prompt = build_system_prompt("Base prompt.", "1234")
        assert prompt.startswith("Base prompt.")


class TestBuildPrompt:
    def test_bundle_nonce_matches_across_system_and_user(self) -> None:
        bundle = build_prompt("Apa itu SOP?", [], "Base prompt.")
        assert f"<user_input_{bundle.nonce}>" in bundle.user_turn
        assert f"<user_input_{bundle.nonce}>" in bundle.system_prompt

    def test_default_system_prompt_is_indonesian(self) -> None:
        assert "Bahasa Indonesia" in DEFAULT_SYSTEM_PROMPT_ID

    def test_empty_contexts_still_delimits_user_message(self) -> None:
        bundle = build_prompt("Halo", [], "Base prompt.")
        assert f"<user_input_{bundle.nonce}>" in bundle.user_turn
        assert "Halo" in bundle.user_turn

    def test_contexts_included_in_user_turn(self) -> None:
        ctx = _make_context("Isi SOP penting.")
        bundle = build_prompt("Apa isi SOP?", [ctx], "Base prompt.")
        assert "Isi SOP penting." in bundle.user_turn


class TestBuildPromptWithoutNonce:
    """``use_nonce=False`` ablates the structural injection defense.

    It is a separate layer from the classifier-based IVM guard, so measuring
    what it contributes requires being able to turn it off on its own.
    """

    def test_user_message_is_not_wrapped(self) -> None:
        bundle = build_prompt("Apa itu SOP?", [], "Base prompt.", use_nonce=False)
        assert "<user_input_" not in bundle.user_turn
        assert "Apa itu SOP?" in bundle.user_turn

    def test_system_prompt_drops_the_delimiter_defense(self) -> None:
        bundle = build_prompt("Apa itu SOP?", [], "Base prompt.", use_nonce=False)
        assert bundle.system_prompt == "Base prompt."
        assert "user_input_" not in bundle.system_prompt

    def test_nonce_is_empty(self) -> None:
        bundle = build_prompt("Apa itu SOP?", [], "Base prompt.", use_nonce=False)
        assert bundle.nonce == ""

    def test_context_still_included(self) -> None:
        ctx = _make_context("Isi SOP penting.")
        bundle = build_prompt("Apa isi SOP?", [ctx], "Base prompt.", use_nonce=False)
        assert "Isi SOP penting." in bundle.user_turn
        assert "Apa isi SOP?" in bundle.user_turn
