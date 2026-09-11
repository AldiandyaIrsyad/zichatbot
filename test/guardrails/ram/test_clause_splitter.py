"""Tests for clause splitting.

The Stanza pipeline itself is not exercised (heavy model download); these
tests drive ``_split_one_sentence`` with stub objects shaped like Stanza's,
plus the regex fallback path, which is where the logic lives.

The stubs mirror Stanza's real split of responsibilities — ``deprel`` on the
``Word``, character offsets on its parent ``Token`` — because an earlier stub
that put both on one object let a version that read ``sent.tokens[i].deprel``
pass every test while raising ``AttributeError`` on every real sentence.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from app.guardrails.ram.clause_splitter import ClauseSplitter


@dataclass
class _Token:
    """Stanza's surface token: carries offsets, but no ``deprel``."""

    text: str
    start_char: int
    end_char: int


@dataclass
class _Word:
    """Stanza's syntactic word: carries ``deprel``, offsets via ``parent``."""

    text: str
    deprel: Optional[str]
    parent: _Token


@dataclass
class _Sentence:
    words: List[_Word]

    @property
    def text(self) -> str:
        return ""


def _tokenize(text: str, cc: tuple[str, ...] = ()) -> _Sentence:
    """Build a stub Stanza sentence, labelling ``cc`` words as conjunctions.

    Offsets are the real ones from ``text``, which is the whole point: the
    splitter slices the original string rather than rejoining word texts.
    """
    words: List[_Word] = []
    cursor = 0
    for word in text.split(" "):
        start = text.index(word, cursor)
        cursor = start + len(word)
        words.append(
            _Word(word, "cc" if word in cc else "case", _Token(word, start, cursor))
        )
    return _Sentence(words)


class TestSplitOneSentence:
    def test_preserves_original_spacing_and_punctuation(self) -> None:
        # A token join would produce "ayat ( 1 )" — the offsets must be sliced.
        text = "Ketentuan pada ayat (1) berlaku."
        sent = _tokenize(text)
        assert ClauseSplitter._split_one_sentence(sent, text) == [text]

    def test_splits_at_coordinating_conjunction(self) -> None:
        text = "Pasal 1 mengatur ayat (1) dan Pasal 2 mengatur ayat (2)."
        sent = _tokenize(text, cc=("dan",))
        clauses = ClauseSplitter._split_one_sentence(sent, text)
        assert clauses == [
            "Pasal 1 mengatur ayat (1) ",
            "dan Pasal 2 mengatur ayat (2).",
        ]
        # Concatenated, the clauses reproduce the sentence exactly.
        assert "".join(clauses) == text

    def test_sentence_initial_conjunction_is_not_a_boundary(self) -> None:
        text = "dan Pasal 2 berlaku."
        sent = _tokenize(text, cc=("dan",))
        assert ClauseSplitter._split_one_sentence(sent, text) == [text]

    def test_empty_word_list(self) -> None:
        assert ClauseSplitter._split_one_sentence(_Sentence([]), "") == []

    def test_offsets_are_read_from_the_parent_token(self) -> None:
        # A word whose own attributes would mislead: only parent offsets are
        # correct, which is the multi-word-token case Stanza models this way.
        text = "Pasal 1 dan Pasal 2."
        sent = _tokenize(text, cc=("dan",))
        for word in sent.words:
            word.start_char = 0  # type: ignore[attr-defined]
            word.end_char = 0  # type: ignore[attr-defined]
        assert "".join(ClauseSplitter._split_one_sentence(sent, text)) == text


class TestSplitClausesDisabled:
    def test_disabled_returns_sentence_unchanged(self) -> None:
        splitter = ClauseSplitter(enabled=False)
        assert splitter.split_clauses("Ayat (1) berlaku.") == ["Ayat (1) berlaku."]


class TestFallback:
    def test_splits_on_comma_conjunction(self) -> None:
        clauses = ClauseSplitter.split_clauses_fallback(
            "Permohonan diajukan, dan berkas diverifikasi."
        )
        assert clauses == ["Permohonan diajukan", "dan berkas diverifikasi."]

    def test_no_conjunction_returns_whole_sentence(self) -> None:
        text = "Ayat (1) berlaku."
        assert ClauseSplitter.split_clauses_fallback(text) == [text]
