"""Dependency-parser-backed clause splitting for the Response Assessment Module.

Splits a compound Indonesian sentence into atomic clauses so each can be
verified separately by NLI. Uses Stanza's Indonesian (``id``) Universal
Dependencies model for a dependency-informed split on coordinating
conjunctions; falls back to a regex conjunction split when Stanza is disabled
or unavailable, so the RAM never hard-depends on a heavy model at import time.

The pipeline is lazy-loaded (first use) and process-local; Stanza models are
large, so callers should construct one :class:`ClauseSplitter` per process and
reuse it (see ``app/chat/dependency.py``).
"""

from __future__ import annotations

import re
import threading
from typing import List

import structlog

logger = structlog.get_logger(__name__)

# Conjunction tokens whose dependency label (``cc``) marks a coordination
# boundary. Clause splitting happens immediately before these tokens; the
# conjunction is carried into the following clause.
_FALLBACK_CONJ_RE = re.compile(r"(?i)(,\s*(?:yang|dan|serta|atau|tetapi|namun|sedangkan|sementara|karena|sehingga)\s+)")


class ClauseSplitter:
    """Lazily-loaded Indonesian dependency parser for atomic claim extraction."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._nlp = None
        # Stanza pipelines are not thread-safe. The parser is offloaded to a
        # worker thread (asyncio.to_thread) from the chat service, so multiple
        # requests could call ``nlp()`` concurrently; this lock serializes them.
        self._lock = threading.Lock()

    def _load(self):
        """Load the Stanza ``id`` pipeline on first use (never at import)."""
        if self._nlp is None:
            import stanza  # imported lazily: heavy + model download on first use

            self._nlp = stanza.Pipeline(
                "id",
                # ``lemma`` is not optional: the ``id`` depparse model takes
                # lemmas as a feature and raises at call time without it, so
                # omitting it silently degraded every sentence to the regex
                # fallback.
                processors="tokenize,pos,lemma,depparse",
                verbose=False,
                use_gpu=False,
            )
        return self._nlp

    def split_clauses(self, sentence: str) -> List[str]:
        """Split ``sentence`` into atomic clauses, or ``[sentence]`` unchanged.

        Never raises: a parser outage degrades to the regex fallback, and a
        degenerate result degrades to the whole sentence.
        """
        if not self.enabled:
            return [sentence]

        clauses: List[str] = []
        try:
            nlp = self._load()
            with self._lock:
                doc = nlp(sentence)
            for sent in doc.sentences:
                clauses.extend(self._split_one_sentence(sent, sentence))
        except Exception as exc:
            logger.warning("ram.clause_splitter.failed", error=str(exc))

        clauses = [c.strip() for c in clauses if c and c.strip()]
        return clauses or [sentence]

    @staticmethod
    def _char_span(word) -> tuple[int, int]:
        """Character offsets of a Stanza ``Word``.

        Offsets live on the ``Token``, not the ``Word`` — a multi-word token
        expands to several words sharing one span — so they are read through
        ``word.parent``, falling back to the word itself for stubs and for
        parsers that put the offsets there.
        """
        holder = getattr(word, "parent", None) or word
        return holder.start_char, holder.end_char

    @classmethod
    def _split_one_sentence(cls, sent, text: str) -> List[str]:
        """Split one Stanza sentence at ``cc`` (coordinating conjunction) words.

        Iterates ``sent.words``, not ``sent.tokens``: ``deprel`` is a property
        of the syntactic word, and a ``Token`` does not carry it at all — the
        earlier token-based version raised ``'Token' object has no attribute
        'deprel'`` on every real sentence and silently fell back to regex.

        Clauses are *sliced* out of ``text`` (the string handed to the
        pipeline) by character offsets rather than rebuilt by joining word
        texts, so the original spacing and punctuation survive verbatim — a
        join turns "ayat (1)" into "ayat ( 1 )".
        """
        words = getattr(sent, "words", None) or []
        if not words:
            return []

        # Find boundary word indices: a cc word begins a new clause, and the
        # previous clause ends right before it. Skip a boundary that would
        # produce an empty clause (e.g. a sentence-initial conjunction).
        boundaries = [
            i for i, w in enumerate(words)
            if w.deprel == "cc" and 0 < i < len(words) - 1
        ]

        end = cls._char_span(words[-1])[1]
        first_start = cls._char_span(words[0])[0]
        if not boundaries:
            return [text[first_start:end]]

        clauses: List[str] = []
        start = first_start
        for b in boundaries:
            b_start = cls._char_span(words[b])[0]
            if b_start > start:
                clauses.append(text[start:b_start])
            start = b_start
        clauses.append(text[start:end])
        return clauses

    @staticmethod
    def split_clauses_fallback(sentence: str) -> List[str]:
        """Regex-only fallback used when Stanza is disabled (kept for tests)."""
        parts = _FALLBACK_CONJ_RE.split(sentence)
        clauses: List[str] = []
        current = ""
        for part in parts:
            if part is None:
                continue
            if _FALLBACK_CONJ_RE.fullmatch(part):
                if current.strip():
                    clauses.append(current.strip())
                current = part.lstrip(", ")
            else:
                current += part
        if current.strip():
            clauses.append(current.strip())
        return clauses or [sentence]
