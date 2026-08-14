"""Shared text-splitting helpers for the Response Assessment Module.

Used both to decide what generated text to run NLI on
(``ChatService._split_propositions``) and to window the retrieved KB text for
reverse-mapping the exact NLI premise (``RAMService.assess_sentence``). Pure
Python (stdlib ``re`` + ``shared.table_converter``) — no infra
imports, per the ``thesis/`` purity rule.
"""
import re
from typing import List, Tuple

from app.shared.table_converter import is_markdown_table, split_markdown_table_lines

# A digit-preceded "."/"?"/"!" is almost always a markdown list marker (e.g.
# "1.", "2."), not a sentence end, so the negative lookbehind excludes it.
# Newlines delimit list items/paragraphs unambiguously, so they're boundaries
# in their own right. One capturing group makes split() also return the exact
# separator (e.g. "\n\n" for a paragraph break vs. " " for a sentence gap).
_SENTENCE_BOUNDARY = re.compile(r'((?:(?<!\d[.?!])(?<=[.?!])\s+)|(?:\n+))')


def split_sentences_with_seps(text: str) -> List[Tuple[str, str]]:
    """Split text into (sentence, trailing_separator) pairs.

    The trailing separator is the exact whitespace that followed the sentence
    (e.g. "\\n\\n" for a paragraph break), so callers can reconstruct the
    original formatting instead of always joining with a single space. Returns
    non-empty stripped fragments paired with their separator ("" for a trailing
    unterminated fragment).
    """
    raw = _SENTENCE_BOUNDARY.split(text)
    pairs: List[Tuple[str, str]] = []
    for i in range(0, len(raw), 2):
        frag = raw[i]
        sep = raw[i + 1] if i + 1 < len(raw) else ""
        if frag and frag.strip():
            pairs.append((frag.strip(), sep))
    return pairs


def split_sentences(text: str) -> List[str]:
    """Split text into non-empty, stripped sentence-like units."""
    return [s for s, _ in split_sentences_with_seps(text)]


def split_table_windows(
    text: str,
    rows_per_window: int = 3,
    row_step: int = 2,
) -> List[str]:
    """Split a Markdown table into overlapping row-group windows.

    Unlike ``split_sentences`` (which treats every newline as a boundary,
    shredding a table into headerless fragments after the first window), this
    prepends the header + separator row to *every* window — the same
    header-repetition used for child-chunk embedding, but sized for
    reranker/NLI windows. ``row_step`` < ``rows_per_window`` yields overlap.

    Returns ``"<header>\\n<separator>\\n<row>..."`` strings with no synthetic
    trailing period (punctuation after a row's closing ``|`` would corrupt it).
    Returns ``[]`` if ``text`` isn't a parseable Markdown table — callers fall
    back to ``split_sentences``.
    """
    if not is_markdown_table(text):
        return []
    parsed = split_markdown_table_lines(text)
    if parsed is None:
        return []
    header, separator, data_rows = parsed
    if not data_rows:
        return []

    windows: List[str] = []
    for i in range(0, max(1, len(data_rows)), row_step):
        row_group = data_rows[i:i + rows_per_window]
        if not row_group:
            continue
        windows.append("\n".join([header, separator] + row_group))
    return windows
