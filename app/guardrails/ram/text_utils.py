"""Shared text-splitting helpers for the Response Assessment Module.

Two jobs: ``split_sentences_and_tail`` splits a streamed buffer into
(sentence, trailing-separator) pairs so claim boundaries survive streaming,
and ``split_sentences``/``split_table_windows`` window a retrieved KB chunk so
the RAM can locate the exact sub-passage an NLI premise should be built from.
Pure Python (stdlib ``re`` + ``shared.table_converter``) — no infra imports,
per the ``thesis/`` purity rule.
"""
import re
from typing import List, Tuple

from app.shared.table_converter import is_markdown_table, split_markdown_table_lines

# Abbreviations whose trailing "." is not a sentence end. "Rp." is the one that
# matters most in this corpus — every tariff figure is written "Rp. 500.000",
# and splitting there strands a claim on "Rp." alone, so the verification badge
# renders in the middle of the amount.
_ABBREVIATIONS = (
    "Rp", "No", "Nomor", "Hlm", "hlm", "Jl", "Jln",
    "Drs", "Dra", "Prof", "Dr", "Ir", "Hj", "Sdr", "Sdri",
    "dll", "dsb", "dst", "tgl", "yg", "ttd",
)

# Python's re requires each lookbehind to be fixed-width, but several separate
# lookbehinds of differing widths may be chained — one per abbreviation.
_ABBREV_GUARD = "".join(rf"(?<!\b{abbr}\.)" for abbr in _ABBREVIATIONS)

# A digit-preceded "."/"?"/"!" is almost always a markdown list marker (e.g.
# "1.", "2."), not a sentence end, so the negative lookbehind excludes it.
# Newlines delimit list items/paragraphs unambiguously, so they're boundaries
# in their own right. One capturing group makes split() also return the exact
# separator (e.g. "\n\n" for a paragraph break vs. " " for a sentence gap).
_SENTENCE_BOUNDARY = re.compile(
    r'((?:(?<!\d[.?!])' + _ABBREV_GUARD + r'(?<=[.?!])\s+)|(?:\n+))'
)


def split_sentences_and_tail(text: str) -> Tuple[List[Tuple[str, str]], str]:
    """Split text into terminated (sentence, separator) pairs plus a raw tail.

    Each pair is a fragment that was followed by a real boundary, stripped and
    paired with the exact whitespace that followed it (e.g. "\\n\\n" for a
    paragraph break), so callers can reconstruct the original formatting
    instead of always joining with a single space.

    The tail is the trailing fragment that had no boundary after it, returned
    **verbatim** — unstripped. Streaming callers append the next chunk to it, so
    stripping here would weld the last word of one chunk onto the first word of
    the next.
    """
    # One capturing group means split() always returns an odd-length list:
    # [frag, sep, frag, sep, ..., frag]. The final element is the raw tail.
    raw = _SENTENCE_BOUNDARY.split(text)
    pairs: List[Tuple[str, str]] = []
    for i in range(0, len(raw) - 1, 2):
        frag, sep = raw[i], raw[i + 1]
        if frag.strip():
            pairs.append((frag.strip(), sep))
    return pairs, raw[-1]


def split_sentences_with_seps(text: str) -> List[Tuple[str, str]]:
    """Split text into (sentence, trailing_separator) pairs.

    Convenience wrapper over :func:`split_sentences_and_tail` for callers that
    have the whole text already and don't need the tail kept separate: the
    unterminated tail comes back stripped, paired with an empty separator.
    """
    pairs, tail = split_sentences_and_tail(text)
    if tail.strip():
        pairs.append((tail.strip(), ""))
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


# Sentences that cannot carry a citation because they are not claims about the
# corpus at all: the assistant describing its own knowledge, or telling the
# user what to do. Badging these "Klaim tanpa kutipan sumber" puts a warning on
# an honest refusal — precisely the behaviour the guardrail exists to encourage
# — and teaches users to ignore the badge.
#
# Deliberately narrow. Each pattern needs an explicit self-reference or an
# explicit address to the user; a bare negation is not enough, because
# "Dokumen yang tersedia tidak mencakup seluruh program studi" *is* a checkable
# claim about the corpus and must keep its badge.
_UNCITEABLE_PATTERNS = (
    # The assistant describing what it does or does not have.
    r"\bsaya\s+(?:tidak|belum)\s+(?:memiliki|menemukan|mempunyai|dapat|bisa)\b",
    r"\bsaya\s+(?:sarankan|menyarankan)\b",
    r"\b(?:tidak\s+ada|tidak\s+terdapat|tidak\s+tersedia)\b[^.]*\bdalam\s+konteks\b",
    r"\bkonteks\s+yang\s+(?:diberikan|tersedia|saya\s+miliki)\b[^.]*\btidak\b",
    # The assistant addressing the user.
    r"\b[Aa]nda\s+(?:perlu|dapat|bisa|harus|sebaiknya|dianjurkan)\b",
    r"\bsilakan\b",
    r"\bdisarankan\b",
    r"\bsebaiknya\s+[Aa]nda\b",
)

_UNCITEABLE_RE = re.compile("|".join(_UNCITEABLE_PATTERNS), re.IGNORECASE)


def is_unciteable_statement(text: str) -> bool:
    """True when a sentence is not a factual claim about the retrieved corpus.

    Used to decide whether an uncited sentence deserves the "Unverified"
    badge. Refusals ("saya tidak memiliki informasi mengenai tarif 2026"),
    statements about the context itself, and advice to the user ("Anda perlu
    mengetahui program studi") have no citable source by construction.
    """
    if not text or not text.strip():
        return False
    return bool(_UNCITEABLE_RE.search(text))
