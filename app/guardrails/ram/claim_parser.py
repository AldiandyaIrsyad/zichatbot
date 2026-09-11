"""Deterministic Markdown claim parser for the Response Assessment Module.

Extracts claim units from LLM-generated Markdown and associates each with its
``[CIT:N]`` citation markers. Pure Python (stdlib ``re`` + shared table
helpers) — no infra imports, per the purity rule.

Block model kept deliberately small: paragraphs, list items, and GFM tables are
the only Markdown structures the citation protocol requires the model to use.
Headings/code fences that slip through are treated as ordinary prose lines.
"""

from __future__ import annotations

import re
import structlog
from dataclasses import dataclass, replace
from typing import List, Optional, Tuple

from app.shared.table_converter import is_markdown_table, split_markdown_table_lines

from .interfaces import ClaimUnit
from .text_utils import split_sentences_and_tail

logger = structlog.get_logger(__name__)

# [CIT:1] / [CIT:1,2] / [CIT: 1, 2]
CITATION_RE = re.compile(r"\[CIT:\s*([0-9]+(?:\s*,\s*[0-9]+)*)\s*\]")

# A GFM table row (header, separator, or data row) starts and ends with "|".
TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")

# Bullet/ordered list item marker at the start of a line.
LIST_ITEM_RE = re.compile(r"^(\s*)(?:[-*+]|\d+[.)])\s+")

# Punctuation the clause splitter can leave as a unit of its own ("." / ":"),
# which must not be mistaken for prose that ends a list's citation scope.
_PUNCTUATION_ONLY = " .,:;!?-–—"

# Indonesian conjunction split (fallback when no dependency parser is wired).
_CONJ_SPLIT_RE = re.compile(r"(?i)(,\s*yang\s+|,\s*dan\s+|,\s*karena\s+|,\s*sehingga\s+)")


def extract_citations(text: str) -> Tuple[str, Tuple[int, ...]]:
    """Strip all ``[CIT:...]`` markers, returning ``(clean_text, citation_ids)``.

    Duplicate ids within one claim are removed; order is preserved.
    """
    ids: List[int] = []

    def _collect(match: re.Match) -> str:
        for token in match.group(1).split(","):
            token = token.strip()
            if token.isdigit():
                ids.append(int(token))
        return ""

    clean = CITATION_RE.sub(_collect, text)
    seen = set()
    unique_ids = tuple(i for i in ids if not (i in seen or seen.add(i)))
    return clean, unique_ids


def _ends_with_citation(text: str) -> bool:
    """True if ``text`` ends with a closed ``[CIT:...]`` marker."""
    stripped = text.rstrip()
    if not stripped.endswith("]"):
        return False
    return bool(CITATION_RE.search(text))


def _is_table_row(text: str) -> bool:
    return bool(TABLE_ROW_RE.match(text.strip()))


def _is_list_item(text: str) -> bool:
    return bool(LIST_ITEM_RE.match(text))


def split_cells(row: str) -> List[str]:
    """Split a Markdown table row into trimmed cell texts.

    Escaped ``\\|`` inside cells is not handled — LLM-generated tables in this
    domain do not contain literal pipes, so a plain split stays deterministic.
    """
    s = row.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    return [c.strip() for c in s.split("|")]


@dataclass(frozen=True)
class TableCellClaim:
    """A single table cell carrying one or more ``[CIT:N]`` markers."""

    row_index: int
    col_index: int
    cell_text: str
    citation_ids: Tuple[int, ...]


@dataclass(frozen=True)
class TableBlock:
    """A parsed Markdown table: header/separator/rows plus per-cell claims."""

    header: str
    separator: str
    data_lines: Tuple[str, ...]
    claims: Tuple[TableCellClaim, ...]
    # Citation ids found on the header/separator lines (rare) — callers may
    # treat the whole table as one block-level claim if no cell claims exist.
    block_citation_ids: Tuple[int, ...]


def parse_table_block(table_text: str) -> Optional[TableBlock]:
    """Parse a complete Markdown table into header/separator/rows + cell claims.

    Returns ``None`` when ``table_text`` isn't a parseable GFM table (callers
    fall back to prose handling).
    """
    if not is_markdown_table(table_text):
        return None
    parsed = split_markdown_table_lines(table_text)
    if parsed is None:
        return None
    header, separator, data_lines = parsed

    claims: List[TableCellClaim] = []
    for row_index, line in enumerate(data_lines):
        for col_index, cell in enumerate(split_cells(line)):
            clean, ids = extract_citations(cell)
            if ids:
                claims.append(TableCellClaim(row_index, col_index, clean.strip(), ids))

    block_ids: List[int] = []
    for line in (header, separator):
        _, ids = extract_citations(line)
        block_ids.extend(ids)

    return TableBlock(
        header=header,
        separator=separator,
        data_lines=tuple(data_lines),
        claims=tuple(claims),
        block_citation_ids=tuple(block_ids),
    )


def _split_clauses(text: str, clause_splitter=None) -> List[str]:
    """Split a prose sentence into atomic clauses.

    Uses the dependency-parser-backed ``clause_splitter`` when provided;
    otherwise falls back to the same Indonesian conjunction split the old
    proposition splitter used.
    """
    if clause_splitter is not None:
        try:
            clauses = clause_splitter.split_clauses(text)
            if clauses and len(clauses) > 1:
                return [c for c in clauses if c.strip()]
        except Exception as exc:
            # split_clauses is documented to never raise; an exception here is
            # a contract violation. Log it (not a silent pass) and fall back.
            logger.warning("ram.clause_split_contract_violation", error=str(exc))

    parts = _CONJ_SPLIT_RE.split(text)
    clauses: List[str] = []
    current = ""
    for part in parts:
        if part is None:
            continue
        if _CONJ_SPLIT_RE.fullmatch(part):
            if current.strip():
                clauses.append(current.strip())
            current = part.lstrip(", ")
        else:
            current += part
    if current.strip():
        clauses.append(current.strip())
    return clauses or [text]


def _clauses_to_units(
    text: str,
    citation_ids: Tuple[int, ...],
    separator: str,
    clause_splitter,
) -> List[ClaimUnit]:
    """Split one prose span into atomic clauses, each carrying ``citation_ids``.

    Only the final clause is marked ``is_sentence_end``; the others are
    assessed individually but re-emitted without a full stop or badge so the
    sentence reads as one sentence.
    """
    clauses = [c for c in _split_clauses(text, clause_splitter) if c.strip()]
    return [
        ClaimUnit(
            text=c,
            citation_ids=citation_ids,
            kind="prose",
            separator=separator,
            is_sentence_end=(i == len(clauses) - 1),
        )
        for i, c in enumerate(clauses)
    ]


def _split_prose_claims(text: str, separator: str, clause_splitter) -> List[ClaimUnit]:
    """Split prose into claims, attaching each ``[CIT:n]`` to the text before it.

    The citation marker is a claim boundary in its own right, so two cited
    sentences ``"A.[CIT:1] B.[CIT:2]"`` split into two claims even though the
    sentence-boundary regex sees no ``". "`` between them.
    """
    parts = CITATION_RE.split(text)
    units: List[ClaimUnit] = []
    pending = ""
    for i in range(0, len(parts), 2):
        seg = parts[i]
        ids_str = parts[i + 1] if i + 1 < len(parts) else None
        pending += seg
        if ids_str is not None:
            ids = tuple(int(t) for t in ids_str.split(",") if t.strip().isdigit())
            # Dedup preserving order (mirrors extract_citations) so [CIT:1,1]
            # doesn't trigger duplicate NLI calls on the same chunk.
            seen: set[int] = set()
            ids = tuple(i for i in ids if not (i in seen or seen.add(i)))
            units.extend(_clauses_to_units(pending, ids, " ", clause_splitter))
            pending = ""
    if pending.strip():
        units.extend(_clauses_to_units(pending, (), " ", clause_splitter))

    # Only the last clause carries the piece's real trailing separator.
    if units:
        last = units[-1]
        units[-1] = ClaimUnit(
            text=last.text,
            citation_ids=last.citation_ids,
            kind=last.kind,
            separator=separator,
            is_sentence_end=last.is_sentence_end,
        )
    return units


def _classify_piece(text: str, separator: str, clause_splitter=None) -> List[ClaimUnit]:
    """Classify one complete ``(text, separator)`` piece into claim units."""
    stripped = text.strip()
    if not stripped:
        return []

    if _is_table_row(stripped):
        # Keep the row verbatim (markers intact) so the table parser sees them.
        return [ClaimUnit(text=text, kind="table_row", separator=separator)]

    if _is_list_item(stripped):
        clean, ids = extract_citations(stripped)
        return [ClaimUnit(text=clean, citation_ids=ids, kind="list_item", separator=separator)]

    return _split_prose_claims(text, separator, clause_splitter)


def split_claims(buffer: str, clause_splitter=None) -> Tuple[List[ClaimUnit], str]:
    """Incrementally split ``buffer`` into complete claim units + a remainder.

    Complete units are safe to assess and emit immediately; ``remainder`` is the
    trailing fragment that may still grow (an unterminated sentence, list item,
    or a partially-written ``[CIT`` marker).
    """
    if not buffer:
        return [], ""

    complete_pairs, tail = split_sentences_and_tail(buffer)

    # The tail had no boundary after it, so it is normally still growing. The
    # exception is a closed citation marker, which terminates a claim on its
    # own; it never carries trailing whitespace (the marker *is* the end), so
    # synthesise the single space that would otherwise separate it from the
    # next claim.
    if tail.strip() and _ends_with_citation(tail):
        complete_pairs = complete_pairs + [(tail.strip(), " ")]
        remainder = ""
    else:
        # Verbatim, including any trailing whitespace: the caller concatenates
        # the next stream chunk onto this.
        remainder = tail

    units: List[ClaimUnit] = []
    # A bulleted list normally carries its citation on the sentence that
    # introduces it ("...dibagi menjadi 8 kelompok, yaitu [CIT:4]:"), never on
    # each bullet. Assessed piece-by-piece the bullets look uncited, so every
    # one of them was badged "Unverified" — a claim-without-a-source label
    # printed directly beneath the source. Bullets therefore inherit the
    # citations of the colon lead-in they belong to.
    lead_citations: Tuple[int, ...] = ()
    prev_citations: Tuple[int, ...] = ()
    for text, sep in complete_pairs:
        for unit in _classify_piece(text, sep, clause_splitter):
            if unit.kind == "list_item":
                if not unit.citation_ids and lead_citations:
                    unit = replace(unit, citation_ids=lead_citations)
                units.append(unit)
                continue

            units.append(unit)
            if unit.kind == "table_row":
                continue

            stripped = unit.text.strip()
            if stripped.endswith(":"):
                # The lead-in may carry the marker itself ("yaitu [CIT:4]:") or
                # arrive as a bare ":" after the clause splitter took the text.
                lead_citations = unit.citation_ids or prev_citations
            elif stripped.strip(_PUNCTUATION_ONLY):
                # Any real prose ends the list's scope.
                prev_citations = unit.citation_ids
                lead_citations = ()
    return units, remainder
