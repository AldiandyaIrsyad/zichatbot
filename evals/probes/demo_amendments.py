"""Demo — which Pasal an amendment actually patches, and which it leaves behind.

For each amendment where both documents are in the corpus: the articles the
amending document rewrites, inserts or deletes, set against every article the
original contains. The gap between the two is the point — those articles exist
only in the old document, are still in force, and disappear if it is deleted.

    python -m evals.probes.demo_amendments

Writes results/demo_amendments.md. Read-only.
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Dict, List, Optional, Set, Tuple

from sqlalchemy import select

from app.kb.domain.models import ParentChunk
from app.shared.db import async_session_maker

from ._common import RESULTS_DIR, load_documents, quiet_logs

# "Pasal 5", "Pasal 25A", and the OCR form "Pasal I" for "Pasal 1".
_PASAL_RE = re.compile(r"\bPasal\s+(\d{1,3}[A-Z]?|[IVXLC]{1,4})(?![\w])")

# Clauses that operate on an article, each tagged with what it does.
_OPS: List[Tuple[str, re.Pattern[str]]] = [
    ("rewrites", re.compile(
        r"Ketentuan\s+(?:ayat\s*\(\d+\)\s+)?Pasal\s+(\d{1,3}[A-Z]?|[IVXLC]{1,4})"
        r"[^.]{0,120}?di\s?[ur]bah", re.IGNORECASE)),
    ("deletes", re.compile(
        r"(?:Ketentuan\s+)?Pasal\s+(\d{1,3}[A-Z]?|[IVXLC]{1,4})\s+dihapus",
        re.IGNORECASE)),
]
# Insertions name the *new* article after "yaitu"/"yakni".
_INSERT_RE = re.compile(
    r"[Dd]i\s?antara\s+Pasal\s+[\w]+\s+dan\s+Pasal\s+[\w]+[^.]{0,120}?"
    r"disisipkan[^.]{0,160}", re.IGNORECASE)


def roman_to_int(token: str) -> Optional[int]:
    vals = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100}
    if not token or any(c not in vals for c in token.upper()):
        return None
    total, prev = 0, 0
    for c in reversed(token.upper()):
        v = vals[c]
        total += -v if v < prev else v
        prev = max(prev, v)
    return total


def norm_pasal(token: str) -> str:
    """OCR renders 'Pasal 1' as 'Pasal I'. Fold romans back to digits."""
    token = token.strip()
    if token.isdigit() or (token[:-1].isdigit() and token[-1].isalpha()):
        return token.upper()
    n = roman_to_int(token)
    return str(n) if n else token.upper()


def sort_key(p: str):
    m = re.match(r"^(\d+)([A-Z]?)$", p)
    return (int(m.group(1)), m.group(2)) if m else (10**6, p)


async def doc_text(doc_id: str) -> str:
    async with async_session_maker() as session:
        rows = (await session.execute(
            select(ParentChunk.text).where(ParentChunk.doc_id == doc_id)
            .order_by(ParentChunk.ordinal)
        )).all()
    return "\n".join(r[0] or "" for r in rows)


def articles_in(text: str) -> Set[str]:
    """Every article number the document mentions."""
    return {norm_pasal(m) for m in _PASAL_RE.findall(text)}


def operations(text: str) -> Dict[str, Set[str]]:
    """Which articles the amending document rewrites, deletes or inserts."""
    ops: Dict[str, Set[str]] = {"rewrites": set(), "deletes": set(), "inserts": set()}
    for label, pattern in _OPS:
        for m in pattern.finditer(text):
            ops[label].add(norm_pasal(m.group(1)))
    for m in _INSERT_RE.finditer(text):
        tail = m.group(0)
        after = re.split(r"\byaitu\b|\byakni\b", tail, maxsplit=1)
        if len(after) > 1:
            ops["inserts"].update(norm_pasal(p) for p in _PASAL_RE.findall(after[1]))
    ops["rewrites"] -= ops["deletes"]
    return ops


def fmt(items) -> str:
    return ", ".join(sorted(items, key=sort_key)) if items else "—"


# A document numbering its articles 1..N should yield close to N of them. Well
# below that means the scan lost most headings and the article list cannot be
# enumerated — in this corpus that ranges from 100% (the parking regulation,
# 35 of 35) down to 0.7% (a 2017 scan with 6 "Pasal" mentions and one number).
MIN_COVERAGE = 0.60


def coverage(articles: Set[str]) -> Tuple[float, int, int]:
    """Fraction of the implied article range that was actually recovered."""
    nums = sorted(int(a) for a in articles if a.isdigit())
    if not nums:
        return 0.0, 0, 0
    return len(articles) / nums[-1], len(articles), nums[-1]


async def main() -> None:
    quiet_logs()
    print("Demo — what an amendment patches, and what it leaves behind\n")

    docs = await load_documents()
    summary = json.loads(
        (RESULTS_DIR / "probe3_amendments.summary.json").read_text(encoding="utf-8")
    )
    families = [
        p for p in summary["families_both_sides_in_corpus"]["pairs"]
        if p["article_level"]
    ]
    # Lead with the cases whose originals survived OCR best, so the clearest
    # evidence is first rather than buried behind an unreadable scan.
    order = {"2151-UN40-HK-2019": 0, "7739-UN40-HK-2015": 1, "6489-UN40-HK-2015": 2}
    families.sort(key=lambda f: order.get(
        docs[f["amended"]].title.split(" - ")[0], 9))

    lines: List[str] = [
        "# What an amendment patches, and what it leaves behind\n",
        "Every case below is an amending document whose target is also in our",
        "corpus, and which operates on individual articles rather than replacing",
        "the whole regulation.\n",
        "**Untouched** is the set of articles that appear in the original and are",
        "not mentioned by the amendment. Those articles exist *only* in the old",
        "document. The amendment does not restate them, so deleting the original",
        "deletes law that is still in force.\n",
        "Article numbers are recovered from OCR'd scans: `Pasal I` is folded to",
        "`Pasal 1`, and the odd number is mangled beyond repair (`Pasal 148` in",
        "the parking case is really `Pasal 14B`).\n",
    ]

    for fam in families:
        amender, amended = fam["amender"], fam["amended"]
        a_text, b_text = await doc_text(amender), await doc_text(amended)
        ops = operations(a_text)
        touched = ops["rewrites"] | ops["deletes"] | ops["inserts"]
        original = articles_in(b_text)
        # Inserted articles are new, so they cannot be "untouched" originals.
        untouched = {p for p in original if p not in touched}

        if not touched:
            continue

        cov, found, highest = coverage(original)
        reliable = cov >= MIN_COVERAGE
        a_year = fam["amender_year"] or docs[amender].tahun or "?"

        print(f"\n  {a_year}  {docs[amender].title[:76]}")
        print(f"    patches {fmt(touched)}")
        if reliable:
            print(f"    original: {found} articles (1..{highest}, "
                  f"{cov:.0%} recovered) -> {len(untouched)} untouched")
        else:
            print(f"    original: only {found} of ~{highest} articles recovered "
                  f"({cov:.0%}) - too sparse to enumerate")

        lines += [
            "\n---\n",
            f"## {docs[amender].title}\n",
            f"**Amends** ({fam['amended_year']}): {docs[amended].title}\n",
            "| | Pasal |",
            "|---|---|",
            f"| rewritten | {fmt(ops['rewrites'])} |",
            f"| deleted | {fmt(ops['deletes'])} |",
            f"| inserted (new) | {fmt(ops['inserts'])} |",
        ]
        if reliable:
            lines += [
                f"| **untouched in the original** | **{fmt(untouched)}** |",
                "",
                f"The original numbers its articles up to Pasal {highest} and "
                f"{found} of them ({cov:.0%}) survive text extraction. The "
                f"amendment touches {len(touched)}. The remaining "
                f"**{len(untouched)}** exist only in the "
                f"{fam['amended_year']} document.\n",
            ]
        else:
            lines += [
                "",
                f"*The original's article list cannot be enumerated: only "
                f"{found} of an implied {highest} articles survive text "
                f"extraction ({cov:.0%}), so the untouched set is not "
                f"reported here. The amendment's own clauses are quoted below "
                f"and are unaffected — they name their targets explicitly.*\n",
            ]

        # Quote the operative clauses so the reader can check the extraction.
        quotes: List[str] = []
        for _, pattern in _OPS:
            for m in list(pattern.finditer(a_text))[:2]:
                quotes.append(" ".join(a_text[m.start():m.start() + 170].split()))
        for m in list(_INSERT_RE.finditer(a_text))[:1]:
            quotes.append(" ".join(m.group(0)[:190].split()))
        if quotes:
            lines.append("Operative clauses, as they appear in the scan:\n")
            lines += [f"> {q}\n" for q in quotes[:4]]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "demo_amendments.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  Written to {path}")


if __name__ == "__main__":
    asyncio.run(main())
