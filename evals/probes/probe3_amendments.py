"""Probe 3 — do documents amend each other, and are the amendments partial?

The question behind this is "why can't we just delete superseded documents".
If an amending instrument replaces a whole regulation, deleting the old one is
safe. If it only rewrites Pasal 16 and inserts Pasal 25A, then the old document
still carries every article the amendment did not touch, and deleting it loses
law that is still in force.

Two layers:

* **title layer** — every document whose title declares it an amendment
  ("Perubahan Ketiga Atas ...") or a repeal ("Pencabutan ...").
* **body layer** — the operative clauses in ``parent_chunks.text``.

The body layer deliberately does **not** match ``sebagaimana telah diubah``.
That phrase appears in the *considerans* ("Mengingat: Undang-Undang Nomor 12
Tahun 2012 ... sebagaimana telah diubah dengan ...") of nearly every document
in the corpus and says nothing about what *this* document amends. Matching it
would report ~879 of 922 documents as amenders. Only clauses that operate on a
specific article count.

    python -m evals.probes.probe3_amendments

Writes results/probe3_amendments.csv and results/probe3_amendments.summary.json.
Read-only: SELECT against Postgres, no models, no network.
"""

from __future__ import annotations

import asyncio
import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

from sqlalchemy import select

from app.kb.domain.models import ParentChunk
from app.shared.db import async_session_maker

from ._common import DocMeta, load_documents, quiet_logs, write_results

# --- OCR-tolerant fragments -------------------------------------------------
# The scans mangle text in ways a clean regex misses: "schingga" for "sehingga",
# "Pasal I" for "Pasal 1", "diantara" closed up. Each fragment absorbs one.
_SEHINGGA = r"s[ce]h?ingga"
_PASAL_NO = r"[IVXLC0-9]+\s?[A-Z]?"          # 1, 25A, I (OCR for 1)
_DIANTARA = r"[Dd]i\s?antara"

PATTERNS: List[Tuple[str, re.Pattern[str]]] = [
    (
        "partial_amend",
        re.compile(
            rf"Ketentuan\s+(?:ayat\s*\(\d+\)\s+)?Pasal\s+{_PASAL_NO}[^.]{{0,120}}?di\s?[ur]bah",
            re.IGNORECASE,
        ),
    ),
    (
        "partial_amend",
        re.compile(rf"di\s?[ur]bah\s+{_SEHINGGA}\s+berbunyi", re.IGNORECASE),
    ),
    (
        "insert",
        re.compile(
            rf"{_DIANTARA}\s+Pasal\s+{_PASAL_NO}\s+dan\s+Pasal\s+{_PASAL_NO}[^.]{{0,80}}?disisipkan",
            re.IGNORECASE,
        ),
    ),
    (
        "insert",
        re.compile(r"disisipkan\s+\d+\s*\([a-z]+\)\s*(?:pasal|ayat)", re.IGNORECASE),
    ),
    (
        "delete",
        re.compile(rf"Pasal\s+{_PASAL_NO}\s+dihapus", re.IGNORECASE),
    ),
    (
        "delete",
        re.compile(r"ayat\s*\(\d+\)[^.]{0,60}?dihapus", re.IGNORECASE),
    ),
    (
        "full_repeal",
        re.compile(r"dicabut\s+dan\s+dinyatakan\s+tidak\s+berlaku", re.IGNORECASE),
    ),
]

# The considerans boilerplate we must NOT count, kept only to measure it.
BOILERPLATE = re.compile(r"sebagaimana\s+telah\s+di\s?[ur]bah", re.IGNORECASE)

# Marks a chunk as considerans / citation context rather than operative body.
CONSIDERANS = re.compile(r"\b(Mengingat|Menimbang)\s*:", re.IGNORECASE)

# Extraction must be tighter than matching: "Pasal 5 diubah" has to yield "5",
# not "5 d". A suffix letter is only part of the number when it is attached
# ("Pasal 25A"), so no space is allowed before it.
_AFFECTED_RE = re.compile(r"Pasal\s+(\d+[A-Z]?|[IVXLC]+)(?![\w])")
_AYAT_RE = re.compile(r"ayat\s*\((\d+[a-z]?)\)", re.IGNORECASE)

# "Nomor 3233-UN40-KP.06.00-2025" / "03/PER/MWA UPI/2015" -> canonical form
_CODE_CHARS = re.compile(r"[^0-9A-Za-z.]+")


def canon_code(raw: Optional[str]) -> str:
    """Normalise a document code so a title reference can be matched to a row.

    Titles write ``3233-UN40-KP.06.00-2025`` while ``description`` writes
    ``3233/UN40/KP.06.00/2025``; both collapse to the same token here.
    """
    if not raw:
        return ""
    m = re.search(r"(\d[\w./\- ]*UN40[\w./\- ]*\d{4})", raw)
    token = m.group(1) if m else raw
    return _CODE_CHARS.sub("/", token.strip()).strip("/").upper()


def _desc_code(description: str) -> str:
    m = re.search(r"Code:\s*([^\s,;]+)", description or "")
    return canon_code(m.group(1)) if m else ""


def snippet(text: str, at: int, width: int = 160) -> str:
    lo = max(0, at - width // 3)
    return " ".join(text[lo : lo + width].split())


async def scan_bodies() -> Tuple[Dict[str, List[dict]], Dict[str, int], int]:
    """Find operative amendment clauses in every parent chunk.

    Returns (hits by doc, boilerplate doc counts, total chunks scanned).
    """
    hits: Dict[str, List[dict]] = defaultdict(list)
    boiler: Dict[str, int] = defaultdict(int)
    scanned = 0

    async with async_session_maker() as session:
        result = await session.stream(
            select(ParentChunk.doc_id, ParentChunk.id, ParentChunk.text)
        )
        async for doc_id, chunk_id, text in result:
            scanned += 1
            if not text:
                continue
            if BOILERPLATE.search(text):
                boiler[doc_id] += 1
            is_considerans = bool(CONSIDERANS.search(text))

            for relation, pattern in PATTERNS:
                for m in pattern.finditer(text):
                    # A citation inside Mengingat/Menimbang describes some other
                    # law's history, not an operation this document performs.
                    if is_considerans and relation != "full_repeal":
                        continue
                    window = text[m.start() : m.start() + 400]
                    hits[doc_id].append({
                        "relation": relation,
                        "chunk_id": chunk_id,
                        "affected_pasal": ";".join(
                            dict.fromkeys(_AFFECTED_RE.findall(window))
                        )[:60],
                        "affected_ayat": ";".join(
                            dict.fromkeys(_AYAT_RE.findall(window))
                        )[:40],
                        "evidence": snippet(text, m.start()),
                    })
    return hits, boiler, scanned


def link_families(docs: Dict[str, DocMeta]) -> Dict[str, Optional[str]]:
    """Resolve each amending document's title reference to a doc_id we hold.

    Returns amender doc_id -> amended doc_id (or None when the target is not in
    the corpus, which is common: many amend national law we never ingested).
    """
    by_code: Dict[str, str] = {}
    for d in docs.values():
        code = canon_code(d.nomor) if d.nomor else ""
        if code:
            by_code.setdefault(code, d.doc_id)

    async_note = {}
    for d in docs.values():
        if not d.is_perubahan or not d.target_ref:
            continue
        async_note[d.doc_id] = by_code.get(canon_code(d.target_ref))
    return async_note


async def main() -> None:
    quiet_logs()
    print("Probe 3 — amendment survey\n")

    docs = await load_documents()
    print(f"  corpus: {len(docs)} documents")

    # Enrich the code index with the description code, which is populated for
    # the whole corpus even where the title prefix does not parse.
    async with async_session_maker() as session:
        from app.kb.domain.models import PDFDocument

        rows = (await session.execute(
            select(PDFDocument.id, PDFDocument.description)
        )).all()
    desc_code = {r[0]: _desc_code(r[1] or "") for r in rows}

    hits, boiler, scanned = await scan_bodies()
    print(f"  scanned {scanned} parent chunks")

    # --- title layer --------------------------------------------------------
    title_amenders = {d.doc_id for d in docs.values() if d.is_perubahan}
    title_repealers = {d.doc_id for d in docs.values() if d.is_pencabutan}

    # --- family linkage -----------------------------------------------------
    by_code: Dict[str, str] = {}
    for doc_id, code in desc_code.items():
        if code:
            by_code.setdefault(code, doc_id)
    for d in docs.values():
        code = canon_code(d.nomor) if d.nomor else ""
        if code:
            by_code.setdefault(code, d.doc_id)

    families: Dict[str, Optional[str]] = {}
    for d in docs.values():
        if d.is_perubahan and d.target_ref:
            families[d.doc_id] = by_code.get(canon_code(d.target_ref))

    resolved = {a: t for a, t in families.items() if t}

    # --- rows ---------------------------------------------------------------
    rows_out: List[dict] = []
    for doc_id, doc_hits in sorted(hits.items()):
        d = docs.get(doc_id)
        if d is None:
            continue
        for h in doc_hits:
            rows_out.append({
                "amender_doc_id": doc_id,
                "amender_title": d.title,
                "amender_year": d.tahun or "",
                "title_declares_perubahan": int(d.is_perubahan),
                "relation": h["relation"],
                "affected_pasal": h["affected_pasal"],
                "affected_ayat": h["affected_ayat"],
                "target_ref": d.target_ref or "",
                "target_doc_id": families.get(doc_id) or "",
                "evidence_snippet": h["evidence"],
            })

    # --- aggregate ----------------------------------------------------------
    docs_by_relation: Dict[str, set] = defaultdict(set)
    for r in rows_out:
        docs_by_relation[r["relation"]].add(r["amender_doc_id"])

    article_level = (
        docs_by_relation["partial_amend"]
        | docs_by_relation["insert"]
        | docs_by_relation["delete"]
    )
    whole_doc = docs_by_relation["full_repeal"]
    both = article_level & whole_doc

    # A title can declare "Perubahan Atas" without performing article surgery:
    # most Keputusan Rektor amendments replace an attached list of names
    # wholesale. Separating the two matters, because only the article-level
    # kind makes the superseded document partly-still-valid.
    title_with_article_ops = title_amenders & article_level
    title_without_ops = title_amenders - set(hits.keys())

    # Families where both sides are in the corpus, split by what the amender does.
    family_article_level = {a for a in resolved if a in article_level}

    summary = {
        "corpus_documents": len(docs),
        "parent_chunks_scanned": scanned,
        "title_layer": {
            "declares_perubahan": len(title_amenders),
            "declares_pencabutan": len(title_repealers),
            "target_resolved_in_corpus": len(resolved),
            "target_outside_corpus": len(families) - len(resolved),
        },
        "body_layer": {
            "documents_with_operative_clause": len(set(r["amender_doc_id"] for r in rows_out)),
            "clauses_found": len(rows_out),
            "documents_by_relation": {k: len(v) for k, v in sorted(docs_by_relation.items())},
            "clauses_by_relation": dict(Counter(r["relation"] for r in rows_out)),
        },
        "partial_vs_whole": {
            "article_level_only": len(article_level - whole_doc),
            "whole_document_repeal_only": len(whole_doc - article_level),
            "both": len(both),
            "article_level_share": round(
                len(article_level) / max(1, len(article_level | whole_doc)), 3
            ),
        },
        "title_vs_body": {
            "title_perubahan_with_article_ops": len(title_with_article_ops),
            "title_perubahan_without_any_clause": len(title_without_ops),
            "note": (
                "A 'Perubahan Atas' title without an article-level clause is "
                "usually a Keputusan Rektor replacing an attached list "
                "wholesale. Only the article-level kind leaves the superseded "
                "document partly in force."
            ),
        },
        "families_both_sides_in_corpus": {
            "total": len(resolved),
            "amender_does_article_surgery": len(family_article_level),
            "pairs": [
                {
                    "amender": a,
                    "amender_title": docs[a].title,
                    "amender_year": docs[a].tahun,
                    "amended": t,
                    "amended_title": docs[t].title,
                    "amended_year": docs[t].tahun,
                    "article_level": a in article_level,
                }
                for a, t in sorted(resolved.items())
            ],
        },
        "boilerplate_control": {
            "documents_matching_sebagaimana_telah_diubah": len(boiler),
            "note": (
                "This is the considerans citation count. It is reported only to "
                "show what the operative-clause filter excludes; it is not a "
                "count of amending instruments."
            ),
        },
    }

    write_results("probe3_amendments", rows_out, summary)

    # --- report -------------------------------------------------------------
    tl, bl, pw = summary["title_layer"], summary["body_layer"], summary["partial_vs_whole"]
    print(f"\n  Title layer")
    print(f"    declares 'Perubahan Atas'      : {tl['declares_perubahan']}")
    print(f"    declares 'Pencabutan'          : {tl['declares_pencabutan']}")
    print(f"    amended doc is in our corpus   : {tl['target_resolved_in_corpus']}")
    print(f"    amended doc is NOT in corpus   : {tl['target_outside_corpus']}")
    print(f"\n  Body layer (operative clauses only)")
    print(f"    documents with a clause        : {bl['documents_with_operative_clause']}")
    print(f"    clauses found                  : {bl['clauses_found']}")
    for k, v in bl["documents_by_relation"].items():
        print(f"      {k:16s} {v:4d} documents")
    print(f"\n  Partial vs whole-document")
    print(f"    article-level only             : {pw['article_level_only']}")
    print(f"    whole-document repeal only     : {pw['whole_document_repeal_only']}")
    print(f"    both                           : {pw['both']}")
    print(f"    article-level share            : {pw['article_level_share']:.1%}")
    tv = summary["title_vs_body"]
    print(f"\n  Title says 'Perubahan' ...")
    print(f"    and body does article surgery  : {tv['title_perubahan_with_article_ops']}")
    print(f"    but body has no clause at all  : {tv['title_perubahan_without_any_clause']}")
    fam = summary["families_both_sides_in_corpus"]
    print(f"\n  Amender + amended both in corpus : {fam['total']}")
    print(f"    of which article-level         : {fam['amender_does_article_surgery']}")
    print(f"\n  Control: 'sebagaimana telah diubah' (considerans boilerplate, excluded)")
    print(f"    would have matched             : {summary['boilerplate_control']['documents_matching_sebagaimana_telah_diubah']} documents")

    print("\n  Sample operative clauses:")
    shown = set()
    for r in rows_out:
        if r["relation"] in shown or not r["affected_pasal"]:
            continue
        shown.add(r["relation"])
        print(f"\n    [{r['relation']}] {r['amender_title'][:72]}")
        print(f"      Pasal {r['affected_pasal']}  ->  {r['evidence_snippet'][:150]}")
        if len(shown) == 4:
            break


if __name__ == "__main__":
    asyncio.run(main())
