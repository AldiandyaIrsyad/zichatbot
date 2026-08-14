"""Audit already-ingested documents: compare parser "input" against stored
chunk "output" to spot-check ingestion quality.

Samples N random completed documents from the production ``pdf_documents``
table, inspects their already-stored ``parent_chunks``/``child_chunks``
(the "output" side, no re-parsing needed), and — unless ``--skip-reparse``
is passed — re-parses each sampled PDF through the real Unstructured
Platform API (the "input" side) to cross-check whether any raw ayat-marker
("(1)", "(2)", ...) elements were misclassified as section-boundary Titles
by the hi_res layout model, the exact failure mode
``app/thesis/chunking/logic.py::_AYAT_MARKER_RE`` now guards against for
future ingestions.

The re-parse path calls a real third-party API once per sampled document
(job-based, can take from tens of seconds to several minutes per PDF) and
is not free — use ``--skip-reparse`` for a fast, DB-only pass.

Usage::

    .venv/bin/python -m tools.audit_ingestion
    .venv/bin/python -m tools.audit_ingestion --n 5 --seed 42
    .venv/bin/python -m tools.audit_ingestion --skip-reparse
"""

from __future__ import annotations

import argparse
import asyncio
import random
import re
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import structlog
from sqlalchemy import func, select

from app.kb.config import get_unstructured_settings
from app.kb.domain.models import ChildChunk, ParentChunk, PDFDocument
from app.kb.infra.unstructured_client import UnstructuredClient
from app.shared.db import async_session_maker
from app.rag.chunking.logic import _AYAT_MARKER_RE

logger = structlog.get_logger(__name__)

# Tolerant of the legacy "[Context: ...]\n\n" prefix that pre-dates this
# session's breadcrumb-tag redesign — already-ingested parent chunks may
# still carry it until re-ingested.
_LEGACY_PREFIX_RE = re.compile(r"^\[Context: [^\]]*\]\n\n")

_BAB_RE = re.compile(r"^BAB\s+[IVXLC]+")
_TINY_BODY_CHARS = 15


def _strip_legacy_prefix(text: str) -> str:
    return _LEGACY_PREFIX_RE.sub("", text, count=1)


@dataclass
class DocFindings:
    doc_id: str
    title: str
    parent_count: int = 0
    child_count: int = 0
    content_type_counts: dict = field(default_factory=dict)
    body_lengths: List[int] = field(default_factory=list)
    mid_article_splits: List[Tuple[int, List[str], str]] = field(default_factory=list)
    duplicate_breadcrumbs: List[Tuple[List[str], int]] = field(default_factory=list)
    depth0_resets: List[Tuple[int, List[str]]] = field(default_factory=list)
    tiny_parents: List[Tuple[int, List[str], int]] = field(default_factory=list)
    raw_ayat_titles: Optional[int] = None  # None = not re-parsed
    reparse_error: Optional[str] = None

    @property
    def has_flags(self) -> bool:
        return bool(
            self.mid_article_splits or self.duplicate_breadcrumbs
            or self.depth0_resets or self.tiny_parents
        )


async def _sample_documents(n: int, seed: Optional[int]) -> List[PDFDocument]:
    async with async_session_maker() as session:
        result = await session.execute(
            select(PDFDocument).where(PDFDocument.ingestion_status == "completed")
        )
        all_docs = list(result.scalars().all())

    if not all_docs:
        return []

    rng = random.Random(seed)
    return rng.sample(all_docs, min(n, len(all_docs)))


def _check_output_side(findings: DocFindings, parents: List[ParentChunk]) -> None:
    seen_breadcrumbs: dict = {}
    prev_depth = 0

    for i, p in enumerate(parents):
        body = _strip_legacy_prefix(p.text).strip()
        findings.body_lengths.append(len(body))
        findings.content_type_counts[p.content_type] = (
            findings.content_type_counts.get(p.content_type, 0) + 1
        )

        # Mid-article split: body starts at ayat (N) where N != 1 — a fresh
        # Pasal section normally opens with chapeau prose or ayat (1).
        m = _AYAT_MARKER_RE.match(body)
        if m and body[1:m.end() - 2].strip() != "1":
            findings.mid_article_splits.append((i, list(p.breadcrumbs or []), body[:80]))

        # Duplicate breadcrumbs: two parents claiming the same section path.
        crumbs_key = tuple(p.breadcrumbs or [])
        if crumbs_key:
            seen_breadcrumbs[crumbs_key] = seen_breadcrumbs.get(crumbs_key, 0) + 1

        # Suspicious depth-0 reset: breadcrumbs collapse to a single
        # non-"BAB ..." crumb right after a deeper section — signature of
        # the heading-stack-pop bug from an ayat marker mistagged as Title.
        depth = len(p.breadcrumbs or [])
        if depth <= 1 and prev_depth >= 2:
            crumb = (p.breadcrumbs or [""])[0] if p.breadcrumbs else ""
            if not _BAB_RE.match(crumb):
                findings.depth0_resets.append((i, list(p.breadcrumbs or [])))
        prev_depth = depth

        # Tiny/gibberish parent body.
        if len(body) < _TINY_BODY_CHARS:
            findings.tiny_parents.append((i, list(p.breadcrumbs or []), len(body)))

    findings.duplicate_breadcrumbs = [
        (list(crumbs), count) for crumbs, count in seen_breadcrumbs.items() if count > 1
    ]


async def _check_input_side(findings: DocFindings, pdf_path: str) -> None:
    settings = get_unstructured_settings()
    parser = UnstructuredClient(
        base_url=settings.base_url,
        extract_images=False,  # audit only needs text/Title elements
        api_key=settings.api_key,
    )
    try:
        start = time.perf_counter()
        elements = await parser.parse_pdf(pdf_path)
        elapsed = time.perf_counter() - start
        raw_ayat_titles = sum(
            1 for el in elements
            if el.element_type == "Title" and _AYAT_MARKER_RE.match(el.text.strip())
        )
        findings.raw_ayat_titles = raw_ayat_titles
        logger.info(
            "audit.reparse.done",
            doc=findings.title,
            elements=len(elements),
            raw_ayat_titles=raw_ayat_titles,
            elapsed_sec=round(elapsed, 1),
        )
    except Exception as exc:
        findings.reparse_error = str(exc)
        logger.warning("audit.reparse.failed", doc=findings.title, error=str(exc))
    finally:
        await parser.close()


async def _audit_document(doc: PDFDocument, skip_reparse: bool) -> DocFindings:
    findings = DocFindings(doc_id=doc.id, title=doc.title or doc.id)

    async with async_session_maker() as session:
        parents = list(
            (
                await session.execute(
                    select(ParentChunk)
                    .where(ParentChunk.doc_id == doc.id)
                    .order_by(ParentChunk.chunk_index)
                )
            ).scalars().all()
        )
        child_count = (
            await session.execute(
                select(func.count()).select_from(ChildChunk).where(ChildChunk.doc_id == doc.id)
            )
        ).scalar_one()

    findings.parent_count = len(parents)
    findings.child_count = child_count
    _check_output_side(findings, parents)

    if not skip_reparse:
        await _check_input_side(findings, doc.pdf_path)

    return findings


def _print_report(all_findings: List[DocFindings]) -> None:
    print("\n" + "=" * 78)
    print(f"INGESTION AUDIT — {len(all_findings)} sampled documents")
    print("=" * 78)

    for f in all_findings:
        print(f"\n--- {f.title} ({f.doc_id}) ---")
        print(f"  parents={f.parent_count}  children={f.child_count}  content_types={f.content_type_counts}")
        if f.body_lengths:
            print(
                f"  body length: min={min(f.body_lengths)} "
                f"max={max(f.body_lengths)} avg={sum(f.body_lengths) // len(f.body_lengths)}"
            )
        if f.raw_ayat_titles is not None:
            print(f"  raw parser: ayat-marker Title elements = {f.raw_ayat_titles}")
        elif f.reparse_error:
            print(f"  raw parser: FAILED ({f.reparse_error})")

        if f.mid_article_splits:
            print(f"  [FLAG] mid-article split candidates ({len(f.mid_article_splits)}):")
            for idx, crumbs, preview in f.mid_article_splits[:5]:
                print(f"    parent#{idx} breadcrumbs={crumbs} body[:80]={preview!r}")
        if f.duplicate_breadcrumbs:
            print(f"  [FLAG] duplicate breadcrumbs ({len(f.duplicate_breadcrumbs)}):")
            for crumbs, count in f.duplicate_breadcrumbs[:5]:
                print(f"    {crumbs} appears in {count} parent chunks")
        if f.depth0_resets:
            print(f"  [FLAG] suspicious depth-0 resets ({len(f.depth0_resets)}):")
            for idx, crumbs in f.depth0_resets[:5]:
                print(f"    parent#{idx} breadcrumbs={crumbs}")
        if f.tiny_parents:
            print(f"  [FLAG] tiny/gibberish parents ({len(f.tiny_parents)}):")
            for idx, crumbs, length in f.tiny_parents[:5]:
                print(f"    parent#{idx} breadcrumbs={crumbs} body_len={length}")
        if not f.has_flags:
            print("  no anomalies flagged")

    flagged = [f for f in all_findings if f.has_flags]
    reparsed = [f for f in all_findings if f.raw_ayat_titles is not None]
    correlated = [
        f for f in reparsed
        if f.raw_ayat_titles and f.raw_ayat_titles > 0 and f.has_flags
    ]

    print("\n" + "=" * 78)
    print("AGGREGATE SUMMARY")
    print("=" * 78)
    print(f"  {len(flagged)}/{len(all_findings)} sampled docs have >=1 output-side anomaly flagged")
    if reparsed:
        with_raw_ayat_titles = sum(1 for f in reparsed if (f.raw_ayat_titles or 0) > 0)
        print(
            f"  {with_raw_ayat_titles}/{len(reparsed)} re-parsed docs have >=1 raw "
            f"ayat-marker Title element in the parser output"
        )
        print(
            f"  {len(correlated)}/{len(reparsed)} re-parsed docs show BOTH a raw "
            f"ayat-marker Title AND a correlated output-side split anomaly"
        )
    print()


async def run_audit(n: int, seed: Optional[int], skip_reparse: bool) -> List[DocFindings]:
    docs = await _sample_documents(n, seed)
    if not docs:
        print("No completed documents found to sample.")
        return []

    print(f"Sampled {len(docs)} completed documents (seed={seed!r}).")
    all_findings: List[DocFindings] = []
    for i, doc in enumerate(docs, start=1):
        print(f"[{i}/{len(docs)}] auditing {doc.title!r}...")
        findings = await _audit_document(doc, skip_reparse=skip_reparse)
        all_findings.append(findings)

    _print_report(all_findings)
    return all_findings


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--n", type=int, default=10, help="Number of documents to sample (default: 10)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible sampling")
    parser.add_argument(
        "--skip-reparse",
        action="store_true",
        help="Skip re-parsing PDFs via the Unstructured API (DB-only, fast, no API cost)",
    )
    args = parser.parse_args()

    asyncio.run(run_audit(n=args.n, seed=args.seed, skip_reparse=args.skip_reparse))


if __name__ == "__main__":
    main()
