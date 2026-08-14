"""Build the seeded 300-document RQ1 collection manifest."""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from sqlalchemy import select

from app.kb.domain.models import PDFDocument
from app.shared.db import async_session_maker


def title_stratum(title: str) -> str:
    text = title.lower()
    if "keputusan rektor" in text or re.search(r"\bsk rektor\b", text):
        return "rector_decision"
    if "peraturan rektor" in text:
        return "rector_regulation"
    if "majelis wali amanat" in text or "mwa" in text:
        return "mwa"
    if "senat akademik" in text:
        return "academic_senate"
    if "pedoman" in text or "panduan" in text:
        return "guideline"
    return "other"


async def build(subset_a: str, output: str, seed: int = 42) -> None:
    with open(subset_a, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    gold_ids = sorted({r["source_doc_id"] for r in rows if r["source_doc_id"] != "NONE"})
    if len(gold_ids) != 115:
        raise AssertionError(f"expected 115 unique gold documents, got {len(gold_ids)}")

    async with async_session_maker() as session:
        docs = list((await session.execute(
            select(PDFDocument).where(PDFDocument.active.is_(True))
        )).scalars())
    by_id = {str(d.id): d for d in docs}
    missing = sorted(set(gold_ids) - set(by_id))
    if missing:
        raise AssertionError(f"gold documents absent/inactive: {missing}")

    rng = random.Random(seed)
    candidates: Dict[str, List[PDFDocument]] = defaultdict(list)
    for doc in docs:
        if str(doc.id) not in gold_ids:
            candidates[title_stratum(doc.title or "")].append(doc)
    for group in candidates.values():
        rng.shuffle(group)

    # Seeded round-robin preserves representation from every available title
    # stratum without making corpus-year metadata assumptions.
    distractors: List[PDFDocument] = []
    strata = sorted(candidates)
    while len(distractors) < 185:
        progressed = False
        for stratum in strata:
            group = candidates[stratum]
            if group and len(distractors) < 185:
                distractors.append(group.pop())
                progressed = True
        if not progressed:
            raise AssertionError("fewer than 185 eligible distractors")

    manifest = []
    for doc_id in gold_ids:
        doc = by_id[doc_id]
        manifest.append({
            "doc_id": doc_id, "title": doc.title or "", "role": "gold",
            "stratum": title_stratum(doc.title or ""),
        })
    for doc in distractors:
        manifest.append({
            "doc_id": str(doc.id), "title": doc.title or "", "role": "distractor",
            "stratum": title_stratum(doc.title or ""),
        })
    if len(manifest) != 300 or len({r["doc_id"] for r in manifest}) != 300:
        raise AssertionError("manifest must contain exactly 300 unique documents")

    out = Path(output); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["doc_id", "title", "role", "stratum"])
        writer.writeheader(); writer.writerows(manifest)
    meta = {
        "version": "v2", "seed": seed, "documents": 300,
        "gold_documents": 115, "distractors": 185,
        "subset_a": subset_a,
    }
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset-a", default="evals/data/subset_a.csv")
    parser.add_argument("--output", default="evals/data/rq1_manifest.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    asyncio.run(build(args.subset_a, args.output, args.seed))

if __name__ == "__main__":
    main()
