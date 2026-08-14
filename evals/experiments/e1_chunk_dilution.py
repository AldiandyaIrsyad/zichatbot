"""E1 — is hierarchical chunking's breadcrumb tag diluting the embedding?

Every child chunk is embedded as ``"BAB II > Pasal 5\\n\\n" + body``
(`logic.py:458`). Two things could go wrong:

1. **Dilution.** The tag is a near-constant string across a section, so chunks
   in the same section drift toward each other and lose discrimination.
2. **Asymmetry — the more interesting one.** A Peraturan is structured
   (BAB / Pasal) and gets rich breadcrumbs. A Keputusan Rektor is a flat decree
   ("MEMUTUSKAN: KESATU / KEDUA / KETIGA") and gets almost none. If the tag
   dilutes, it dilutes *regulations specifically* — the exact class the audit
   found losing to decrees.

The test strips the tag back off, re-embeds, and compares. `_strip_breadcrumb_tag`
from the production chunker is reused so the strip is exactly the inverse of
what ingestion did.

    python -m evals.experiments.e1_chunk_dilution

Writes results/e1_chunk_dilution.json. No LLM, no API cost.
"""

from __future__ import annotations

import asyncio
import json
from collections import defaultdict
from statistics import mean
from typing import Dict, List, Tuple

import numpy as np
from sqlalchemy import select

from app.kb.domain.models import ChildChunk, ParentChunk
from app.shared.db import async_session_maker
from app.rag.chunking.logic import _strip_breadcrumb_tag

from evals.probes._common import RESULTS_DIR, Query, Retriever, cached_dense, quiet_logs, save_dense
from ._bench import QUESTIONS, doc_type


async def pool_chunks(doc_ids: List[str]) -> List[Tuple[str, str, str, str]]:
    """(chunk_id, doc_id, text_with_tag, text_without_tag) for a set of docs."""
    async with async_session_maker() as session:
        rows = (await session.execute(
            select(ChildChunk.id, ChildChunk.doc_id, ChildChunk.text,
                   ParentChunk.breadcrumbs)
            .join(ParentChunk, ParentChunk.id == ChildChunk.parent_chunk_id)
            .where(ChildChunk.doc_id.in_(doc_ids))
        )).all()
    out = []
    for cid, did, text, crumbs in rows:
        text = text or ""
        out.append((cid, did, text, _strip_breadcrumb_tag(text, list(crumbs or []))))
    return out


async def main() -> None:
    quiet_logs()
    print("E1 — does the breadcrumb tag dilute the embedding?\n")

    retriever = Retriever()
    queries = [Query(qid=f"e{i}", text=q) for i, (q, _) in enumerate(QUESTIONS)]
    cands = await retriever.retrieve(queries)
    doc_ids = sorted({c.doc_id for cs in cands.values() for c in cs})

    from evals.probes._common import load_documents

    docs = await load_documents()
    titles = {d.doc_id: d.title for d in docs.values()}

    chunks = await pool_chunks(doc_ids)
    print(f"  pool: {len(chunks)} chunks from {len(doc_ids)} documents")

    tagged = sum(1 for _, _, a, b in chunks if a != b)
    print(f"  chunks that actually carry a tag: {tagged} ({tagged/len(chunks):.0%})")

    # How much tag does each document class carry?
    by_type: Dict[str, List[float]] = defaultdict(list)
    for _, did, a, b in chunks:
        share = 1 - (len(b) / len(a)) if a else 0.0
        by_type[doc_type(titles.get(did, ""))].append(share)
    print("\n  share of a chunk's characters that are breadcrumb tag:")
    for t, vals in sorted(by_type.items()):
        carried = sum(1 for v in vals if v > 0) / len(vals)
        print(f"    {t:10s} mean {mean(vals):.1%} of chunk, "
              f"{carried:.0%} of chunks carry one  (n={len(vals)})")

    # Embed both ways.
    cache = cached_dense("e1_pool")
    ids = [c[0] for c in chunks]
    want = {f"{k}|{i}" for k in ("with", "without") for i in ids}
    if cache is None or set(cache[0]) != want:
        print(f"\n  embedding {2*len(chunks)} variants ...")
        keys = [f"with|{c[0]}" for c in chunks] + [f"without|{c[0]}" for c in chunks]
        payload = [c[2] for c in chunks] + [c[3] for c in chunks]
        vecs = await retriever.embed(payload)
        save_dense("e1_pool", keys, vecs)
        cache = (keys, np.asarray(vecs, dtype="float32"))
    keys, mat = cache
    mat = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12)
    idx = {k: n for n, k in enumerate(keys)}

    qv = np.asarray(await retriever.embed([q.text for q in queries]), dtype="float32")
    qv /= np.linalg.norm(qv, axis=1, keepdims=True) + 1e-12
    await retriever.close()

    owner = {c[0]: c[1] for c in chunks}

    # Where does the expected document land, with and without the tag?
    print(f"\n  {'question':52s} {'with tag':>9s} {'without':>9s}")
    ranks: Dict[str, List[int]] = {"with": [], "without": []}
    per_question = []
    for n, (qtext, expected) in enumerate(QUESTIONS):
        row = {"question": qtext, "expected": expected}
        for variant in ("with", "without"):
            sims = np.stack([mat[idx[f"{variant}|{i}"]] for i in ids]) @ qv[n]
            order = np.argsort(-sims)
            seen: List[str] = []
            for j in order:
                d = owner[ids[j]]
                if d not in seen:
                    seen.append(d)
            pos = next((k + 1 for k, d in enumerate(seen)
                        if expected and titles.get(d, "").startswith(expected)), None)
            row[variant] = pos
            row[f"{variant}_top"] = titles.get(seen[0], "")[:60]
            row[f"{variant}_top_type"] = doc_type(titles.get(seen[0], ""))
            if pos:
                ranks[variant].append(pos)
        per_question.append(row)
        w = f"#{row['with']}" if row["with"] else "—"
        wo = f"#{row['without']}" if row["without"] else "—"
        print(f"  {qtext[:52]:52s} {w:>9s} {wo:>9s}")

    kep = {v: sum(1 for r in per_question if r[f"{v}_top_type"] == "Keputusan")
           for v in ("with", "without")}
    print(f"\n  decree on top:  with tag {kep['with']}/10   without tag {kep['without']}/10")
    if ranks["with"] and ranks["without"]:
        print(f"  mean rank of expected doc (where it has one): "
              f"with {mean(ranks['with']):.1f}  without {mean(ranks['without']):.1f}")

    # Dilution proper: does the tag pull a document's chunks together?
    bydoc: Dict[str, List[str]] = defaultdict(list)
    for cid, did, _, _ in chunks:
        bydoc[did].append(cid)
    intra: Dict[str, List[float]] = {"with": [], "without": []}
    for did, cs in bydoc.items():
        if not 8 <= len(cs) <= 300:
            continue
        for variant in ("with", "without"):
            M = np.stack([mat[idx[f"{variant}|{c}"]] for c in cs])
            S = M @ M.T
            iu = np.triu_indices(len(cs), 1)
            intra[variant].append(float(S[iu].mean()))
    print(f"\n  intra-document cosine: with tag {mean(intra['with']):.4f}  "
          f"without tag {mean(intra['without']):.4f}  "
          f"(delta {mean(intra['with']) - mean(intra['without']):+.4f})")

    summary = {
        "pool_chunks": len(chunks),
        "pool_documents": len(doc_ids),
        "chunks_carrying_tag": tagged,
        "tag_share_by_doc_type": {
            t: {"mean_share_of_chunk": round(mean(v), 4),
                "pct_chunks_with_tag": round(sum(1 for x in v if x > 0) / len(v), 4),
                "n": len(v)}
            for t, v in by_type.items()
        },
        "top_source_keputusan": kep,
        "expected_doc_rank": {
            "with_tag": ranks["with"], "without_tag": ranks["without"],
            "mean_with": round(mean(ranks["with"]), 2) if ranks["with"] else None,
            "mean_without": round(mean(ranks["without"]), 2) if ranks["without"] else None,
        },
        "intra_document_cosine": {
            "with_tag": round(mean(intra["with"]), 4),
            "without_tag": round(mean(intra["without"]), 4),
        },
        "per_question": per_question,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "e1_chunk_dilution.json"
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  Written to {path}")


if __name__ == "__main__":
    asyncio.run(main())
