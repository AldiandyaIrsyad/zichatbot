"""Demo — the UKT case, before and after the title fix.

Ask "Berapa biaya UKT?" today and the top hit is a decree about outbound
student mobility that happens to contain "Rp 500.000 per semester" — an
administrative fee for one exchange programme, not the tuition tariff. The
document that actually sets UKT by income bracket ("Kelompok Tarif Uang Kuliah
Tunggal") does not appear at all.

Nothing about the chunk is wrong. It really does say 500.000 and it really is
about a per-semester payment. What the ranker never sees is that the *document*
is about sending four students to Sookmyung Women's University.

This runs the same query through the current ranking (A) and through variant B
(the reranker is shown "title\\n" + chunk) and prints them side by side.

    python -m evals.probes.demo_title

Writes results/demo_title.md. Read-only.
"""

from __future__ import annotations

import asyncio
from typing import Dict, List, Sequence

from ._common import (
    RESULTS_DIR,
    Candidate,
    Query,
    Retriever,
    get_reranker,
    load_child_texts,
    load_documents,
    quiet_logs,
)

# Tuition questions a student would actually type, plus the two registers from
# LLM_Generated_studentQ.csv that ask the same thing.
DEMO_QUERIES = [
    "Berapa biaya UKT?",
    "Biaya kuliah per semester berapa?",
    "Berapa tarif UKT yang harus saya bayar?",
    "UKT saya masuk golongan berapa?",
    "Biaya smstr brp ya?",
]

TOP_N = 3


def rank(cands: Sequence[Candidate], scores: Sequence[float]) -> List[int]:
    return sorted(range(len(cands)), key=lambda i: scores[i], reverse=True)


def top_docs(cands, scores, order, titles, texts, n=TOP_N):
    """First n distinct documents, with the chunk that put each one there."""
    out, seen = [], set()
    for i in order:
        doc = cands[i].doc_id
        if doc in seen:
            continue
        seen.add(doc)
        out.append({
            "score": scores[i],
            "doc_id": doc,
            "title": titles.get(doc, ""),
            "snippet": " ".join(texts.get(cands[i].chunk_id, "").split()),
        })
        if len(out) == n:
            break
    return out


async def main() -> None:
    quiet_logs()
    print("Demo — the UKT case\n")

    docs = await load_documents()
    titles = {d.doc_id: d.title for d in docs.values()}

    retriever = Retriever()
    reranker = get_reranker()
    queries = [Query(qid=f"demo{i}", text=t) for i, t in enumerate(DEMO_QUERIES)]

    try:
        cands_by_qid = await retriever.retrieve(queries)
        chunk_ids = sorted({c.chunk_id for cs in cands_by_qid.values() for c in cs})
        texts = await load_child_texts(chunk_ids)

        lines: List[str] = [
            "# Demo — the UKT case, before and after the title fix\n",
            "For each question: the top 3 **documents** returned by the current",
            "ranking (A) and by variant B, which shows the reranker the document",
            "title alongside the chunk. Nothing else changes — same retrieval,",
            "same candidates, same cross-encoder.\n",
        ]

        for q in queries:
            cands = cands_by_qid[q.qid]
            plain = [texts.get(c.chunk_id, "") for c in cands]
            titled = [f"{titles.get(c.doc_id, '')}\n{t}" for c, t in zip(cands, plain)]

            res_a = await reranker.rerank(query=q.text, documents=plain)
            res_b = await reranker.rerank(query=q.text, documents=titled)
            s_a, s_b = [0.0] * len(cands), [0.0] * len(cands)
            for r in res_a:
                s_a[r.index] = float(r.score)
            for r in res_b:
                s_b[r.index] = float(r.score)

            a = top_docs(cands, s_a, rank(cands, s_a), titles, texts)
            b = top_docs(cands, s_b, rank(cands, s_b), titles, texts)

            print(f"\n### {q.text}")
            print(f"  BEFORE  {a[0]['title'][:74]}")
            print(f"          {a[0]['snippet'][:100]}")
            print(f"  AFTER   {b[0]['title'][:74]}")
            print(f"          {b[0]['snippet'][:100]}")
            print(f"  {'changed' if a[0]['doc_id'] != b[0]['doc_id'] else 'unchanged'}")

            lines.append(f"\n---\n\n## {q.text}\n")
            for label, ranked in (("Before (current)", a), ("After (variant B)", b)):
                lines.append(f"**{label}**\n")
                for n, d in enumerate(ranked, 1):
                    lines.append(f"{n}. `{d['score']:+.2f}` **{d['title']}**")
                    lines.append(f"   > {d['snippet'][:230]}")
                lines.append("")
            lines.append(
                "Top document changed: "
                f"**{'yes' if a[0]['doc_id'] != b[0]['doc_id'] else 'no'}**\n"
            )

        # --- why the cheap fix is not enough, and what would be ------------
        lines += await diagnose(queries, cands_by_qid, titles, texts, retriever)

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        path = RESULTS_DIR / "demo_title.md"
        path.write_text("\n".join(lines), encoding="utf-8")
        print(f"\n  Written to {path}")
    finally:
        await retriever.close()
        await reranker.close()


GOLD_TITLE_PREFIX = "003 Tahun 2022"   # the document that actually sets UKT


async def diagnose(queries, cands_by_qid, titles, texts, retriever) -> List[str]:
    """Is the right document missing from retrieval, or just ranked badly?

    These are different bugs with different fixes, and the UKT case turns out
    to have both. Then: simulate embedding "title + chunk" instead of "chunk",
    over the chunks of every document in the candidate pool, and see where the
    right document lands. This is the dense half of retrieval only — no sparse,
    no RRF — but it is enough to tell whether the title belongs in the vector.
    """
    import numpy as np
    from sqlalchemy import select

    from app.kb.domain.models import ChildChunk
    from app.shared.db import async_session_maker

    from ._common import CACHE_DIR, cached_dense, save_dense

    gold = {i for i, t in titles.items() if t.startswith(GOLD_TITLE_PREFIX)}
    pool_docs = sorted({c.doc_id for cs in cands_by_qid.values() for c in cs} | gold)

    async with async_session_maker() as session:
        rows = (await session.execute(
            select(ChildChunk.id, ChildChunk.doc_id, ChildChunk.text)
            .where(ChildChunk.doc_id.in_(pool_docs))
        )).all()
    ids = [r[0] for r in rows]
    owner = {r[0]: r[1] for r in rows}
    body = {r[0]: r[2] or "" for r in rows}

    print(f"\n  simulating over {len(rows)} chunks from {len(pool_docs)} documents")

    cache = cached_dense("demo_pool")
    if cache is None or set(cache[0]) != set(f"{k}|{v}" for k in ("cur", "tit") for v in ids):
        keys = [f"cur|{i}" for i in ids] + [f"tit|{i}" for i in ids]
        payload = [body[i] for i in ids] + [
            f"{titles.get(owner[i], '')}\n{body[i]}" for i in ids
        ]
        vecs = await retriever.embed(payload)
        save_dense("demo_pool", keys, vecs)
        cache = (keys, np.asarray(vecs, dtype="float32"))
    keys, mat = cache
    mat = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12)
    idx = {k: n for n, k in enumerate(keys)}
    cur = np.stack([mat[idx[f"cur|{i}"]] for i in ids])
    tit = np.stack([mat[idx[f"tit|{i}"]] for i in ids])

    qv = await retriever.embed([q.text for q in queries])
    qv = np.asarray(qv, dtype="float32")
    qv = qv / (np.linalg.norm(qv, axis=1, keepdims=True) + 1e-12)

    out = [
        "\n---\n\n## Why variant B does not fix this, and what would\n",
        "Two different bugs are hiding in the UKT case.\n",
        "| question | right doc in top-50 today? | rank under A | rank under B |",
        "|---|---|---|---|",
    ]
    print("\n  retrieval vs ranking:")
    for q in queries:
        cands = cands_by_qid[q.qid]
        present = [n for n, c in enumerate(cands) if c.doc_id in gold]
        if not present:
            out.append(f"| {q.text} | **no** | — | — |")
            print(f"    {q.text[:44]:46s} NOT RETRIEVED")
            continue
        ranks = []
        for label, docs_in in (("A", [texts.get(c.chunk_id, "") for c in cands]),
                               ("B", [f"{titles.get(c.doc_id, '')}\n{texts.get(c.chunk_id, '')}"
                                      for c in cands])):
            rr = get_reranker()
            res = await rr.rerank(query=q.text, documents=docs_in)
            await rr.close()
            sc = [0.0] * len(cands)
            for x in res:
                sc[x.index] = float(x.score)
            seen: List[str] = []
            for i in sorted(range(len(cands)), key=lambda i: sc[i], reverse=True):
                if cands[i].doc_id not in seen:
                    seen.append(cands[i].doc_id)
            ranks.append(next((n + 1 for n, d in enumerate(seen) if d in gold), None))
        out.append(f"| {q.text} | yes | #{ranks[0]} | #{ranks[1]} |")
        print(f"    {q.text[:44]:46s} in candidates, A #{ranks[0]}, B #{ranks[1]}")

    out += [
        "",
        "So there are two failures, not one. For the colloquial phrasings the",
        "right document is **not retrieved at all** — no reranking change can",
        "reach it. For the rest it *is* retrieved but the cross-encoder ranks it",
        "near the bottom, because the outbound decrees' chunks match \"biaya ...",
        "per semester\" more literally than the tariff table does.\n",
        "### Simulating the title inside the embedding\n",
        "Dense-only ranking over every chunk of the "
        f"{len(pool_docs)} candidate documents ({len(rows)} chunks), embedded",
        "two ways: as today (`breadcrumbs + body`) and with the document title",
        "prepended. Position of the correct UKT document:\n",
        "| question | today | title in embedding |",
        "|---|---|---|",
    ]
    print("\n  dense-only simulation, rank of the correct UKT document:")
    gold_mask = np.array([owner[i] in gold for i in ids])
    for n, q in enumerate(queries):
        row = []
        for mtx in (cur, tit):
            sims = mtx @ qv[n]
            order = np.argsort(-sims)
            seen: List[str] = []
            for j in order:
                d = owner[ids[j]]
                if d not in seen:
                    seen.append(d)
                if len(seen) >= 40:
                    break
            row.append(next((k + 1 for k, d in enumerate(seen) if d in gold), None))
        out.append(f"| {q.text} | #{row[0]} | #{row[1]} |")
        print(f"    {q.text[:44]:46s} {row[0]} -> {row[1]}")

    out += [
        "",
        "Read this as direction, not as a benchmark: it is dense similarity",
        "only, with no sparse channel and no RRF, over a pool of 44 documents",
        "rather than 922. What it does show is whether the title carries the",
        "signal that pulls the tariff document up — which is the question the",
        "cheap reranker variant cannot answer.\n",
    ]
    return out


if __name__ == "__main__":
    asyncio.run(main())
