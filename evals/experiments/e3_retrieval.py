"""E3/E4/E5 — three retrieval interventions on the ten audited questions.

E3  document-type prior   down-weight Keputusan Rektor (administrative
                          decisions about named people) for general questions
E4  document-title channel  rank all 922 titles, pull the best-matching
                          documents' chunks into the pool, fuse by RRF
E5  HyDE                  hypothetical-answer expansion, dense channel only,
                          fused exactly as search_service would

Scored on two objective measures that need no judge:

* **decree on top** — how often the winning document is a Keputusan. Every
  mismatch in the audit was one, so this is the failure rate proxy.
* **expected document first** — for the five questions where an obviously
  correct document exists in the corpus.

    python -m evals.experiments.e3_retrieval

Writes results/e3_retrieval.json. HyDE spends API credit; the rest is local.
"""

from __future__ import annotations

import asyncio
import json
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sqlalchemy import select

from app.kb.domain.models import ChildChunk
from app.shared.db import async_session_maker

from evals.probes._common import (
    RESULTS_DIR,
    Candidate,
    Query,
    cached_dense,
    quiet_logs,
    save_dense,
)
from ._bench import MODEL, QUESTIONS, Bench, baseline, doc_type, summarise

RRF_K = 60
TITLE_TOP_DOCS = 15         # documents pulled in by title match. 5 was too
                            # tight: the correct UKT document ranks #6 by title
                            # for one of these questions, so a top-5 cut lost
                            # exactly the case the channel exists to fix.
TITLE_CHUNKS_PER_DOC = 30   # of their chunks, reranked against the query
KEPUTUSAN_PENALTY = 0.5     # multiplicative, on min-max normalised scores


def rrf(rankings: Sequence[Sequence[str]], k: int = RRF_K) -> Dict[str, float]:
    out: Dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, cid in enumerate(ranking, 1):
            out[cid] += 1.0 / (k + rank)
    return out


def norm(vals: Sequence[float]) -> List[float]:
    if not vals:
        return []
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-12:
        return [0.5] * len(vals)
    return [(v - lo) / (hi - lo) for v in vals]


# --- E3 -------------------------------------------------------------------

def doctype_prior(q: Query, cands: Sequence[Candidate], ctx: dict) -> List[float]:
    """Multiply a decree's normalised score by KEPUTUSAN_PENALTY.

    A Peraturan states the rule; a Keputusan states one case of it. A general
    question wants the rule, but the decree matches the words better because it
    names a concrete figure. This just says: prefer the rule, all else equal.
    """
    titles = ctx["titles"]
    base = norm(ctx["rerank"])
    return [
        s * (KEPUTUSAN_PENALTY if doc_type(titles.get(c.doc_id, "")) == "Keputusan" else 1.0)
        for s, c in zip(base, cands)
    ]


# --- E4 -------------------------------------------------------------------

class TitleChannel:
    """A second retrieval channel over document titles.

    922 vectors, one per document. It can surface a document whose *chunks*
    never matched — the failure no reranking change can reach.
    """

    def __init__(self) -> None:
        self.ids: List[str] = []
        self.mat: Optional[np.ndarray] = None

    async def build(self, bench: Bench) -> None:
        cache = cached_dense("title_emb")
        if cache is None:
            ids = list(bench.titles)
            vecs = await bench.retriever.embed([bench.titles[i] for i in ids])
            save_dense("title_emb", ids, vecs)
            cache = (ids, np.asarray(vecs, dtype="float32"))
        self.ids, mat = cache
        self.mat = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12)

    def top_docs(self, qvec: np.ndarray, n: int = TITLE_TOP_DOCS) -> List[str]:  # noqa: D401
        sims = self.mat @ qvec
        return [self.ids[j] for j in np.argsort(-sims)[:n]]


async def extra_chunks(doc_ids: Sequence[str], exclude: set) -> List[Tuple[str, str, str, str]]:
    """(chunk_id, parent_id, doc_id, text) for documents the title channel found."""
    if not doc_ids:
        return []
    async with async_session_maker() as session:
        rows = (await session.execute(
            select(ChildChunk.id, ChildChunk.parent_chunk_id,
                   ChildChunk.doc_id, ChildChunk.text)
            .where(ChildChunk.doc_id.in_(list(doc_ids)))
            .order_by(ChildChunk.doc_id, ChildChunk.ordinal)
        )).all()
    per: Dict[str, int] = defaultdict(int)
    out = []
    for cid, pid, did, text in rows:
        if cid in exclude or per[did] >= TITLE_CHUNKS_PER_DOC:
            continue
        per[did] += 1
        out.append((cid, pid, did, text or ""))
    return out


# --- runner ---------------------------------------------------------------

async def run_title_channel(bench: Bench, channel: TitleChannel,
                            qvecs: Dict[str, np.ndarray]) -> Dict[str, List[Candidate]]:
    """Baseline candidates plus title-matched documents' chunks, RRF fused."""
    fused: Dict[str, List[Candidate]] = {}
    for q in bench.queries:
        base = bench.candidates(q)
        have = {c.chunk_id for c in base}
        picked = channel.top_docs(qvecs[q.qid])
        extra = await extra_chunks(picked, have)

        pool = list(base) + [Candidate(c, p, d, 0.0) for c, p, d, _ in extra]
        texts = dict(bench.child_text)
        texts.update({c: t for c, _, _, t in extra})

        res = await bench.reranker.rerank(
            query=q.text, documents=[texts.get(c.chunk_id, "") for c in pool]
        )
        sc = [0.0] * len(pool)
        for r in res:
            sc[r.index] = float(r.score)
        pool_rank = [pool[i].chunk_id for i in
                     sorted(range(len(pool)), key=lambda i: sc[i], reverse=True)]
        base_rank = [base[i].chunk_id for i in
                     sorted(range(len(base)), key=lambda i: bench._rerank[f"plain::{q.qid}"][i],
                            reverse=True)]
        scores = rrf([base_rank, pool_rank])
        by_id = {c.chunk_id: c for c in pool}
        order = sorted(scores, key=lambda c: scores[c], reverse=True)
        fused[q.qid] = [by_id[c] for c in order if c in by_id]
        bench.child_text.update({c: t for c, _, _, t in extra})
    return fused


async def run_hyde(bench: Bench, model: str) -> Dict[str, List[Candidate]]:
    """HyDE dense retrieval fused with the raw-query ranking, as production would."""
    from dotenv import load_dotenv

    load_dotenv(".env")
    from app.chat.config import ChatConfig
    from app.chat.infra.hyde_expander import HyDEExpander
    from app.chat.infra.llm_connection import LLMConnection
    from app.kb.config import get_qdrant_settings
    from app.kb.infra.qdrant_store import QdrantStore

    cfg = ChatConfig()
    qcfg = get_qdrant_settings()
    store = QdrantStore(qcfg.host, qcfg.port, qcfg.collection_name)
    llm = LLMConnection(base_url=cfg.llm_base_url, api_key=cfg.llm_api_key)
    exp = HyDEExpander(llm, model, cfg.hyde_prompt_template, cfg.hyde_system_prompt,
                       cfg.hyde_max_tokens, cfg.hyde_temperature, cfg.hyde_num_passages)
    bench.retriever._ensure_clients()
    embedder = bench.retriever._embedder

    out: Dict[str, List[Candidate]] = {}
    try:
        for q in bench.queries:
            base = bench.candidates(q)
            raw = (await embedder.embed_texts([q.text]))[0]
            passages = await exp.expand_many(q.text)
            hyde_hits = []
            if passages:
                embs = await embedder.embed_texts(passages)
                mean = [sum(e.dense[i] for e in embs) / len(embs)
                        for i in range(len(embs[0].dense))]
                n = sum(v * v for v in mean) ** 0.5 or 1.0
                hyde_hits = await store.hybrid_search(
                    dense_vector=[v / n for v in mean],
                    sparse_indices=raw.sparse_indices,
                    sparse_values=raw.sparse_values, top_k=50, mode="dense")
            by_id = {c.chunk_id: c for c in base}
            by_id.update({h.chunk_id: Candidate(h.chunk_id, h.parent_chunk_id,
                                                h.doc_id, h.score)
                          for h in hyde_hits})
            base_rank = [base[i].chunk_id for i in
                         sorted(range(len(base)),
                                key=lambda i: bench._rerank[f"plain::{q.qid}"][i],
                                reverse=True)]
            scores = rrf([base_rank, [h.chunk_id for h in hyde_hits]])
            order = sorted(scores, key=lambda c: scores[c], reverse=True)
            out[q.qid] = [by_id[c] for c in order if c in by_id]
            missing = [c for c in out[q.qid] if c.chunk_id not in bench.child_text]
            if missing:
                from evals.probes._common import load_child_texts
                bench.child_text.update(
                    await load_child_texts([c.chunk_id for c in missing]))
    finally:
        await llm.close()
    return out


def report(name: str, bench: Bench, ranked: Dict[str, List[Candidate]]) -> dict:
    rows = []
    for q in bench.queries:
        cands = ranked[q.qid]
        top = cands[0]
        title = bench.titles.get(top.doc_id, "")
        expected = bench.expected[q.qid]
        seen: List[str] = []
        for c in cands:
            if c.doc_id not in seen:
                seen.append(c.doc_id)
        pos = next((i + 1 for i, d in enumerate(seen)
                    if expected and bench.titles.get(d, "").startswith(expected)), None)
        rows.append({
            "question": q.text, "expected": expected,
            "top_title": title, "top_type": doc_type(title),
            "expected_rank": pos,
            "expected_first": bool(expected) and bool(pos == 1),
        })
    kep = sum(1 for r in rows if r["top_type"] == "Keputusan")
    scored = [r for r in rows if r["expected"]]
    return {
        "intervention": name,
        "top_source_keputusan": kep,
        "questions": len(rows),
        "expected_doc_available": len(scored),
        "expected_doc_first": sum(1 for r in scored if r["expected_first"]),
        "rows": rows,
    }


async def main() -> None:
    quiet_logs()
    print(f"E3/E4/E5 — retrieval interventions  [{MODEL}]\n")

    bench = Bench()
    await bench.setup()
    for q in bench.queries:
        await bench.rerank_scores(q)

    results: List[dict] = []

    base_ranked = {}
    for q in bench.queries:
        sc = bench._rerank[f"plain::{q.qid}"]
        cs = bench.candidates(q)
        base_ranked[q.qid] = [cs[i] for i in
                              sorted(range(len(cs)), key=lambda i: sc[i], reverse=True)]
    results.append(report("baseline", bench, base_ranked))

    dt_ranked = {}
    for q in bench.queries:
        cs = bench.candidates(q)
        sc = doctype_prior(q, cs, {"titles": bench.titles,
                                   "rerank": bench._rerank[f"plain::{q.qid}"]})
        dt_ranked[q.qid] = [cs[i] for i in
                            sorted(range(len(cs)), key=lambda i: sc[i], reverse=True)]
    results.append(report("E3 doctype prior", bench, dt_ranked))

    channel = TitleChannel()
    await channel.build(bench)
    qv = np.asarray(await bench.retriever.embed([q.text for q in bench.queries]),
                    dtype="float32")
    qv /= np.linalg.norm(qv, axis=1, keepdims=True) + 1e-12
    qvecs = {q.qid: qv[i] for i, q in enumerate(bench.queries)}
    tc_ranked = await run_title_channel(bench, channel, qvecs)
    results.append(report("E4 title channel", bench, tc_ranked))

    # E4b — title similarity is a property of the *document*, so use it as a
    # multiplicative prior on that document's chunks rather than as a rival
    # chunk ranking. RRF against a strong base ranking barely moves the top.
    tsim: Dict[str, Dict[str, float]] = {}
    for i, q in enumerate(bench.queries):
        sims = channel.mat @ qv[i]
        tsim[q.qid] = {channel.ids[j]: float(sims[j]) for j in range(len(channel.ids))}
    for weight in (1.0, 2.0):
        ranked = {}
        for q in bench.queries:
            cs = tc_ranked[q.qid]
            base = norm([len(cs) - i for i in range(len(cs))])
            sc = [b * (1.0 + weight * max(0.0, tsim[q.qid].get(c.doc_id, 0.0)))
                  for b, c in zip(base, cs)]
            ranked[q.qid] = [cs[i] for i in
                             sorted(range(len(cs)), key=lambda i: sc[i], reverse=True)]
        results.append(report(f"E4b title prior w={weight}", bench, ranked))

        both_b = {}
        for q in bench.queries:
            cs = ranked[q.qid]
            base = norm([len(cs) - i for i in range(len(cs))])
            sc = [b * (KEPUTUSAN_PENALTY if doc_type(bench.titles.get(c.doc_id, "")) == "Keputusan" else 1.0)
                  for b, c in zip(base, cs)]
            both_b[q.qid] = [cs[i] for i in
                             sorted(range(len(cs)), key=lambda i: sc[i], reverse=True)]
        results.append(report(f"E3+E4b w={weight}", bench, both_b))

    # E3 + E4 together
    both = {}
    for q in bench.queries:
        cs = tc_ranked[q.qid]
        base = norm([len(cs) - i for i in range(len(cs))])
        sc = [b * (KEPUTUSAN_PENALTY
                   if doc_type(bench.titles.get(c.doc_id, "")) == "Keputusan" else 1.0)
              for b, c in zip(base, cs)]
        both[q.qid] = [cs[i] for i in
                       sorted(range(len(cs)), key=lambda i: sc[i], reverse=True)]
    results.append(report("E3+E4 combined", bench, both))

    hy_ranked = await run_hyde(bench, MODEL)
    results.append(report("E5 HyDE", bench, hy_ranked))

    await bench.close()

    print(f"  {'intervention':22s} {'decree on top':>14s} {'expected doc first':>19s}")
    for r in results:
        print(f"  {r['intervention']:22s} {r['top_source_keputusan']:>10d}/{r['questions']} "
              f"{r['expected_doc_first']:>15d}/{r['expected_doc_available']}")

    print(f"\n  per question (top source type):")
    print(f"  {'question':46s} " + " ".join(f"{r['intervention'][:9]:>9s}" for r in results))
    for i, q in enumerate(bench.queries):
        cells = " ".join(f"{r['rows'][i]['top_type'][:9]:>9s}" for r in results)
        print(f"  {q.text[:46]:46s} {cells}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "e3_retrieval.json").write_text(
        json.dumps({"model": MODEL, "results": results}, indent=2, ensure_ascii=False),
        encoding="utf-8")
    print(f"\n  Written to {RESULTS_DIR / 'e3_retrieval.json'}")


if __name__ == "__main__":
    asyncio.run(main())
