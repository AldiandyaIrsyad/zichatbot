"""Probe 5 — does HyDE find the document the query never names?

HyDE is off in production (`CHAT_HYDE_ENABLED=false`, "enable only after
paired validation") and the earlier experiments are why: on subset_a it is
neutral at best and destructive at worst.

    HyDE off, no rerank   hit@1 0.704   MRR 0.784
    HyDE on,  no rerank   hit@1 0.496   MRR 0.578      (exp2b_hyde_rerank)
    HyDE off, rerank      hit@1 0.765   MRR 0.810
    HyDE on,  rerank      hit@1 0.713   MRR 0.756
    HyDE on,  RRF fusion  hit@1 0.765   MRR 0.808      (exp2b_hyde_fusion_v4)

But subset_a is the one query set where HyDE *cannot* help: its questions are
written from the documents, so they already carry the nomor, the tahun and the
document's own vocabulary. HyDE can only dilute that — which is exactly what
`diagnose_hyde.py` found by measuring identifier retention.

The UKT case is the opposite: a student asks "Biaya kuliah per semester
berapa?" with none of the tariff document's vocabulary, and the right document
is never retrieved. That is precisely what HyDE was designed for. Whether the
earlier negative result generalises to this kind of query is an open question,
and this probe answers it for the one case we can score exactly.

    python -m evals.probes.probe5_hyde

Writes results/probe5_hyde.csv / .summary.json. Spends API credit (one LLM
call per passage per query, ~3 per query).
"""

from __future__ import annotations

import argparse
import asyncio
import os
from typing import Dict, List, Optional, Sequence, Tuple

from dotenv import load_dotenv

from ._common import (
    INITIAL_SEARCH_TOP_K,
    Retriever,
    doc_ranking,
    get_reranker,
    load_child_texts,
    load_documents,
    quiet_logs,
    write_results,
)

# Tuition questions, from the formal end to the SMS end of the register range.
QUERIES = [
    "Berapa biaya UKT?",
    "Biaya kuliah per semester berapa?",
    "Berapa tarif UKT yang harus saya bayar?",
    "UKT saya masuk golongan berapa?",
    "Berapa sih bayaran kuliah per semester?",
    "Biaya smstr brp ya?",
    "Gmn cara byr ukt?",
]

# The document that actually sets UKT by income bracket.
GOLD_TITLE_PREFIX = "003 Tahun 2022"


def rrf_fuse(rankings: Sequence[Sequence[str]], k: int = 60) -> List[str]:
    """Reciprocal rank fusion over chunk-id rankings, as search_service does."""
    scores: Dict[str, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, 1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda c: scores[c], reverse=True)


async def main() -> None:
    quiet_logs()
    load_dotenv()

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=os.environ.get("CHAT_LLM_MODEL", "qwen/qwen3-14b"))
    args = ap.parse_args()

    print(f"Probe 5 — HyDE on the UKT case  [model={args.model}]\n")

    docs = await load_documents()
    titles = {d.doc_id: d.title for d in docs.values()}
    gold = {i for i, t in titles.items() if t.startswith(GOLD_TITLE_PREFIX)}
    print(f"  target document: {titles[next(iter(gold))][:78]}\n")

    from app.chat.config import ChatConfig
    from app.chat.infra.hyde_expander import HyDEExpander
    from app.chat.infra.llm_connection import LLMConnection
    from app.kb.config import get_qdrant_settings
    from app.kb.infra.qdrant_store import QdrantStore

    cfg = ChatConfig()
    qcfg = get_qdrant_settings()
    store = QdrantStore(qcfg.host, qcfg.port, qcfg.collection_name)
    retriever = Retriever()
    retriever._ensure_clients()
    embedder = retriever._embedder
    reranker = get_reranker()

    llm = LLMConnection(base_url=cfg.llm_base_url, api_key=cfg.llm_api_key)
    expander = HyDEExpander(
        llm, args.model, cfg.hyde_prompt_template, cfg.hyde_system_prompt,
        cfg.hyde_max_tokens, cfg.hyde_temperature, cfg.hyde_num_passages,
    )

    rows: List[dict] = []
    try:
        for q in QUERIES:
            raw = (await embedder.embed_texts([q]))[0]
            raw_hits = await store.hybrid_search(
                dense_vector=raw.dense, sparse_indices=raw.sparse_indices,
                sparse_values=raw.sparse_values, top_k=INITIAL_SEARCH_TOP_K,
            )

            passages = await expander.expand_many(q)
            hyde_hits = []
            if passages:
                embs = await embedder.embed_texts(passages)
                mean = [
                    sum(e.dense[i] for e in embs) / len(embs)
                    for i in range(len(embs[0].dense))
                ]
                norm = sum(v * v for v in mean) ** 0.5 or 1.0
                mean = [v / norm for v in mean]
                # Mirrors search_service: HyDE drives the dense channel only,
                # the sparse vector stays the raw query's.
                hyde_hits = await store.hybrid_search(
                    dense_vector=mean, sparse_indices=raw.sparse_indices,
                    sparse_values=raw.sparse_values,
                    top_k=INITIAL_SEARCH_TOP_K, mode="dense",
                )

            by_id = {h.chunk_id: h for h in list(raw_hits) + list(hyde_hits)}
            fused_ids = rrf_fuse([
                [h.chunk_id for h in raw_hits],
                [h.chunk_id for h in hyde_hits],
            ]) if hyde_hits else [h.chunk_id for h in raw_hits]

            conditions = {
                "off": [h.chunk_id for h in raw_hits],
                "hyde_only": [h.chunk_id for h in hyde_hits],
                "fusion": fused_ids,
            }

            result = {"question": q, "hyde_passages": len(passages)}
            for name, ids in conditions.items():
                cands = [by_id[i] for i in ids if i in by_id]
                docs_rank = doc_ranking(cands)
                result[f"{name}_retrieved"] = int(any(d in gold for d in docs_rank))
                result[f"{name}_doc_rank"] = next(
                    (n for n, d in enumerate(docs_rank, 1) if d in gold), 0
                )
                # After reranking, which is what the user actually sees.
                if cands:
                    texts = await load_child_texts([c.chunk_id for c in cands])
                    rr = await reranker.rerank(
                        query=q, documents=[texts.get(c.chunk_id, "") for c in cands]
                    )
                    sc = [0.0] * len(cands)
                    for x in rr:
                        sc[x.index] = float(x.score)
                    order = sorted(range(len(cands)), key=lambda i: sc[i], reverse=True)
                    reranked = doc_ranking([cands[i] for i in order])
                    result[f"{name}_rerank_rank"] = next(
                        (n for n, d in enumerate(reranked, 1) if d in gold), 0
                    )
                    result[f"{name}_top1_title"] = titles.get(reranked[0], "")[:70]
                else:
                    result[f"{name}_rerank_rank"] = 0
                    result[f"{name}_top1_title"] = ""
            result["hyde_sample"] = " ".join(passages[0].split())[:300] if passages else ""
            rows.append(result)

            def show(n: int) -> str:
                return f"#{n}" if n else "not retrieved"

            print(f"  {q}")
            print(f"    HyDE off        {show(result['off_rerank_rank']):16s} "
                  f"top1: {result['off_top1_title'][:52]}")
            print(f"    HyDE only       {show(result['hyde_only_rerank_rank']):16s} "
                  f"top1: {result['hyde_only_top1_title'][:52]}")
            print(f"    RRF fusion      {show(result['fusion_rerank_rank']):16s} "
                  f"top1: {result['fusion_top1_title'][:52]}")
    finally:
        await llm.close()
        await retriever.close()
        await reranker.close()

    found = {
        name: sum(1 for r in rows if r[f"{name}_rerank_rank"]) for name in
        ("off", "hyde_only", "fusion")
    }
    top1 = {
        name: sum(1 for r in rows if r[f"{name}_rerank_rank"] == 1) for name in
        ("off", "hyde_only", "fusion")
    }
    summary = {
        "model": args.model,
        "queries": len(QUERIES),
        "target_document": titles[next(iter(gold))],
        "gold_document_retrieved_at_all": found,
        "gold_document_ranked_first": top1,
        "prior_subset_a_evidence": {
            "note": "From evals/data/results/. subset_a questions are written "
                    "from the documents, so they already carry the vocabulary "
                    "HyDE would otherwise supply.",
            "hyde_off_rerank": {"hit@1": 0.765, "mrr": 0.810},
            "hyde_on_rerank": {"hit@1": 0.713, "mrr": 0.756},
            "hyde_on_rrf_fusion": {"hit@1": 0.765, "mrr": 0.808},
        },
    }
    write_results("probe5_hyde", rows, summary)

    print(f"\n  target document found (of {len(QUERIES)} queries)")
    for name in ("off", "hyde_only", "fusion"):
        print(f"    {name:12s} retrieved {found[name]}/{len(QUERIES)}   "
              f"ranked #1 {top1[name]}/{len(QUERIES)}")


if __name__ == "__main__":
    asyncio.run(main())
