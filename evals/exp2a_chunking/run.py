"""RQ1 chunking ablation on the seeded 300-document collection."""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import random
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
from qdrant_client import AsyncQdrantClient, models
from scipy.stats import binomtest
from sqlalchemy import select
from transformers import AutoTokenizer

from app.kb.config import get_bge_m3_settings, get_qdrant_settings
from app.kb.domain.interfaces import ChunkVector
from app.kb.domain.models import ParentChunk
from app.kb.infra.bge_m3_embeddings import BGEM3Embeddings
from app.kb.infra.qdrant_store import QdrantStore
from app.shared.db import async_session_maker
from evals._shared.dataset import load_subset_a
from evals._shared.metrics import compute_retrieval_metrics

HIER_COLLECTION = "rq1_v2_hierarchical"
FIXED_COLLECTION = "rq1_v2_fixed_512_64"
PRODUCTION_COLLECTION = "knowledge_base"


def load_manifest(path: str) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 300 or len({r["doc_id"] for r in rows}) != 300:
        raise AssertionError("RQ1 manifest must contain exactly 300 unique documents")
    if Counter(r["role"] for r in rows) != {"gold": 115, "distractor": 185}:
        raise AssertionError("RQ1 manifest must contain 115 gold + 185 distractors")
    return rows


async def recreate_collection(store: QdrantStore) -> None:
    names = {c.name for c in (await store._client.get_collections()).collections}
    if store.collection_name in names:
        await store._client.delete_collection(store.collection_name)
    await store.ensure_collection()


async def copy_hierarchical_points(
    client: AsyncQdrantClient, source: str, target: str, doc_ids: Sequence[str]
) -> int:
    offset = None
    copied = 0
    while True:
        records, offset = await client.scroll(
            collection_name=source,
            scroll_filter=models.Filter(must=[
                models.FieldCondition(key="doc_id", match=models.MatchAny(any=list(doc_ids)))
            ]),
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )
        if records:
            points = [models.PointStruct(id=r.id, vector=r.vector, payload=r.payload) for r in records]
            await client.upsert(collection_name=target, points=points, wait=True)
            copied += len(points)
            print(f"  hierarchical copied: {copied}", flush=True)
        if offset is None:
            break
    return copied


async def load_document_texts(doc_ids: Sequence[str]) -> Tuple[Dict[str, str], Dict[str, str]]:
    async with async_session_maker() as session:
        result = await session.execute(
            select(ParentChunk).where(ParentChunk.doc_id.in_(list(doc_ids))).order_by(
                ParentChunk.doc_id, ParentChunk.chunk_index, ParentChunk.ordinal
            )
        )
        parents = list(result.scalars())
    by_doc: Dict[str, List[str]] = defaultdict(list)
    parent_text: Dict[str, str] = {}
    for parent in parents:
        text = (parent.text or "").strip()
        if text:
            by_doc[str(parent.doc_id)].append(text)
            parent_text[str(parent.id)] = text
    texts = {doc_id: "\n\n".join(by_doc.get(doc_id, [])) for doc_id in doc_ids}
    if any(not text for text in texts.values()):
        raise AssertionError("every manifest document must have parent text")
    return texts, parent_text


def fixed_chunks(text: str, tokenizer: Any, size: int = 512, overlap: int = 64) -> List[str]:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    step = size - overlap
    chunks = []
    for start in range(0, len(token_ids), step):
        window = token_ids[start:start + size]
        if not window:
            break
        decoded = tokenizer.decode(window, skip_special_tokens=True).strip()
        if decoded:
            chunks.append(decoded)
        if start + size >= len(token_ids):
            break
    return chunks


async def build_indexes(
    manifest_path: str, source_collection: str, hier_collection: str, fixed_collection: str
) -> Dict[str, Any]:
    manifest = load_manifest(manifest_path)
    doc_ids = [r["doc_id"] for r in manifest]
    qcfg = get_qdrant_settings()
    client = AsyncQdrantClient(host=qcfg.host, port=qcfg.port)
    hier = QdrantStore(qcfg.host, qcfg.port, hier_collection)
    fixed = QdrantStore(qcfg.host, qcfg.port, fixed_collection)
    await recreate_collection(hier); await recreate_collection(fixed)
    try:
        copied = await copy_hierarchical_points(client, source_collection, hier_collection, doc_ids)
        texts, _ = await load_document_texts(doc_ids)
        ecfg = get_bge_m3_settings()
        tokenizer = AutoTokenizer.from_pretrained(ecfg.model, local_files_only=True)
        embedder = BGEM3Embeddings(
            model_name=ecfg.model, device=ecfg.device,
            use_fp16=ecfg.use_fp16, batch_size=ecfg.batch_size,
        )
        fixed_count = 0
        batch_texts: List[str] = []
        batch_meta: List[Tuple[str, int]] = []

        async def flush() -> None:
            nonlocal fixed_count, batch_texts, batch_meta
            if not batch_texts:
                return
            vectors = await embedder.embed_texts(batch_texts)
            payload = []
            namespace = uuid.UUID("d50e5de4-b565-4f43-b3e6-77fb82f84fb8")
            for text, (doc_id, idx), emb in zip(batch_texts, batch_meta, vectors):
                chunk_id = str(uuid.uuid5(namespace, f"{doc_id}:{idx}"))
                payload.append(ChunkVector(
                    chunk_id=chunk_id, parent_chunk_id=chunk_id, doc_id=doc_id,
                    dense_vector=emb.dense, sparse_indices=emb.sparse_indices,
                    sparse_values=emb.sparse_values, breadcrumbs=[],
                    content_type="text", text=text,
                ))
            await fixed.upsert_chunks(payload)
            fixed_count += len(payload)
            print(f"  fixed embedded: {fixed_count}", flush=True)
            batch_texts = []; batch_meta = []

        for n, doc_id in enumerate(doc_ids, 1):
            chunks = fixed_chunks(texts[doc_id], tokenizer)
            for idx, text in enumerate(chunks):
                batch_texts.append(text); batch_meta.append((doc_id, idx))
                if len(batch_texts) >= ecfg.batch_size:
                    await flush()
            if n % 10 == 0:
                print(f"  fixed documents prepared: {n}/300", flush=True)
        await flush(); await embedder.close()
        return {"documents": 300, "hierarchical_chunks": copied, "fixed_chunks": fixed_count}
    finally:
        await hier.close(); await fixed.close(); await client.close()


async def query_points(
    client: AsyncQdrantClient, collection: str, embedding: Any, limit: int = 50
) -> List[Any]:
    response = await client.query_points(
        collection_name=collection,
        prefetch=[
            models.Prefetch(
                query=models.SparseVector(indices=embedding.sparse_indices, values=embedding.sparse_values),
                using="bm25", limit=limit * 2,
                filter=models.Filter(must=[models.FieldCondition(key="is_active", match=models.MatchValue(value=True))]),
            ),
            models.Prefetch(
                query=embedding.dense, using="dense", limit=limit * 2,
                filter=models.Filter(must=[models.FieldCondition(key="is_active", match=models.MatchValue(value=True))]),
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=limit,
        with_payload=True,
    )
    return response.points


def budgeted_doc_ranking(
    points: Sequence[Any], condition: str, tokenizer: Any,
    parent_text: Dict[str, str], token_budget: int = 8000,
) -> Tuple[List[str], int]:
    used = 0; ranking: List[str] = []; seen_docs = set(); seen_contexts = set()
    for point in points:
        payload = point.payload or {}
        doc_id = str(payload.get("doc_id", ""))
        context_id = str(payload.get("parent_chunk_id", point.id))
        if context_id in seen_contexts:
            continue
        if condition == "hierarchical":
            text = parent_text.get(context_id, str(payload.get("text", "")))
        else:
            text = str(payload.get("text", ""))
        tokens = len(tokenizer.encode(text, add_special_tokens=False))
        if used and used + tokens > token_budget:
            continue
        used += min(tokens, token_budget)
        seen_contexts.add(context_id)
        if doc_id and doc_id not in seen_docs:
            ranking.append(doc_id); seen_docs.add(doc_id)
        if used >= token_budget:
            break
    return ranking, used


def paired_bootstrap(values_a: Sequence[float], values_b: Sequence[float], seed: int = 42, n: int = 10000) -> Dict[str, float]:
    a=np.asarray(values_a,float); b=np.asarray(values_b,float); rng=np.random.default_rng(seed)
    diffs=np.empty(n)
    for i in range(n):
        idx=rng.integers(0,len(a),len(a)); diffs[i]=np.mean(a[idx]-b[idx])
    return {"difference": float(np.mean(a-b)), "ci_low": float(np.quantile(diffs,.025)), "ci_high": float(np.quantile(diffs,.975))}


def mcnemar_exact(a: Sequence[bool], b: Sequence[bool]) -> Dict[str, Any]:
    b_only=sum((not x) and y for x,y in zip(a,b)); a_only=sum(x and (not y) for x,y in zip(a,b)); n=a_only+b_only
    p=1.0 if n==0 else float(binomtest(min(a_only,b_only),n,0.5,alternative="two-sided").pvalue)
    return {"a_only":a_only,"b_only":b_only,"discordant":n,"p_value":p}


async def evaluate(
    manifest_path: str, dataset_path: str, output_csv: str,
    hier_collection: str, fixed_collection: str,
) -> Dict[str, Any]:
    manifest = load_manifest(manifest_path); manifest_ids={r["doc_id"] for r in manifest}
    dataset=[r for r in load_subset_a(dataset_path) if r.category != "out-of-domain"]
    if len(dataset)!=115 or any(not set(r.gold_doc_ids)<=manifest_ids for r in dataset):
        raise AssertionError("evaluation must contain 115 document-bound queries whose golds are in the manifest")
    qcfg=get_qdrant_settings(); client=AsyncQdrantClient(host=qcfg.host,port=qcfg.port)
    ecfg=get_bge_m3_settings(); tokenizer=AutoTokenizer.from_pretrained(ecfg.model,local_files_only=True)
    embedder=BGEM3Embeddings(model_name=ecfg.model,device=ecfg.device,use_fp16=ecfg.use_fp16,batch_size=ecfg.batch_size)
    _,parent_text=await load_document_texts(list(manifest_ids))
    rows=[]; pairs={"fixed":[],"hierarchical":[]}; hit5={"fixed":[],"hierarchical":[]}; rr5={"fixed":[],"hierarchical":[]}
    try:
        for start in range(0,len(dataset),ecfg.batch_size):
            batch=dataset[start:start+ecfg.batch_size]
            embeddings=await embedder.embed_texts([r.question for r in batch])
            for row,emb in zip(batch,embeddings):
                result={"question":row.question,"category":row.category,"gold_doc_ids":"|".join(row.gold_doc_ids)}
                for condition,collection in (("fixed",fixed_collection),("hierarchical",hier_collection)):
                    points=await query_points(client,collection,emb)
                    ranking,tokens=budgeted_doc_ranking(points,condition,tokenizer,parent_text)
                    pairs[condition].append((ranking,row.gold_doc_ids))
                    h=any(g in ranking[:5] for g in row.gold_doc_ids); hit5[condition].append(h)
                    rank=next((i+1 for i,d in enumerate(ranking[:5]) if d in row.gold_doc_ids),None)
                    rr=0.0 if rank is None else 1.0/rank; rr5[condition].append(rr)
                    result[f"{condition}_doc_ids"]="|".join(ranking)
                    result[f"{condition}_context_tokens"]=tokens
                    result[f"{condition}_hit5"]=int(h); result[f"{condition}_rr5"]=rr
                rows.append(result)
            print(f"  evaluated {min(start+len(batch),len(dataset))}/{len(dataset)}",flush=True)
    finally:
        await embedder.close(); await client.close()
    out=Path(output_csv);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary={}
    for condition in ("fixed","hierarchical"):
        m=compute_retrieval_metrics(pairs[condition]);summary[condition]=m.__dict__
    summary["mcnemar_hit5_fixed_vs_hierarchical"]=mcnemar_exact(hit5["fixed"],hit5["hierarchical"])
    summary["bootstrap_rr5_fixed_minus_hierarchical"]=paired_bootstrap(rr5["fixed"],rr5["hierarchical"])
    out.with_suffix(".summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2));return summary


async def async_main(args: Any) -> None:
    if args.build_indexes:
        meta=await build_indexes(args.manifest,args.source_collection,args.hier_collection,args.fixed_collection)
        Path(args.manifest).with_name("rq1_indexes.meta.json").write_text(json.dumps(meta,indent=2),encoding="utf-8")
        print(json.dumps(meta))
    if args.evaluate:
        await evaluate(args.manifest,args.dataset,args.output,args.hier_collection,args.fixed_collection)


def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument("--manifest",default="evals/data/rq1_manifest.csv")
    p.add_argument("--dataset",default="evals/data/subset_a.csv")
    p.add_argument("--output",default="evals/data/results/exp2a_chunking.csv")
    p.add_argument("--source-collection",default=PRODUCTION_COLLECTION)
    p.add_argument("--hier-collection",default=HIER_COLLECTION)
    p.add_argument("--fixed-collection",default=FIXED_COLLECTION)
    p.add_argument("--build-indexes",action="store_true")
    p.add_argument("--evaluate",action="store_true")
    args=p.parse_args()
    if not args.build_indexes and not args.evaluate: p.error("select --build-indexes and/or --evaluate")
    asyncio.run(async_main(args))

if __name__=="__main__": main()
