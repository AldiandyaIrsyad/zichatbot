"""Diagnose raw child retrieval versus token-budgeted parent ranking."""
from __future__ import annotations
import argparse,asyncio,csv,json,statistics,re
from collections import defaultdict
from pathlib import Path
from qdrant_client import AsyncQdrantClient
from transformers import AutoTokenizer
from app.kb.config import get_bge_m3_settings,get_qdrant_settings
from app.kb.infra.bge_m3_embeddings import BGEM3Embeddings
from evals._shared.dataset import load_subset_a
from evals.exp2a_chunking.run import load_manifest,load_document_texts,query_points,HIER_COLLECTION,FIXED_COLLECTION,paired_bootstrap,mcnemar_exact

def dedup(points):
 out=[];seen=set()
 for p in points:
  d=str((p.payload or {}).get("doc_id",""))
  if d and d not in seen:seen.add(d);out.append(d)
 return out

def rr5(rank,gold):
 for i,d in enumerate(rank[:5],1):
  if d in gold:return 1/i
 return 0.0

def terms(text):return {x.casefold() for x in re.findall(r"[A-Za-zÀ-ÿ0-9]+",text) if len(x)>2}
def recall(needle,haystack):
 a=terms(needle);b=terms(haystack);return len(a&b)/len(a) if a else 0.0

def pack(points,condition,tokenizer,parent_text,token_budget=8000):
 used=0;ranking=[];seen_docs=set();seen_contexts=set();texts=[]
 for point in points:
  payload=point.payload or {};doc=str(payload.get("doc_id",""));context=str(payload.get("parent_chunk_id",point.id))
  if context in seen_contexts:continue
  text=parent_text.get(context,str(payload.get("text",""))) if condition=="hierarchical" else str(payload.get("text",""))
  n=len(tokenizer.encode(text,add_special_tokens=False))
  if used and used+n>token_budget:continue
  used+=min(n,token_budget);seen_contexts.add(context);texts.append(text)
  if doc and doc not in seen_docs:ranking.append(doc);seen_docs.add(doc)
  if used>=token_budget:break
 return ranking,used,"\n".join(texts)

def aggregate(rows,key):
 out={}
 for cond in ("fixed","hierarchical"):
  vals=[r[f"{cond}_{key}"] for r in rows]
  out[cond]=sum(vals)/len(vals)
 return out

async def main(a):
 manifest=load_manifest(a.manifest); ids={r["doc_id"] for r in manifest}; data=[r for r in load_subset_a(a.dataset) if r.category!="out-of-domain"]
 _,parent_text=await load_document_texts(list(ids)); q=get_qdrant_settings();client=AsyncQdrantClient(host=q.host,port=q.port);e=get_bge_m3_settings();tok=AutoTokenizer.from_pretrained(e.model,local_files_only=True);emb=BGEM3Embeddings(model_name=e.model,device=e.device,use_fp16=e.use_fp16,batch_size=e.batch_size)
 rows=[]
 try:
  for start in range(0,len(data),e.batch_size):
   batch=data[start:start+e.batch_size]; vecs=await emb.embed_texts([x.question for x in batch])
   for row,v in zip(batch,vecs):
    rec={"question":row.question,"category":row.category,"gold_doc_ids":"|".join(row.gold_doc_ids)}
    for cond,col in (("fixed",a.fixed_collection),("hierarchical",a.hier_collection)):
     pts=await query_points(client,col,v);raw=dedup(pts);budget,used,context=pack(pts,cond,tok,parent_text)
     rec.update({f"{cond}_top50_points":len(pts),f"{cond}_top50_unique_docs":len(raw),f"{cond}_top8_unique_docs":len(dedup(pts[:8])),f"{cond}_raw_hit5":int(any(x in raw[:5] for x in row.gold_doc_ids)),f"{cond}_raw_rr5":rr5(raw,row.gold_doc_ids),f"{cond}_budget_hit5":int(any(x in budget[:5] for x in row.gold_doc_ids)),f"{cond}_budget_rr5":rr5(budget,row.gold_doc_ids),f"{cond}_budget_tokens":used,f"{cond}_source_context_term_recall":recall(row.source_context,context),f"{cond}_answer_term_recall":recall(row.ground_truth_answer,context)})
    rows.append(rec)
   print(f"diagnosed {min(start+len(batch),len(data))}/{len(data)}",flush=True)
 finally:await emb.close();await client.close()
 op=Path(a.output);op.parent.mkdir(parents=True,exist_ok=True)
 with op.open("w",newline="",encoding="utf-8") as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 meta=json.loads(Path(a.index_meta).read_text()); summary={"query_n":len(rows),"index_chunks":{"fixed":meta["fixed_chunks"],"hierarchical":meta["hierarchical_chunks"]},"raw_hit5":aggregate(rows,"raw_hit5"),"raw_mrr5":aggregate(rows,"raw_rr5"),"budget_hit5":aggregate(rows,"budget_hit5"),"budget_mrr5":aggregate(rows,"budget_rr5"),"mean_top50_unique_docs":{c:statistics.mean(r[f"{c}_top50_unique_docs"] for r in rows) for c in ("fixed","hierarchical")},"mean_top8_unique_docs":{c:statistics.mean(r[f"{c}_top8_unique_docs"] for r in rows) for c in ("fixed","hierarchical")},"budget_changed_hit5":{c:sum(r[f"{c}_raw_hit5"]!=r[f"{c}_budget_hit5"] for r in rows) for c in ("fixed","hierarchical")},"mean_source_context_term_recall":{c:statistics.mean(r[f"{c}_source_context_term_recall"] for r in rows) for c in ("fixed","hierarchical")},"mean_answer_term_recall":{c:statistics.mean(r[f"{c}_answer_term_recall"] for r in rows) for c in ("fixed","hierarchical")},"per_category":{}}
 summary["paired_raw_hit5"]=mcnemar_exact([bool(r["fixed_raw_hit5"]) for r in rows],[bool(r["hierarchical_raw_hit5"]) for r in rows])
 for key in ("raw_rr5","source_context_term_recall","answer_term_recall"):
  summary[f"paired_fixed_minus_hierarchical_{key}"]=paired_bootstrap([r[f"fixed_{key}"] for r in rows],[r[f"hierarchical_{key}"] for r in rows])
 for cat in sorted({r["category"] for r in rows}):
  qrows=[r for r in rows if r["category"]==cat];summary["per_category"][cat]={"n":len(qrows),"raw_hit5":aggregate(qrows,"raw_hit5"),"raw_mrr5":aggregate(qrows,"raw_rr5"),"budget_hit5":aggregate(qrows,"budget_hit5"),"budget_mrr5":aggregate(qrows,"budget_rr5"),"source_context_term_recall":aggregate(qrows,"source_context_term_recall"),"answer_term_recall":aggregate(qrows,"answer_term_recall")}
 op.with_suffix(".summary.json").write_text(json.dumps(summary,indent=2)+"\n");print(json.dumps(summary,indent=2))
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--manifest",default="evals/data/rq1_manifest.csv");p.add_argument("--dataset",default="evals/data/subset_a.csv");p.add_argument("--index-meta",default="evals/data/rq1_indexes.meta.json");p.add_argument("--fixed-collection",default=FIXED_COLLECTION);p.add_argument("--hier-collection",default=HIER_COLLECTION);p.add_argument("--output",default="evals/data/results/exp2a_chunking_diagnostic.csv");asyncio.run(main(p.parse_args()))
