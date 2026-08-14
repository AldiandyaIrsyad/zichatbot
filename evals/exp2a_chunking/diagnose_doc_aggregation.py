"""Diagnostic: aggregate child-level RRF results into document rankings."""
from __future__ import annotations
import argparse, asyncio, csv, json
from collections import defaultdict
from pathlib import Path
import numpy as np
from qdrant_client import AsyncQdrantClient, models
from scipy.stats import binomtest
from app.kb.config import get_bge_m3_settings, get_qdrant_settings
from app.kb.infra.bge_m3_embeddings import BGEM3Embeddings
from evals._shared.dataset import load_subset_a
from evals.exp2a_chunking.run import HIER_COLLECTION, FIXED_COLLECTION, paired_bootstrap

async def points(client, collection, emb, limit):
    f=models.Filter(must=[models.FieldCondition(key="is_active",match=models.MatchValue(value=True))])
    r=await client.query_points(collection_name=collection,prefetch=[
        models.Prefetch(query=models.SparseVector(indices=emb.sparse_indices,values=emb.sparse_values),using="bm25",limit=limit*2,filter=f),
        models.Prefetch(query=emb.dense,using="dense",limit=limit*2,filter=f)],query=models.FusionQuery(fusion=models.Fusion.RRF),limit=limit,with_payload=True)
    return r.points

def aggregate(ps):
    scores={}
    for p in ps:
        doc=str((p.payload or {}).get("doc_id", ""))
        if doc: scores[doc]=max(scores.get(doc,float("-inf")),float(p.score))
    return [d for d,_ in sorted(scores.items(),key=lambda x:(-x[1],x[0]))]

def rr(rank,gold):
    for i,d in enumerate(rank[:5],1):
        if d in gold:return 1/i
    return 0.0

def mc(a,b):
    ao=sum(x and not y for x,y in zip(a,b));bo=sum((not x) and y for x,y in zip(a,b));n=ao+bo
    return {"fixed_only":ao,"hierarchical_only":bo,"p_value":1.0 if not n else float(binomtest(min(ao,bo),n,.5).pvalue)}

async def main(a):
    data=[r for r in load_subset_a(a.dataset) if r.category!="out-of-domain"]
    q=get_qdrant_settings(); client=AsyncQdrantClient(host=q.host,port=q.port); e=get_bge_m3_settings(); emb=BGEM3Embeddings(model_name=e.model,device=e.device,use_fp16=e.use_fp16,batch_size=e.batch_size); rows=[]
    try:
      for start in range(0,len(data),e.batch_size):
       batch=data[start:start+e.batch_size]; vecs=await emb.embed_texts([r.question for r in batch])
       for row,v in zip(batch,vecs):
        rec={"question":row.question,"category":row.category,"gold_doc_ids":"|".join(row.gold_doc_ids)}
        for name,col in (("fixed",a.fixed_collection),("hierarchical",a.hierarchical_collection)):
          rank=aggregate(await points(client,col,v,a.candidate_limit));rec[name+"_unique_docs"]=len(rank);rec[name+"_hit5"]=int(any(d in rank[:5] for d in row.gold_doc_ids));rec[name+"_rr5"]=rr(rank,row.gold_doc_ids)
        rows.append(rec)
       print(f"aggregated {min(start+len(batch),len(data))}/{len(data)}",flush=True)
    finally:
      await emb.close();await client.close()
    op=Path(a.output);op.parent.mkdir(parents=True,exist_ok=True)
    with op.open("w",newline="",encoding="utf-8") as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    hf=[bool(r["fixed_hit5"]) for r in rows];hh=[bool(r["hierarchical_hit5"]) for r in rows]; rf=[r["fixed_rr5"] for r in rows];rh=[r["hierarchical_rr5"] for r in rows]
    out={
      "diagnostic_only":True, "candidate_limit":a.candidate_limit, "n":len(rows),
      "document_max_aggregation":{
        "fixed":{"hr5":sum(hf)/len(hf),"mrr5":sum(rf)/len(rf),"mean_unique_docs":float(np.mean([r["fixed_unique_docs"] for r in rows]))},
        "hierarchical":{"hr5":sum(hh)/len(hh),"mrr5":sum(rh)/len(rh),"mean_unique_docs":float(np.mean([r["hierarchical_unique_docs"] for r in rows]))}
      },
      "hit5_mcnemar":mc(hf,hh), "fixed_minus_hierarchical_mrr":paired_bootstrap(rf,rh)
    }
    op.with_suffix(".summary.json").write_text(json.dumps(out,indent=2)+"\n",encoding="utf-8");print(json.dumps(out,indent=2))

if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--dataset",default="evals/data/subset_a.csv");p.add_argument("--fixed-collection",default=FIXED_COLLECTION);p.add_argument("--hierarchical-collection",default=HIER_COLLECTION);p.add_argument("--candidate-limit",type=int,default=500);p.add_argument("--output",default="evals/data/results/exp2a_document_aggregation_diagnostic.csv");asyncio.run(main(p.parse_args()))
