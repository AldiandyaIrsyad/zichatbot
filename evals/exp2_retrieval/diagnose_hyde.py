"""Diagnostic sample for the negative HyDE result; not a replacement benchmark."""
from __future__ import annotations
import argparse,asyncio,csv,json,random,re
from collections import defaultdict
from pathlib import Path
from dotenv import load_dotenv
from app.chat.config import ChatConfig
from app.chat.infra.hyde_expander import HyDEExpander
from app.chat.infra.llm_connection import LLMConnection
from app.kb.infra.postgres_repo import PostgresKBRepository
from app.shared.db import async_session_maker
from evals._shared.dataset import load_subset_a

IDENT=re.compile(r"(?:\b[A-Z]{2,}[A-Z0-9.-]*\b|\b\d+(?:[/.-][A-Za-z0-9]+)+\b|\b\d{2,4}\b)")
WORD=re.compile(r"[A-Za-zÀ-ÿ0-9]+",re.UNICODE)
STOP={"yang","dan","dari","untuk","dalam","dengan","pada","apa","bagaimana","berapa","adalah","ini","itu","the","of","to"}
def terms(text):return {x.casefold() for x in WORD.findall(text) if len(x)>3 and x.casefold() not in STOP}
def jaccard(a,b):return len(a&b)/len(a|b) if a|b else 1.0

def load_retrieval(path):
 rows=list(csv.DictReader(open(path,encoding="utf-8")))
 return {(r["question"],r["rerank"],r["hyde"]):r for r in rows if r["mode"]=="hybrid"}

async def main(a):
 load_dotenv(); cfg=ChatConfig(); rows=[r for r in load_subset_a(a.dataset) if r.category!="out-of-domain"]
 by=defaultdict(list)
 for r in rows:by[r.category].append(r)
 rng=random.Random(a.seed); sample=[]
 for cat in sorted(by):rng.shuffle(by[cat]);sample.extend(by[cat][:a.per_category])
 retrieval=load_retrieval(a.retrieval)
 async with async_session_maker() as session:
  repo=PostgresKBRepository(session); docs=[d for d in await repo.get_all_pdfs() if d.active][:cfg.hyde_context_max_docs]
  context_ids={str(d.id) for d in docs}
  llm=LLMConnection(base_url=cfg.llm_base_url,api_key=cfg.llm_api_key)
  exp=HyDEExpander(llm,cfg.llm_model,cfg.hyde_prompt_template,cfg.hyde_system_prompt,cfg.hyde_max_tokens,cfg.hyde_temperature,repo,cfg.hyde_context_max_docs,cfg.hyde_context_refresh_seconds)
  out=[]
  try:
   for i,row in enumerate(sample,1):
    text=await exp.expand(row.question)
    ids=IDENT.findall(row.question); retained=[x for x in ids if x.casefold() in text.casefold()]; qids={x.casefold() for x in ids}; new_ids=[x for x in IDENT.findall(text) if x.casefold() not in qids]; low=text.casefold()
    off=retrieval[(row.question,"False","false")]; on=retrieval[(row.question,"False","true")]
    out.append({"question":row.question,"category":row.category,"gold_doc_ids":"|".join(row.gold_doc_ids),"gold_in_hyde_context20":any(x in context_ids for x in row.gold_doc_ids),"query_identifiers":"|".join(ids),"retained_identifiers":"|".join(retained),"identifier_retention":len(retained)/len(ids) if ids else "","query_expansion_term_jaccard":jaccard(terms(row.question),terms(text)),"hyde_text":text,"new_hyde_identifiers":"|".join(new_ids),"new_hyde_identifier_count":len(new_ids),"hyde_claims_absence":bool(re.search(r"tidak terdapat|tidak ada informasi|belum terdapat|tidak disebutkan|tidak dapat ditentukan",low)),"hyde_contains_placeholder":bool(re.search(r"\[[^\]]+\]",text)),"off_hit5":off["hit_at_5"],"on_hit5":on["hit_at_5"],"off_rr":off["reciprocal_rank"],"on_rr":on["reciprocal_rank"]})
    print(f"expanded {i}/{len(sample)}",flush=True)
  finally:await llm.close()
 op=Path(a.output);op.parent.mkdir(parents=True,exist_ok=True)
 with op.open("w",newline="",encoding="utf-8") as f:w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
 ident=[float(x["identifier_retention"]) for x in out if x["identifier_retention"]!=""]
 summary={"diagnostic_only":True,"model":cfg.llm_model,"sample_n":len(out),"per_category":a.per_category,"hyde_context_document_limit":cfg.hyde_context_max_docs,"sample_gold_present_in_context20":sum(x["gold_in_hyde_context20"] for x in out),"identifier_queries_n":len(ident),"mean_identifier_retention":sum(ident)/len(ident) if ident else None,"zero_identifier_retention_n":sum(x==0 for x in ident),"expansions_with_new_identifiers":sum(int(x["new_hyde_identifier_count"])>0 for x in out),"mean_new_identifiers_per_expansion":sum(int(x["new_hyde_identifier_count"]) for x in out)/len(out),"expansions_claiming_information_absent":sum(bool(x["hyde_claims_absence"]) for x in out),"expansions_with_placeholders":sum(bool(x["hyde_contains_placeholder"]) for x in out),"mean_query_expansion_term_jaccard":sum(float(x["query_expansion_term_jaccard"]) for x in out)/len(out),"hyde_hit5_losses":sum(float(x["on_hit5"])<float(x["off_hit5"]) for x in out),"hyde_hit5_gains":sum(float(x["on_hit5"])>float(x["off_hit5"]) for x in out),"interpretation_scope":"Prompt/mechanism diagnostic over a seeded sample; does not replace the paired 115-query ablation."}
 op.with_suffix(".summary.json").write_text(json.dumps(summary,indent=2)+"\n")
 print(json.dumps(summary,indent=2))
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--dataset",default="evals/data/subset_a.csv");p.add_argument("--retrieval",default="evals/data/results/exp2b_hyde_rerank.csv");p.add_argument("--output",default="evals/data/results/exp2b_hyde_diagnostic.csv");p.add_argument("--per-category",type=int,default=10);p.add_argument("--seed",type=int,default=42);asyncio.run(main(p.parse_args()))
