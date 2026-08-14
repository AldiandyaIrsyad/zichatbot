"""Corrected Experiment 4: refusal, acceptance, latency, and traceability only."""
from __future__ import annotations
import argparse,asyncio,csv,json,random,re,statistics
from collections import Counter,defaultdict
from pathlib import Path
from evals._shared.dataset import load_subset_a,load_subset_c,SubsetARow
from evals._shared.metrics import wilson_interval
from evals.exp4_end_to_end.run import run_pipeline,CITATION_PATTERN

def select(a,c,seed):
 rng=random.Random(seed); ind=[x for x in a if x.category not in ('out-of-domain','out_of_domain')]
 by=defaultdict(list)
 for x in ind:by[x.category].append(x)
 for v in by.values():rng.shuffle(v)
 chosen=[]
 while len(chosen)<60:
  moved=False
  for k in sorted(by):
   if by[k] and len(chosen)<60:chosen.append(by[k].pop());moved=True
  if not moved:break
 ood=[x for x in c if x.label=='out_of_domain' and x.split=='locked_test'];rng.shuffle(ood)
 chosen += [SubsetARow(x.query,'out_of_domain','', 'NONE','NONE') for x in ood[:40]]
 return chosen,{x.query:x.subtype for x in ood[:40]}
def qtile(xs,q):
 if not xs:return 0
 ys=sorted(xs);return ys[min(len(ys)-1,max(0,round((len(ys)-1)*q)))]
def summarize(rows,subtypes):
 good=[r for r in rows if not r['errored']];ood=[r for r in good if r['category']=='out_of_domain'];ind=[r for r in good if r['category']!='out_of_domain']
 abst=sum(r['abstained'] for r in ood);fr=sum(r['abstained'] for r in ind);lats=[r['latency_s'] for r in good]
 ai=wilson_interval(abst,len(ood));fi=wilson_interval(fr,len(ind))
 failures=Counter(subtypes.get(r['question'],'unknown') for r in ood if not r['abstained'])
 return {'n':len(rows),'non_error_n':len(good),'errors':len(rows)-len(good),'ood_n':len(ood),'ood_abstained':abst,'ood_abstention_rate':abst/len(ood),'ood_abstention_wilson':[ai.lower,ai.upper],
 'in_domain_n':len(ind),'in_domain_false_refusals':fr,'in_domain_false_refusal_rate':fr/len(ind),'in_domain_false_refusal_wilson':[fi.lower,fi.upper],
 'ood_non_abstention_by_subtype':dict(failures),'total_request_latency_s':{'mean':statistics.mean(lats),'p50':statistics.median(lats),'p95':qtile(lats,.95)}}
async def main(args):
 data,subtypes=select(load_subset_a(args.subset_a),load_subset_c(args.subset_c),args.seed)
 out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
 with (out.parent/'exp4_manifest.csv').open('w',newline='',encoding='utf-8') as f:
  w=csv.writer(f);w.writerow(['question','category','gold_doc_ids','ood_subtype']);w.writerows([[x.question,x.category,x.source_doc_id,subtypes.get(x.question,'')] for x in data])
 sem=asyncio.Semaphore(args.concurrency); allrows=[]
 async def one(condition,row):
  async with sem:
   sw={'skip_ivm':False,'skip_ram':False} if condition=='full' else {'skip_ivm':True,'skip_ram':True}
   r=await run_pipeline(args.api_url,row,**sw)
   cites=[]
   for m in CITATION_PATTERN.finditer(r.response):cites.append({'status':m.group('status'),'source':m.group('source'),'page':m.group('page'),'doc_id':m.group('doc_id'),'evidence':m.group('evidence')})
   return {'condition':condition,'question':row.question,'category':row.category,'gold_doc_ids':row.source_doc_id,'ood_subtype':subtypes.get(row.question,''),'abstained':r.abstained,'errored':r.errored,'latency_s':r.latency_s,'marker_count':len(cites),'citations_json':json.dumps(cites,ensure_ascii=False),'retrieved_context':r.retrieved_context,'response':r.response}
 for cond in ('full','baseline'):
  tasks=[asyncio.create_task(one(cond,x)) for x in data]
  n=0
  for fut in asyncio.as_completed(tasks):
   allrows.append(await fut);n+=1
   if n%10==0:print(f'{cond} {n}/100',flush=True)
 fields=list(allrows[0])
 with out.open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fields);w.writeheader();w.writerows(allrows)
 summary={'design':{'in_domain_per_condition':60,'ood_per_condition':40,'conditions':['full','baseline'],'excluded_metrics':['BERTScore','NLI faithfulness','citation faithfulness'],'latency_note':'End-to-end request latency; stream exposes no component timestamps, so this is not guardrail overhead.'},'conditions':{}}
 for cond in ('full','baseline'):summary['conditions'][cond]=summarize([r for r in allrows if r['condition']==cond],subtypes)
 fullaccepted=[r for r in allrows if r['condition']=='full' and r['category']!='out_of_domain' and not r['abstained'] and not r['errored']][:30]
 audit=[]
 for r in fullaccepted:
  cs=json.loads(r['citations_json']);gold=set(x for x in r['gold_doc_ids'].split('|') if x and x!='NONE')
  evidence=[bool(c.get('evidence')) and c['evidence'].replace('…','').strip() in r['retrieved_context'] for c in cs]
  doc=[c.get('doc_id') in gold for c in cs if c.get('doc_id')]
  audit.append({'question':r['question'],'gold_doc_ids':r['gold_doc_ids'],'accepted':True,'marker_count':r['marker_count'],'marker_present':r['marker_count']>0,'evidence_snippets_present_in_retrieved_context':all(evidence) if evidence else None,'doc_ids_match_gold_when_emitted':all(doc) if doc else None,'page_fields_present':all(bool(c.get('page')) for c in cs) if cs else None,'human_evidence_correct':'','human_document_correct':'','human_page_correct':'','human_notes':''})
 ap=out.parent/'exp4_traceability_audit.csv'
 audit_fields=['question','gold_doc_ids','accepted','marker_count','marker_present','evidence_snippets_present_in_retrieved_context','doc_ids_match_gold_when_emitted','page_fields_present','human_evidence_correct','human_document_correct','human_page_correct','human_notes']
 with ap.open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,audit_fields);w.writeheader();w.writerows(audit)
 summary['traceability_automated']={'audit_n':len(audit),'answers_with_marker':sum(x['marker_present'] for x in audit),'human_audit_status':'pending sole-researcher decisions; blank fields are intentionally not fabricated'}
 out.with_suffix('.summary.json').write_text(json.dumps(summary,indent=2)+"\n")
 print(json.dumps(summary,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--subset-a',default='evals/data/subset_a.csv');p.add_argument('--subset-c',default='evals/data/subset_c.csv');p.add_argument('--api-url',default='http://localhost:8000');p.add_argument('--output',default='evals/data/results/exp4_end_to_end.csv');p.add_argument('--seed',type=int,default=42);p.add_argument('--concurrency',type=int,default=2);asyncio.run(main(p.parse_args()))
