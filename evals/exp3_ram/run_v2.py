"""Calibrated v2 RAM evaluation on a 30-row calibration / 180-row locked test."""
from __future__ import annotations
import argparse,asyncio,csv,json,random
from pathlib import Path
from app.kb.infra import InfinityReranker
from evals._shared.clients import EvalNLIClient
from evals._shared.dataset import load_subset_d
from evals._shared.metrics import compute_multiclass_metrics,token_containment_similarity
from app.guardrails.ram.text_utils import split_sentences
LABELS=['entailment','neutral','contradiction']

def windows(text):
 s=[x if x.endswith(('.','?','!')) else x+'.' for x in split_sentences(text)]
 return [' '.join(s[i:i+3]) for i in range(0,len(s),2) if len(' '.join(s[i:i+3]))>20] or [text[:4000]]
def pred(es,cs,te,tc):
 if es>=te:return 'entailment'
 if cs>=tc:return 'contradiction'
 return 'neutral'
def calibrate(items):
 best=None
 for tei in range(20,96,5):
  for tci in range(20,96,5):
   te=tei/100;tc=tci/100; ps=[pred(x['e'],x['c'],te,tc) for x in items]
   m=compute_multiclass_metrics(ps,[x['truth'] for x in items],LABELS)
   key=(m.macro_f1,m.accuracy,-abs(te-.5)-abs(tc-.7))
   if best is None or key>best[0]:best=(key,te,tc)
 return best[1],best[2]
def metrics_dict(ps,ys):
 m=compute_multiclass_metrics(ps,ys,LABELS)
 return {'n':len(ys),'accuracy':m.accuracy,'macro_precision':m.macro_precision,'macro_recall':m.macro_recall,'macro_f1':m.macro_f1,'kappa':m.cohen_kappa,'per_class':{k:{'precision':v[0],'recall':v[1],'f1':v[2]} for k,v in m.per_class.items()},'confusion':m.confusion}
def bootstrap_diff(a,b,y,seed=42,nboot=2000):
 rng=random.Random(seed);n=len(y);vals=[]
 point=compute_multiclass_metrics(a,y,LABELS).macro_f1-compute_multiclass_metrics(b,y,LABELS).macro_f1
 for _ in range(nboot):
  ix=[rng.randrange(n) for __ in range(n)]
  vals.append(compute_multiclass_metrics([a[i] for i in ix],[y[i] for i in ix],LABELS).macro_f1-compute_multiclass_metrics([b[i] for i in ix],[y[i] for i in ix],LABELS).macro_f1)
 vals.sort();return {'point':point,'lower':vals[49],'upper':vals[1949]}
async def main(args):
 rows=load_subset_d(args.dataset);nli=EvalNLIClient(args.infinity_url,args.nli_model);rr=InfinityReranker(args.infinity_url,args.reranker_model);sem=asyncio.Semaphore(args.concurrency)
 async def one(i,row):
  async with sem:
   ws=windows(row.retrieved_context)
   full=await nli.check(row.retrieved_context,row.sentence_text)
   first=[]
   for w in ws[:2]:first.append(await nli.check(w,row.sentence_text))
   ranked=await rr.rerank(row.sentence_text,ws,top_k=2)
   learned=[]
   for x in ranked:learned.append(await nli.check(ws[x.index],row.sentence_text))
   def mx(rs):return {'e':max((x.entailment_score for x in rs),default=0),'c':max((x.contradiction_score for x in rs),default=0)}
   sim=token_containment_similarity(row.sentence_text,row.retrieved_context)
   return {'idx':i,'question_id':row.question_id,'sentence_id':row.sentence_id,'truth':row.label,'evaluation_split':row.evaluation_split,'split':row.split,
    'containment':{'e':sim,'c':1-sim},'single_pass_bounded_context':mx([full]),'window_no_rerank':mx(first),'window_rerank':mx(learned)}
 try:
  tasks=[asyncio.create_task(one(i,r)) for i,r in enumerate(rows)]
  done=[]
  for fut in asyncio.as_completed(tasks):
   done.append(await fut)
   if len(done)%10==0:print(f'{len(done)}/{len(rows)}',flush=True)
 finally:await nli.aclose();await rr.close()
 done.sort(key=lambda x:x['idx']); systems=['containment','single_pass_bounded_context','window_no_rerank','window_rerank'];summary={'dataset':args.dataset,'calibration_n':30,'locked_n':180,'systems':{},'paired_bootstrap_macro_f1':{}}
 predictions={}
 for s in systems:
  cal=[{**x[s],'truth':x['truth']} for x in done if x['evaluation_split']=='calibration'];te,tc=calibrate(cal)
  test=[x for x in done if x['evaluation_split']=='locked_test'];ys=[x['truth'] for x in test];ps=[pred(x[s]['e'],x[s]['c'],te,tc) for x in test];predictions[s]=ps
  summary['systems'][s]={'thresholds':{'entailment':te,'contradiction':tc},**metrics_dict(ps,ys)}
 ys=[x['truth'] for x in done if x['evaluation_split']=='locked_test']
 for i,a in enumerate(systems):
  for b in systems[i+1:]:summary['paired_bootstrap_macro_f1'][f'{a} minus {b}']=bootstrap_diff(predictions[a],predictions[b],ys)
 out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
 with out.open('w',newline='',encoding='utf-8') as f:
  fields=['question_id','sentence_id','true_label','evaluation_split','split']+[f'{s}_{z}' for s in systems for z in ('entailment_score','contradiction_score','prediction')]
  w=csv.DictWriter(f,fields);w.writeheader()
  for x in done:
   d={'question_id':x['question_id'],'sentence_id':x['sentence_id'],'true_label':x['truth'],'evaluation_split':x['evaluation_split'],'split':x['split']}
   for s in systems:
    th=summary['systems'][s]['thresholds'];d[f'{s}_entailment_score']=x[s]['e'];d[f'{s}_contradiction_score']=x[s]['c'];d[f'{s}_prediction']=pred(x[s]['e'],x[s]['c'],th['entailment'],th['contradiction'])
   w.writerow(d)
 out.with_suffix('.summary.json').write_text(json.dumps(summary,indent=2)+"\n")
 print(json.dumps(summary,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--dataset',default='evals/data/subset_d.csv');p.add_argument('--output',default='evals/data/results/exp3_ram.csv');p.add_argument('--infinity-url',default='http://localhost:7997');p.add_argument('--nli-model',default='StevenLimcorn/indo-roberta-indonli');p.add_argument('--reranker-model',default='BAAI/bge-reranker-v2-m3');p.add_argument('--concurrency',type=int,default=6);asyncio.run(main(p.parse_args()))
