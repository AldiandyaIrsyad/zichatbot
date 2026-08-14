"""Leakage-free Exp1b relevance calibration and locked evaluation."""
from __future__ import annotations
import argparse,asyncio,csv,json,math
from collections import Counter
from pathlib import Path
from typing import Any,Dict,List,Sequence,Tuple
from dotenv import load_dotenv
from scipy.stats import binomtest

from evals._shared.clients import EvalNLIClient,get_llm_client_from_env
from evals._shared.dataset import SubsetCRow,load_subset_c
from evals._shared.metrics import compute_binary_metrics,bootstrap_binary_ci
from evals.exp1b_relevance.run import fetch_kb_titles,derive_jdih_lexicon,retrieve_contexts_detailed
from app.guardrails.ivm.judge import LLMJudge
from app.guardrails.ivm.checkers import LLMJudgeRelevanceChecker


def balanced_accuracy(pred:Sequence[bool],truth:Sequence[bool])->float:
 tp=sum(p and t for p,t in zip(pred,truth));fn=sum((not p) and t for p,t in zip(pred,truth))
 tn=sum((not p) and (not t) for p,t in zip(pred,truth));fp=sum(p and (not t) for p,t in zip(pred,truth))
 return ((tp/(tp+fn) if tp+fn else 0)+(tn/(tn+fp) if tn+fp else 0))/2

def calibrate(scores:Sequence[float],truth:Sequence[bool])->Tuple[float,float]:
 vals=sorted(set(scores)); candidates=[vals[0]-1e-9]+[(a+b)/2 for a,b in zip(vals,vals[1:])]+[vals[-1]+1e-9]
 ranked=[]
 for threshold in candidates:
  pred=[s>=threshold for s in scores];ba=balanced_accuracy(pred,truth)
  false_refusal=sum((not p) and t for p,t in zip(pred,truth))
  ranked.append((ba,-false_refusal,-threshold,threshold))
 best=max(ranked);return best[3],best[0]

def mcnemar_correct(a:Sequence[bool],b:Sequence[bool],truth:Sequence[bool])->Dict[str,Any]:
 ac=[p==t for p,t in zip(a,truth)];bc=[p==t for p,t in zip(b,truth)]
 ao=sum(x and not y for x,y in zip(ac,bc));bo=sum((not x) and y for x,y in zip(ac,bc));n=ao+bo
 return {'a_only_correct':ao,'b_only_correct':bo,'discordant':n,'p_value':1.0 if not n else float(binomtest(min(ao,bo),n,.5).pvalue)}

def metrics_dict(pred:Sequence[bool],truth:Sequence[bool])->Dict[str,Any]:
 m=compute_binary_metrics(list(pred),list(truth));ci=bootstrap_binary_ci(list(pred),list(truth),metric='accuracy')
 return {**m.__dict__,'accuracy':m.accuracy,'precision':m.precision,'recall':m.recall,'f1':m.f1,
  'accuracy_ci':[ci.lower,ci.upper],
  'in_domain_false_refusal_rate':sum((not p) and t for p,t in zip(pred,truth))/max(1,sum(truth)),
  'ood_attack_pass_rate':sum(p and (not t) for p,t in zip(pred,truth))/max(1,sum(not t for t in truth))}

async def main_async(args:Any)->None:
 rows=load_subset_c(args.dataset);cal=[r for r in rows if r.split=='calibration'];test=[r for r in rows if r.split=='locked_test']
 if len(cal)!=60 or len(test)!=140:raise AssertionError(f'expected 60/140 split, got {len(cal)}/{len(test)}')
 titles=await fetch_kb_titles(args.api_url);lexicon=derive_jdih_lexicon(titles,min_doc_freq=args.lexicon_min_doc_freq)
 all_rows=cal+test; contexts=[]
 for i,row in enumerate(all_rows,1):
  chunks,scores=await retrieve_contexts_detailed(args.api_url,row.query,args.top_k,False);contexts.append((chunks,scores))
  if i%20==0:print(f'retrieved {i}/{len(all_rows)}',flush=True)
 keyword_scores=[float(sum(1 for term in lexicon if term in r.query.lower())) for r in all_rows]
 similarity_scores=[max(scores) if scores else float('-inf') for _,scores in contexts]
 nli=EvalNLIClient(args.infinity_url,args.nli_model);nli_scores=[]
 try:
  for i,(row,(chunks,_)) in enumerate(zip(all_rows,contexts),1):
   if not chunks:nli_scores.append(0.0)
   else:nli_scores.append((await nli.check(premise='\n'.join(chunks),hypothesis=row.query)).entailment_score)
   if i%20==0:print(f'nli {i}/{len(all_rows)}',flush=True)
 finally:await nli.aclose()
 cal_truth=[r.label=='in_domain' for r in cal];test_truth=[r.label=='in_domain' for r in test]
 methods={};thresholds={}
 for name,scores in [('keyword',keyword_scores),('similarity',similarity_scores),('nli',nli_scores)]:
  threshold,ba=calibrate(scores[:60],cal_truth);thresholds[name]={'threshold':threshold,'calibration_balanced_accuracy':ba}
  methods[name]=[s>=threshold for s in scores[60:]]
 load_dotenv();client=get_llm_client_from_env(model=args.judge_model);judge=LLMJudgeRelevanceChecker(LLMJudge(client,client.model));judge_pred=[];errors=0
 try:
  for i,(row,(chunks,scores)) in enumerate(zip(test,contexts[60:]),1):
   try:judge_pred.append(await judge.check_query(row.query,chunks,scores))
   except Exception:judge_pred.append(False);errors+=1
   if i%20==0:print(f'judge {i}/{len(test)}',flush=True)
 finally:await client.aclose()
 methods['llm_judge']=judge_pred
 summary={'calibration_n':60,'locked_test_n':140,'thresholds':thresholds,'judge_model':args.judge_model,'judge_errors':errors,'methods':{k:metrics_dict(v,test_truth) for k,v in methods.items()},'paired':{}}
 names=list(methods)
 for i,a in enumerate(names):
  for b in names[i+1:]:summary['paired'][f'{a}_vs_{b}']=mcnemar_correct(methods[a],methods[b],test_truth)
 out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
 with out.open('w',newline='',encoding='utf-8') as f:
  fields=['query','label','subtype','split','keyword_score','similarity_score','nli_score']+[f'{n}_pred' for n in methods]
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
  for i,row in enumerate(all_rows):
   rec={'query':row.query,'label':row.label,'subtype':row.subtype,'split':row.split,'keyword_score':keyword_scores[i],'similarity_score':similarity_scores[i],'nli_score':nli_scores[i]}
   if row.split=='locked_test':
    j=i-60
    for n,p in methods.items():rec[f'{n}_pred']='in_domain' if p[j] else 'out_of_domain'
   w.writerow(rec)
 out.with_suffix('.summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary,indent=2))

def main():
 p=argparse.ArgumentParser();p.add_argument('--dataset',default='evals/data/subset_c.csv');p.add_argument('--api-url',default='http://127.0.0.1:8000');p.add_argument('--infinity-url',default='http://127.0.0.1:7997');p.add_argument('--nli-model',default='StevenLimcorn/indo-roberta-indonli');p.add_argument('--judge-model',default='qwen/qwen3-14b');p.add_argument('--top-k',type=int,default=8);p.add_argument('--lexicon-min-doc-freq',type=int,default=3);p.add_argument('--output',default='evals/data/results/exp1b_relevance.csv');asyncio.run(main_async(p.parse_args()))
if __name__=='__main__':main()
