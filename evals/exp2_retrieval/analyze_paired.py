"""Paired inference for Exp2b retrieval conditions."""
from __future__ import annotations
import argparse,csv,json
from collections import defaultdict
import numpy as np
from scipy.stats import binomtest

def load(path):
 with open(path,newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def key(r): return (r['question'],r['category'])
def cond(r): return (r['mode'],r['rerank'].lower(),r['hyde'].lower())
def mcnemar(a,b):
 ao=sum(x and not y for x,y in zip(a,b));bo=sum((not x) and y for x,y in zip(a,b));n=ao+bo
 return {'a_only':ao,'b_only':bo,'discordant':n,'p_value':1.0 if not n else float(binomtest(min(ao,bo),n,.5).pvalue)}
def boot(a,b,seed=42,n=10000):
 a=np.array(a,float);b=np.array(b,float);rng=np.random.default_rng(seed);d=np.empty(n)
 for i in range(n):
  ix=rng.integers(0,len(a),len(a));d[i]=np.mean(a[ix]-b[ix])
 return {'difference':float(np.mean(a-b)),'ci_low':float(np.quantile(d,.025)),'ci_high':float(np.quantile(d,.975))}
def compare(rows,a,b):
 groups=defaultdict(dict)
 for r in rows:groups[cond(r)][key(r)]=r
 common=sorted(set(groups[a])&set(groups[b]));ra=[groups[a][k] for k in common];rb=[groups[b][k] for k in common]
 return {'n':len(common),'hit5':mcnemar([float(x['hit_at_5']) == 1.0 for x in ra],[float(x['hit_at_5']) == 1.0 for x in rb]),'rr5':boot([float(x['reciprocal_rank']) for x in ra],[float(x['reciprocal_rank']) for x in rb])}
def main():
 p=argparse.ArgumentParser();p.add_argument('--modes',required=True);p.add_argument('--hyde',required=True);p.add_argument('--output',required=True);a=p.parse_args()
 m=load(a.modes);h=load(a.hyde);out={}
 for rr in ('true','false'):
  for other in ('dense','sparse'):out[f'hybrid_vs_{other}_rerank_{rr}']=compare(m,('hybrid',rr,''),(other,rr,''))
 for mode in ('hybrid','dense','sparse'):out[f'{mode}_rerank_on_vs_off']=compare(m,(mode,'true',''),(mode,'false',''))
 for rr in ('true','false'):out[f'hyde_on_vs_off_rerank_{rr}']=compare(h,('hybrid',rr,'true'),('hybrid',rr,'false'))
 open(a.output,'w').write(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
