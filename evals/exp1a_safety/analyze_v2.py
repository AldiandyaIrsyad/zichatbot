"""Normalize v2 safety results using adversarial-as-positive terminology."""
from __future__ import annotations
import csv,json,math
from collections import Counter,defaultdict
from pathlib import Path
ROOT=Path("evals/data/results")

def read(path, system=None):
    with open(path,newline="",encoding="utf-8") as f: rows=list(csv.DictReader(f))
    if system is not None: rows=[r for r in rows if r.get("system")==system and r.get("dataset")=="subset_b"]
    return rows

def key(r): return (r["query"],r["true_label"],r["attack_type"])
def summary(rows):
    n=len(rows); atk=[r for r in rows if r["true_label"]=="malicious"]; safe=[r for r in rows if r["true_label"]=="safe"]
    out={"n":n,"accuracy":sum(r["prediction"]==r["true_label"] for r in rows)/n,
         "attack_n":len(atk),"attack_pass_rate":sum(r["prediction"]=="safe" for r in atk)/len(atk),
         "safe_n":len(safe),"safe_false_block_rate":sum(r["prediction"]=="malicious" for r in safe)/len(safe)}
    out["subtypes"]={}
    for typ in sorted({r["attack_type"] for r in rows}):
        rr=[r for r in rows if r["attack_type"]==typ]
        out["subtypes"][typ]={"n":len(rr),"accuracy":sum(r["prediction"]==r["true_label"] for r in rr)/len(rr)}
    return out

def mcnemar(a,b):
    aa={key(r):r for r in a}; bb={key(r):r for r in b}; common=sorted(set(aa)&set(bb))
    ao=bo=0
    for k in common:
        ca=aa[k]["prediction"]==aa[k]["true_label"]; cb=bb[k]["prediction"]==bb[k]["true_label"]
        ao += ca and not cb; bo += cb and not ca
    n=ao+bo
    p=1.0 if n==0 else min(1.0,2*sum(math.comb(n,k) for k in range(0,min(ao,bo)+1))/(2**n))
    return {"n":len(common),"a_only_correct":ao,"b_only_correct":bo,"exact_p":p}

def main():
    local=read(ROOT/"exp1a_safety.csv")
    systems={
      "Prompt Guard 2 86M base":[r for r in local if r["system"]=="prompt_guard" and r["dataset"]=="subset_b"],
      "Prompt Guard 2 86M fine-tuned":[r for r in local if r["system"]=="prompt_guard_ft" and r["dataset"]=="subset_b"],
      "Qwen3Guard 0.6B local":read(ROOT/"exp1a_qwen_guard_local.csv"),
      "Qwen3-14B zero-shot (production-matched)":read(ROOT/"exp1a_qwen_baseline.csv"),
    }
    out={"positive_class":"adversarial/malicious","systems":{k:summary(v) for k,v in systems.items()},"paired_mcnemar":{}}
    names=list(systems)
    for i,a in enumerate(names):
      for b in names[i+1:]: out["paired_mcnemar"][f"{a} vs {b}"]=mcnemar(systems[a],systems[b])
    (ROOT/"exp1a_safety_summary.json").write_text(json.dumps(out,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps(out,indent=2,ensure_ascii=False))
if __name__=="__main__": main()
