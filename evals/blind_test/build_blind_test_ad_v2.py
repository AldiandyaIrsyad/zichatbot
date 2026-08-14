"""Build the reproducible combined Subset A-D blind human audit."""
from __future__ import annotations
import argparse, base64, csv, hashlib, json, math, random, re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parent.parent; DATA=ROOT/'data'; SEED=42
SOURCES={s:DATA/f'subset_{s}.csv' for s in 'abcd'}
POPS={'a':150,'b':160,'c':200,'d':210}; AUDIT={'a':30,'b':32,'c':40,'d':42}; QC={'a':6,'b':6,'c':8,'d':9}
A_TARGET={'factual':8,'procedural':8,'multi-hop':7,'out-of-domain':7}
B_TARGET={('malicious','hidden_instruction'):10,('malicious','jailbreak'):3,('malicious','dan_attempt'):3,('safe','safe_normal'):8,('safe','safe_complex'):8}
C_TARGET={
 ('in_domain','direct_upi','calibration'):3,('in_domain','direct_upi','locked_test'):7,
 ('in_domain','indirect_upi','calibration'):3,('in_domain','indirect_upi','locked_test'):7,
 ('out_of_domain','near_miss_government','calibration'):2,('out_of_domain','near_miss_government','locked_test'):5,
 ('out_of_domain','adjacent_legal','calibration'):2,('out_of_domain','adjacent_legal','locked_test'):5,
 ('out_of_domain','off_topic','calibration'):2,('out_of_domain','off_topic','locked_test'):4}
D_TARGET={'entailment':17,'neutral':15,'contradiction':10}
D_SPLIT={'entailment':{'calibration':2,'locked_test':15},'neutral':{'calibration':2,'locked_test':13},'contradiction':{'calibration':2,'locked_test':8}}
QUEUE_FIELDS=['item_id','subset','task','display_label','primary_text','secondary_text','tertiary_text','source_row_id']
KEY_FIELDS=['item_id','subset','task','reference_label','origin','qc_family','injected_wrong_label','source_row_id','source_sha256']

def read(path):
 with path.open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def write(path,fields,rows):
 path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def shown(path):
 try:return str(path.relative_to(ROOT))
 except ValueError:return str(path)
def sh(*parts,n=16):return hashlib.sha256('\x1f'.join(parts).encode()).hexdigest()[:n]
def rid(s,r):
 if s=='a': p=[r['question'],r['category'],r['ground_truth_answer'],r['source_doc_id'],r['source_context']]
 elif s=='b':p=[r['query'],r['label'],r['attack_type']]
 elif s=='c':p=[r['query'],r['label'],r['subtype'],r['split'],r['panel_yes'],r['panel_size']]
 else:p=[r['question_id'],r['sentence_id'],r['sentence_text'],r['retrieved_context'],r['label']]
 return sh(s,*p)
def iid(s,row_id,variant):return sh('blind-ad-v2',s,row_id,variant,n=12)
def fixed(rows,keyfn,targets,rng):
 g=defaultdict(list)
 for r in rows:g[keyfn(r)].append(r)
 out=[]
 for k,n in targets.items():
  if len(g[k])<n:raise ValueError(f'stratum {k} has {len(g[k])}, needs {n}')
  out+=rng.sample(g[k],n)
 return out
def hamilton(counts,total):
 pop=sum(counts.values());raw={k:v*total/pop for k,v in counts.items()};out={k:math.floor(v) for k,v in raw.items()}
 ranked=sorted(counts,key=lambda k:(-(raw[k]-out[k]),repr(k)))
 for k in ranked[:total-sum(out.values())]:out[k]+=1
 return out
def consensus(r):
 m=re.search(r'votes:\s*([^)]*)\)',r.get('verifier_note',''))
 if not m:raise ValueError(f"missing votes for {r.get('question_id')}")
 v=[x.strip() for x in m.group(1).split(',')]
 if len(v)!=5:raise ValueError('D row does not contain five votes')
 return '5/5' if len(set(v))==1 else '4/5'
def select_d(rows,rng):
 out=[]
 for label in ('entailment','neutral','contradiction'):
  for split,target in D_SPLIT[label].items():
   g=defaultdict(list)
   for r in rows:
    if r['label']==label and r['evaluation_split']==split:g[(r['construction'],consensus(r))].append(r)
   for k,n in hamilton({k:len(v) for k,v in g.items()},target).items():out+=rng.sample(g[k],n)
 return out
def qrow(s,task,display,primary,secondary,tertiary,row_id,variant):
 return {'item_id':iid(s,row_id,variant),'subset':f'subset_{s}','task':task,'display_label':display,'primary_text':primary,'secondary_text':secondary,'tertiary_text':tertiary,'source_row_id':row_id}
def krow(q,ref,origin,family,wrong,source_hash):
 return {'item_id':q['item_id'],'subset':q['subset'],'task':q['task'],'reference_label':ref,'origin':origin,'qc_family':family,'injected_wrong_label':wrong,'source_row_id':q['source_row_id'],'source_sha256':source_hash}
def authentic(s,r,source_hash):
 row_id=rid(s,r)
 if s=='a':q=qrow(s,'qa_validity',r['category'],r['question'],r['ground_truth_answer'],r['source_context'],row_id,'audit');ref='valid'
 elif s=='b':q=qrow(s,'safety','',r['query'],'','',row_id,'audit');ref=r['label']
 elif s=='c':q=qrow(s,'relevance','',r['query'],'','',row_id,'audit');ref=r['label']
 else:q=qrow(s,'nli','',r['sentence_text'],r['retrieved_context'],r['question'],row_id,'audit');ref=r['label']
 return q,krow(q,ref,'audit_sample','','',source_hash)
def remaining(s,rows,selected):
 used={rid(s,r) for r in selected};return [r for r in rows if rid(s,r) not in used]
def qc_a(rows,selected,source_hash,rng):
 rem=remaining('a',rows,selected);g=defaultdict(list)
 for r in rem:g[r['category']].append(r)
 donors=[r for r in rows if r['category']!='out-of-domain'];out=[]
 for cat,n in {'factual':2,'procedural':2,'multi-hop':1}.items():
  for base in rng.sample(g[cat],n):
   donor=rng.choice([r for r in donors if r['source_doc_id']!=base['source_doc_id'] and r['ground_truth_answer']!=base['ground_truth_answer']])
   row_id=rid('a',base);q=qrow('a','qa_validity',cat,base['question'],donor['ground_truth_answer'],base['source_context'],row_id,'qc-swap-'+sh(donor['question']))
   out.append((q,krow(q,'invalid','qc_control','answer_swap','valid',source_hash)))
 base=rng.choice([r for r in rem if r['category']!='out-of-domain' and r['ground_truth_answer']!='NONE' and r['source_context']!='NONE'])
 row_id=rid('a',base);q=qrow('a','qa_validity','out-of-domain',base['question'],base['ground_truth_answer'],base['source_context'],row_id,'qc-false-ood')
 out.append((q,krow(q,'invalid','qc_control','false_ood','valid',source_hash)));return out
def flip_qc(s,rows,selected,source_hash,rng):
 rem=remaining(s,rows,selected);g=defaultdict(list)
 for r in rem:g[r['label']].append(r)
 if s=='b':quota={'malicious':3,'safe':3};wrong={'malicious':'safe','safe':'malicious'}
 elif s=='c':quota={'in_domain':4,'out_of_domain':4};wrong={'in_domain':'out_of_domain','out_of_domain':'in_domain'}
 else:quota={'entailment':3,'neutral':3,'contradiction':3};wrong={'entailment':'neutral','neutral':'contradiction','contradiction':'entailment'}
 out=[]
 for label,n in quota.items():
  for r in rng.sample(g[label],n):
   row_id=rid(s,r)
   if s=='b':q=qrow(s,'safety','',r['query'],'','',row_id,'qc-flip-'+wrong[label])
   elif s=='c':q=qrow(s,'relevance','',r['query'],'','',row_id,'qc-flip-'+wrong[label])
   else:q=qrow(s,'nli','',r['sentence_text'],r['retrieved_context'],r['question'],row_id,'qc-flip-'+wrong[label])
   out.append((q,krow(q,label,'qc_control','label_flip',wrong[label],source_hash)))
 return out
def b64(x):return base64.b64encode(x.encode()).decode()
def browser_items(queue,key):
 keys={r['item_id']:r for r in key};out=[]
 for r in queue:
  k=keys[r['item_id']];out.append({'id':r['item_id'],'subset':r['subset'],'task':r['task'],'display':r['display_label'],'primary':r['primary_text'],'secondary':r['secondary_text'],'tertiary':r['tertiary_text'],'row':r['source_row_id'],'ref':b64(k['reference_label']),'origin':b64(k['origin']),'qc':b64(k['qc_family']),'wrong':b64(k['injected_wrong_label']),'source':k['source_sha256']})
 return out
def validate(queue,key):
 assert len(queue)==len(key)==173
 ids=[r['item_id'] for r in queue];assert len(ids)==len(set(ids)) and set(ids)=={r['item_id'] for r in key}
 auth=Counter(r['subset'] for r in key if r['origin']=='audit_sample');qc=Counter(r['subset'] for r in key if r['origin']=='qc_control')
 assert auth==Counter({f'subset_{s}':n for s,n in AUDIT.items()});assert qc==Counter({f'subset_{s}':n for s,n in QC.items()})
def build(queue_path,key_path,meta_path,template_path,html_path):
 sources={s:read(p) for s,p in SOURCES.items()};hashes={s:digest(p) for s,p in SOURCES.items()}
 for s,n in POPS.items():
  if len(sources[s])!=n:raise ValueError(f'Subset {s} expected {n}, found {len(sources[s])}')
 rng=random.Random(SEED)
 selected={'a':fixed(sources['a'],lambda r:r['category'],A_TARGET,rng),'b':fixed(sources['b'],lambda r:(r['label'],r['attack_type']),B_TARGET,rng),'c':fixed(sources['c'],lambda r:(r['label'],r['subtype'],r['split']),C_TARGET,rng),'d':select_d(sources['d'],rng)}
 for s,n in AUDIT.items():assert len(selected[s])==n
 pairs=[]
 for s in 'abcd':pairs += [authentic(s,r,hashes[s]) for r in selected[s]]
 pairs+=qc_a(sources['a'],selected['a'],hashes['a'],rng)
 for s in 'bcd':pairs+=flip_qc(s,sources[s],selected[s],hashes[s],rng)
 sections=defaultdict(list)
 for pair in pairs:sections[pair[0]['subset']].append(pair)
 ordered=[]
 for section in ('subset_a','subset_b','subset_c','subset_d'):rng.shuffle(sections[section]);ordered+=sections[section]
 queue=[x[0] for x in ordered];key=[x[1] for x in ordered];validate(queue,key)
 payload={'seed':SEED,'source_hashes':hashes,'audit_targets':AUDIT,'qc_targets':QC,'item_ids':[r['item_id'] for r in queue]};config_hash=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
 write(queue_path,QUEUE_FIELDS,queue);write(key_path,KEY_FIELDS,key)
 class_counts={s:dict(Counter(k['reference_label'] for k in key if k['subset']==f'subset_{s}' and k['origin']=='audit_sample')) for s in 'abcd'}
 meta={'dataset_version':'v2','audit':'combined_subset_a_d_blind_human_audit','generated_at_utc':datetime.now(timezone.utc).isoformat(),'seed':SEED,'source_files':{s:shown(p) for s,p in SOURCES.items()},'source_sha256':hashes,'source_populations':POPS,'authentic_sample_counts':AUDIT,'qc_control_counts':QC,'total_authentic':sum(AUDIT.values()),'total_qc':sum(QC.values()),'total_review_items':len(queue),'authentic_class_counts':class_counts,'sampling':{'a':'20% by generation category','b':'20% by label and attack type','c':'20% by label, subtype, and split','d':'label and split targets, then proportional construction and consensus'},'qc_policy':{'a':'five answer swaps plus one false-OOD triplet','b':'six balanced hidden label flips','c':'eight balanced hidden label flips','d':'nine hidden label flips, three per NLI class','interpretation':'Controlled QC variants are not genuine panel-rejected v2 rows and must be reported separately.'},'config_sha256':config_hash,'review_queue':shown(queue_path),'sealed_key':shown(key_path),'browser_test':shown(html_path)}
 meta_path.write_text(json.dumps(meta,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
 template=template_path.read_text(encoding='utf-8');items=json.dumps(browser_items(queue,key),ensure_ascii=False).replace('</script','<\\/script')
 html=template.replace('__ITEMS_JSON__',items).replace('__CONFIG_HASH__',config_hash).replace('__SOURCE_HASHES_JSON__',json.dumps(hashes))
 html_path.write_text(html,encoding='utf-8')
 print(f'Built {len(queue)} items: {sum(AUDIT.values())} authentic + {sum(QC.values())} QC');print(html_path)
def args():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--queue',type=Path,default=DATA/'blind_check_ad_v2.csv');p.add_argument('--key',type=Path,default=DATA/'blind_check_ad_v2.key.csv');p.add_argument('--meta',type=Path,default=DATA/'blind_check_ad_v2.meta.json');p.add_argument('--template',type=Path,default=ROOT/'blind_test/blind_test_ad_v2.template.html');p.add_argument('--html',type=Path,default=ROOT/'blind_test/blind_test_ad_v2.html');return p.parse_args()
if __name__=='__main__':
 a=args();build(a.queue,a.key,a.meta,a.template,a.html)
