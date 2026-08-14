"""Independently verify and score a completed combined A-D blind audit."""
from __future__ import annotations
import argparse,csv,hashlib,json,math
from collections import Counter,defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent;DATA=ROOT/'data'
ALLOWED={'qa_validity':{'valid','invalid'},'safety':{'malicious','safe'},'relevance':{'in_domain','out_of_domain'},'nli':{'entailment','neutral','contradiction'}}
def read(path):
 with path.open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def wilson(k,n):
 if not n:return[0.0,0.0]
 z=1.95996398454;p=k/n;den=1+z*z/n;mid=(p+z*z/(2*n))/den;half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
 return[max(0,mid-half),min(1,mid+half)]
def metric(rows):
 k=sum(int(r['correct']) for r in rows);n=len(rows);lo,hi=wilson(k,n);return{'n':n,'correct':k,'rate':k/n if n else 0.0,'ci_low':lo,'ci_high':hi}
def write(path,fields,rows):
 with path.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def score(reviewed,key_path,meta_path,out_csv,out_json,out_md):
 meta=json.loads(meta_path.read_text(encoding='utf-8'));keys=read(key_path);key={r['item_id']:r for r in keys};locator={(r['subset'],r['task'],r['source_row_id']):r for r in keys};rows=read(reviewed)
 if len({r['item_id'] for r in rows})!=len(rows):raise ValueError('Duplicate reviewed item IDs')
 for s,rel in meta['source_files'].items():
  actual=digest(ROOT/rel)
  if actual!=meta['source_sha256'][s]:raise ValueError(f'Source hash mismatch for Subset {s.upper()}')
 verified=[]
 for r in rows:
  input_item=r.get('item_id','');item=input_item;recovered=0
  if item not in key:
   k=locator.get((r.get('subset',''),r.get('task',''),r.get('source_row_id','')))
   if not k:raise ValueError(f'Unknown reviewed item {item}')
   item=k['item_id'];recovered=1
  else:k=key[item]
  task=k['task'];human=r.get('human_label','')
  if human not in ALLOWED[task]:raise ValueError(f'Invalid human label {human!r} for {task}')
  if r.get('config_sha256') and r['config_sha256']!=meta['config_sha256']:raise ValueError(f'Config hash mismatch for {item}')
  for field in ('subset','task','reference_label','origin','qc_family','injected_wrong_label','source_row_id','source_sha256'):
   if r.get(field,'')!=k[field]:raise ValueError(f'Sealed field mismatch for {item}: {field}')
  correct=int(human==k['reference_label']);qc_detected=int(k['origin']=='qc_control' and correct and human!=k['injected_wrong_label'])
  verified.append({'item_id':item,'input_item_id':input_item,'item_id_recovered':recovered,'subset':k['subset'],'task':task,'human_label':human,'reference_label':k['reference_label'],'correct':correct,'origin':k['origin'],'qc_family':k['qc_family'],'injected_wrong_label':k['injected_wrong_label'],'qc_detected':qc_detected,'source_row_id':k['source_row_id'],'source_sha256':k['source_sha256'],'config_sha256':meta['config_sha256']})
 by_subset={};by_class={}
 for subset in ('subset_a','subset_b','subset_c','subset_d'):
  auth=[r for r in verified if r['subset']==subset and r['origin']=='audit_sample'];qc=[r for r in verified if r['subset']==subset and r['origin']=='qc_control'];by_subset[subset]={'authentic':metric(auth),'qc':metric(qc)}
  groups=defaultdict(list)
  for r in auth:groups[r['reference_label']].append(r)
  for label,group in groups.items():by_class[f'{subset}|{label}']=metric(group)
 pooled=metric([r for r in verified if r['origin']=='audit_sample']);pooled_qc=metric([r for r in verified if r['origin']=='qc_control'])
 def pct(x):return f'{100*x:.1f}%'
 table=['| Subset | Population | Authentic n | Agree | Concordance | Wilson 95% CI | QC n | QC detected | QC rate |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
 for subset in ('subset_a','subset_b','subset_c','subset_d'):
  s=subset[-1];a=by_subset[subset]['authentic'];q=by_subset[subset]['qc'];table.append(f"| {subset.replace('_',' ').title()} | {meta['source_populations'][s]} | {a['n']} | {a['correct']} | {pct(a['rate'])} | [{pct(a['ci_low'])}, {pct(a['ci_high'])}] | {q['n']} | {q['correct']} | {pct(q['rate'])} |")
 table.append(f"| Descriptive total | {sum(meta['source_populations'].values())} | {pooled['n']} | {pooled['correct']} | {pct(pooled['rate'])} | [{pct(pooled['ci_low'])}, {pct(pooled['ci_high'])}] | {pooled_qc['n']} | {pooled_qc['correct']} | {pct(pooled_qc['rate'])} |")
 summary={'dataset_version':'v2','audit':'combined_subset_a_d_blind_human_audit','config_sha256':meta['config_sha256'],'source_sha256':meta['source_sha256'],'n_expected':meta['total_review_items'],'n_scored':len(verified),'n_skipped_or_missing':meta['total_review_items']-len(verified),'pooled_authentic':pooled,'pooled_qc':pooled_qc,'by_subset':by_subset,'by_class':by_class,'table_markdown':'\n'.join(table)}
 summary['item_ids_recovered']=sum(r['item_id_recovered'] for r in verified)
 result_table=['| Subset | Populasi | Sampel autentik | Konkordan | Konkordansi (Wilson 95% CI) | QC terdeteksi |','|---|---:|---:|---:|---:|---:|']
 names={'subset_a':'A (validitas QA)','subset_b':'B (keamanan)','subset_c':'C (relevansi)','subset_d':'D (NLI)'}
 def id_pct(x):return f'{100*x:.1f}%'.replace('.',',')
 for subset in ('subset_a','subset_b','subset_c','subset_d'):
  s=subset[-1];a=by_subset[subset]['authentic'];q=by_subset[subset]['qc'];result_table.append(f"| {names[subset]} | {meta['source_populations'][s]} | {a['n']} | {a['correct']} | {id_pct(a['rate'])} [{id_pct(a['ci_low'])}; {id_pct(a['ci_high'])}] | {q['correct']}/{q['n']} |")
 result_table.append(f"| Total deskriptif | {sum(meta['source_populations'].values())} | {pooled['n']} | {pooled['correct']} | {id_pct(pooled['rate'])} [{id_pct(pooled['ci_low'])}; {id_pct(pooled['ci_high'])}] | {pooled_qc['correct']}/{pooled_qc['n']} |")
 result_md=chr(10).join(['# Hasil Blind Human Audit A-D v2','','Angka di bawah dihitung ulang dari CSV hasil penilaian manusia, sealed key, dan hash sumber dataset. Baris audit autentik dan kontrol QC dilaporkan terpisah.','','## Tabel 4.6: Hasil Pemeriksaan Buta Manusia (Blind Re-labelling)','',*result_table,'',f"Pada {pooled['n']} baris autentik, {pooled['correct']} label manusia konkordan dengan label referensi ({id_pct(pooled['rate'])}; Wilson 95% CI [{id_pct(pooled['ci_low'])}; {id_pct(pooled['ci_high'])}]). Sebanyak {pooled_qc['correct']}/{pooled_qc['n']} kontrol QC terdeteksi dengan benar ({id_pct(pooled_qc['rate'])}). Total adalah ringkasan deskriptif karena tugas pelabelan berbeda antar-subset.",'','Audit historis B/C sebanyak 77 item unanimous tetap merupakan hasil terpisah dan tidak digabungkan dengan audit A-D ini karena kerangka sampelnya berbeda.',''])
 fields=['item_id','input_item_id','item_id_recovered','subset','task','human_label','reference_label','correct','origin','qc_family','injected_wrong_label','qc_detected','source_row_id','source_sha256','config_sha256']
 for path in (out_csv,out_json,out_md):path.parent.mkdir(parents=True,exist_ok=True)
 write(out_csv,fields,verified);out_json.write_text(json.dumps(summary,indent=2,ensure_ascii=False)+chr(10),encoding='utf-8');out_md.write_text(result_md,encoding='utf-8');print(result_md);print(f'Wrote {out_csv}, {out_json}, and {out_md}')
def args():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('reviewed',type=Path);p.add_argument('--key',type=Path,default=DATA/'blind_check_ad_v2.key.csv');p.add_argument('--meta',type=Path,default=DATA/'blind_check_ad_v2.meta.json');p.add_argument('--output-csv',type=Path);p.add_argument('--output-json',type=Path);p.add_argument('--output-md',type=Path,help='Ready-to-use thesis result table generated from the reviewed CSV');return p.parse_args()
if __name__=='__main__':
 a=args();stem=a.reviewed.name.removesuffix('.csv');out_csv=a.output_csv or a.reviewed.with_name(stem+'.VERIFIED.csv');out_json=a.output_json or a.reviewed.with_name(stem+'.summary.VERIFIED.json');out_md=a.output_md or a.reviewed.with_name(stem+'.result.VERIFIED.md');score(a.reviewed,a.key,a.meta,out_csv,out_json,out_md)
