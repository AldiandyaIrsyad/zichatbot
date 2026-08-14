"""Build the reproducible Subset A blind re-labelling sample and review tool.

The current v2 Subset A has 150 accepted rows but no surviving item-level
panel votes. This audit asks a human to repeat the panel's binary QA-quality
decision under the stored category-specific rubric.

Sampling is stratified at exactly 20 percent within each category:
factual=8, procedural=8, multi-hop=7, and out-of-domain=7.
Ten controlled invalid variants are mixed in as attention and specificity
checks. They are reported separately and are never presented as genuine
panel-rejected v2 rows.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE = ROOT / "evals/data/subset_a.csv"
DEFAULT_QUEUE = ROOT / "evals/data/subset_a_blind_check.csv"
DEFAULT_KEY = ROOT / "evals/data/subset_a_blind_check.key.csv"
DEFAULT_MANIFEST = ROOT / "evals/data/subset_a_blind_check.meta.json"
DEFAULT_HTML = ROOT / "blind_test/blind_test_a_v2.html"

SEED = 42
TARGETS = {
    "factual": 8,
    "procedural": 8,
    "multi-hop": 7,
    "out-of-domain": 7,
}
QC_TARGETS = {
    "factual": 3,
    "procedural": 3,
    "multi-hop": 2,
    "out-of-domain": 2,
}
REQUIRED_FIELDS = {
    "question",
    "category",
    "ground_truth_answer",
    "source_doc_id",
    "source_context",
}


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_FIELDS.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing fields: {sorted(missing)}")
        return list(reader)


def item_id(row: Dict[str, str]) -> str:
    canonical = "\x1f".join(
        row[field]
        for field in (
            "question",
            "category",
            "ground_truth_answer",
            "source_doc_id",
            "source_context",
        )
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


def sample_rows(rows: Iterable[Dict[str, str]]) -> List[Dict[str, str]]:
    grouped: Dict[str, List[Dict[str, str]]] = {category: [] for category in TARGETS}
    for row in rows:
        category = row["category"]
        if category not in grouped:
            raise ValueError(f"Unexpected Subset A category: {category!r}")
        grouped[category].append(row)

    rng = random.Random(SEED)
    selected: List[Dict[str, str]] = []
    for category, target in TARGETS.items():
        pool = grouped[category]
        if len(pool) < target:
            raise ValueError(
                f"Category {category!r} has {len(pool)} rows, fewer than target {target}"
            )
        selected.extend(rng.sample(pool, target))
    rng.shuffle(selected)
    return selected


def make_qc_controls(
    rows: List[Dict[str, str]], selected: List[Dict[str, str]]
) -> List[Dict[str, str]]:
    """Create deterministic invalid controls without calling them panel rejects."""
    selected_ids = {item_id(row) for row in selected}
    grouped = {
        category: [
            row for row in rows
            if row["category"] == category and item_id(row) not in selected_ids
        ]
        for category in TARGETS
    }
    in_domain = [row for row in rows if row["category"] != "out-of-domain"]
    rng = random.Random(SEED + 1)
    controls: List[Dict[str, str]] = []

    for category in ("factual", "procedural", "multi-hop"):
        for base in rng.sample(grouped[category], QC_TARGETS[category]):
            donors = [
                row for row in in_domain
                if row["source_doc_id"] != base["source_doc_id"]
                and row["ground_truth_answer"] != base["ground_truth_answer"]
            ]
            donor = rng.choice(donors)
            control = dict(base)
            control["ground_truth_answer"] = donor["ground_truth_answer"]
            control["_origin"] = "qc_control"
            control["_expected_label"] = "invalid"
            control["_qc_family"] = "answer_swap"
            controls.append(control)

    false_ood_bases = rng.sample(in_domain, QC_TARGETS["out-of-domain"])
    for base in false_ood_bases:
        control = dict(base)
        control["category"] = "out-of-domain"
        control["_origin"] = "qc_control"
        control["_expected_label"] = "invalid"
        control["_qc_family"] = "false_ood"
        controls.append(control)

    return controls


def write_csv(path: Path, fieldnames: List[str], rows: Iterable[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def b64(value: str) -> str:
    return base64.b64encode(value.encode("utf-8")).decode("ascii")


def render_html(items: List[Dict[str, str]], source_hash: str) -> str:
    browser_items = [
        {
            "id": row["item_id"],
            "q": row["question"],
            "a": row["ground_truth_answer"],
            "d": row["source_doc_id"],
            "c": row["source_context"],
            "cat": row["category"],
            "p": b64(row["expected_label"]),
            "o": b64(row["origin"]),
            "f": b64(row["qc_family"]),
        }
        for row in items
    ]
    encoded_items = json.dumps(browser_items, ensure_ascii=False).replace(
        "</script", "<\\/script"
    )
    return f"""<!doctype html>
<html lang="id">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Blind Re-labelling Subset A v2</title>
<style>
:root{{--bg:#f5f7fa;--card:#fff;--ink:#172033;--muted:#667085;--line:#d9dee8;
--accent:#2459d3;--good:#177245;--bad:#b42318;--soft:#eef3ff}}
@media(prefers-color-scheme:dark){{:root{{--bg:#10141c;--card:#181e29;--ink:#edf1f7;
--muted:#a7b0c0;--line:#30394a;--accent:#7ba2ff;--good:#5dd39e;--bad:#ff8178;--soft:#202b42}}}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}}.wrap{{max-width:940px;margin:auto;padding:28px 16px}}
h1{{font-size:1.45rem;margin:0 0 5px}}h2{{font-size:1.05rem;margin:0 0 12px}}p{{margin:7px 0}}
.muted{{color:var(--muted)}}.card{{background:var(--card);border:1px solid var(--line);
border-radius:12px;padding:20px;margin-top:16px}}.hidden{{display:none}}.defs{{display:grid;
grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:9px;margin-top:14px}}.def{{border:1px solid
var(--line);border-radius:8px;padding:10px}}.def b{{display:block}}button{{font:inherit;color:var(--ink);
background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px 14px;cursor:pointer}}
button:hover{{border-color:var(--accent)}}.primary{{background:var(--accent);color:white;border-color:var(--accent)}}
.bar{{height:7px;background:var(--line);border-radius:9px;overflow:hidden;margin:15px 0 5px}}.bar i{{display:block;
height:100%;background:var(--accent);width:0}}.between{{display:flex;justify-content:space-between;gap:12px;
color:var(--muted);font-size:.82rem}}.field{{margin-top:13px}}.label{{font-size:.72rem;letter-spacing:.05em;
text-transform:uppercase;color:var(--muted);font-weight:700;margin-bottom:4px}}.value{{white-space:pre-wrap;
word-break:break-word;background:var(--bg);border:1px solid var(--line);border-radius:8px;padding:12px;
max-height:250px;overflow:auto}}.question{{font-size:1.04rem}}.options{{display:grid;
grid-template-columns:repeat(2,minmax(0,1fr));gap:9px;margin-top:17px}}.option{{font-weight:650}}
.option.selected{{background:var(--accent);border-color:var(--accent);color:white}}kbd{{font-size:.72rem;
opacity:.7;margin-right:7px}}.nav{{display:flex;align-items:center;gap:9px;margin-top:13px}}.spacer{{flex:1}}
.stats{{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:10px}}.stat{{text-align:center;
background:var(--bg);border:1px solid var(--line);border-radius:8px;padding:13px}}.num{{font-size:1.65rem;
font-weight:750}}.good{{color:var(--good)}}.bad{{color:var(--bad)}}table{{width:100%;border-collapse:collapse;
font-size:.86rem}}th,td{{padding:8px;border-bottom:1px solid var(--line);text-align:left}}th{{color:var(--muted);
font-size:.72rem;text-transform:uppercase}}.actions{{display:flex;gap:9px;flex-wrap:wrap;margin-top:14px}}
@media(max-width:620px){{.options{{grid-template-columns:1fr}}}}
</style>
</head>
<body><main class="wrap">
<section id="start">
<h1>Blind Re-labelling Subset A v2</h1>
<p class="muted">30 baris audit (20% dari Subset A) dan 10 kontrol QC invalid, diacak dengan seed 42.</p>
<div class="card"><h2>Tugas Anda</h2>
<p>Putuskan apakah setiap triplet valid atau invalid untuk kategori yang ditampilkan. Status audit/QC dan
keputusan acuan disembunyikan sampai semua penilaian selesai.</p>
<div class="defs">
<div class="def"><b>Factual</b>Jelas, satu fakta eksplisit menjawab pertanyaan, dan seluruh jawaban didukung konteks.</div>
<div class="def"><b>Procedural</b>Jelas, meminta langkah/syarat/proses, dan seluruh jawaban didukung konteks.</div>
<div class="def"><b>Multi-hop</b>Jelas, perlu menggabungkan sekurang-kurangnya dua fakta, semuanya ada dalam konteks.</div>
<div class="def"><b>Out-of-domain</b>Benar-benar di luar dokumen internal JDIH UPI; jawaban dan konteks harus tepat NONE.</div>
</div>
<p class="muted">Pilih Valid hanya jika seluruh kriteria kategori terpenuhi. Kontrol QC adalah varian invalid
terkendali, bukan kandidat yang ditolak panel. Hasil audit dan deteksi QC akan dilaporkan terpisah.</p>
<div class="actions"><button id="begin" class="primary">Mulai</button><button id="clear">Hapus progres</button></div>
</div></section>
<section id="review" class="hidden">
<h1>Blind Re-labelling Subset A</h1><div class="bar"><i id="fill"></i></div>
<div class="between"><span id="pos"></span><span id="progress"></span></div>
<div class="card">
<div class="field"><div class="label">Kategori yang harus diperiksa</div><div class="value" id="category"></div></div>
<div class="field"><div class="label">Pertanyaan</div><div class="value question" id="question"></div></div>
<div class="field"><div class="label">Ground-truth answer</div><div class="value" id="answer"></div></div>
<div class="field"><div class="label">Source context</div><div class="value" id="context"></div></div>
<div class="options" id="options"></div>
<div class="nav"><button id="prev">Kembali</button><button id="skip">Lewati</button><span class="spacer"></span>
<button id="next">Berikutnya</button></div></div>
<div class="actions"><button id="finish" class="primary">Selesai dan buka label</button></div>
</section>
<section id="results" class="hidden">
<h1>Hasil Blind Re-labelling Subset A</h1><p class="muted" id="resultmeta"></p>
<div class="card"><div class="stats" id="stats"></div></div>
<div class="card"><h2>Konfirmasi baris audit per kategori</h2><table id="bycat"></table></div>
<div class="card"><h2>Ketidaksesuaian</h2><table id="disagreements"></table></div>
<div class="card"><h2>Ekspor untuk tabel hasil</h2><p class="muted">Unduh kedua berkas dan berikan kembali
kepada Codex. CSV berisi hasil per item; JSON berisi angka ringkas untuk tabel concordance.</p>
<div class="actions"><button id="csv" class="primary">Unduh reviewed CSV</button>
<button id="summary">Unduh summary JSON</button><button id="back">Kembali meninjau</button></div></div>
</section>
</main>
<script>
(()=>{{"use strict";
const ITEMS={encoded_items};
const LABELS=[["valid","Valid"],["invalid","Invalid"]];
const KEY="blind_validity_a_v2_qc_{source_hash[:12]}",d=id=>document.getElementById(id);
const decode=s=>decodeURIComponent(escape(atob(s)));let judgments={{}},idx=0,last=null;
try{{judgments=JSON.parse(localStorage.getItem(KEY)||"{{}}")}}catch(e){{judgments={{}}}}
const save=()=>localStorage.setItem(KEY,JSON.stringify(judgments));
const show=id=>["start","review","results"].forEach(x=>d(x).classList.toggle("hidden",x!==id));
const answered=()=>ITEMS.filter(x=>judgments[x.id]&&judgments[x.id]!=="skip").length;
function render(){{const x=ITEMS[idx];d("category").textContent=x.cat;d("question").textContent=x.q;
d("answer").textContent=x.a;d("context").textContent=x.c;
d("pos").textContent=`Item ${{idx+1}} dari ${{ITEMS.length}}`;
const n=answered();d("progress").textContent=`${{n}} terjawab`;d("fill").style.width=(n/ITEMS.length*100)+"%";
d("options").innerHTML="";LABELS.forEach((o,i)=>{{const b=document.createElement("button");
b.className="option"+(judgments[x.id]===o[0]?" selected":"");b.innerHTML=`<kbd>${{i+1}}</kbd>${{o[1]}}`;
b.onclick=()=>choose(o[0]);d("options").appendChild(b)}});d("prev").disabled=idx===0;
d("next").disabled=idx===ITEMS.length-1;d("skip").textContent=judgments[x.id]==="skip"?"Dilewati":"Lewati"}}
function choose(v){{judgments[ITEMS[idx].id]=v;save();if(idx<ITEMS.length-1)idx++;render()}}
d("begin").onclick=()=>{{const first=ITEMS.findIndex(x=>!judgments[x.id]);idx=first<0?0:first;show("review");render()}};
d("clear").onclick=()=>{{if(confirm("Hapus semua progres?")){{judgments={{}};save()}}}};
d("prev").onclick=()=>{{if(idx>0)idx--;render()}};d("next").onclick=()=>{{if(idx<ITEMS.length-1)idx++;render()}};
d("skip").onclick=()=>{{judgments[ITEMS[idx].id]="skip";save();if(idx<ITEMS.length-1)idx++;render()}};
document.addEventListener("keydown",e=>{{if(d("review").classList.contains("hidden"))return;
const n=Number(e.key);if(n>=1&&n<=2)choose(LABELS[n-1][0]);else if(e.key==="ArrowLeft")d("prev").click();
else if(e.key==="ArrowRight")d("next").click();else if(e.key.toLowerCase()==="s")d("skip").click()}});
function metric(rows){{const correct=rows.reduce((n,x)=>n+x.correct,0);return{{n:rows.length,correct,
rate:rows.length?correct/rows.length:0}}}}
function score(){{const rows=ITEMS.filter(x=>judgments[x.id]&&judgments[x.id]!=="skip").map(x=>({{
item_id:x.id,subset:"subset_a",task:"qa_validity",category:x.cat,question:x.q,human_label:judgments[x.id],
expected_label:decode(x.p),item_origin:decode(x.o),qc_family:decode(x.f),
correct:judgments[x.id]===decode(x.p)?1:0,source_doc_id:x.d}}));
const auditRows=rows.filter(x=>x.item_origin==="audit_sample"),qcRows=rows.filter(x=>x.item_origin==="qc_control");
const by={{}};auditRows.forEach(x=>{{const g=by[x.category]||{{n:0,correct:0}};g.n++;g.correct+=x.correct;
by[x.category]=g}});Object.values(by).forEach(g=>g.confirmation_rate=g.n?g.correct/g.n:0);
return{{rows,audit:metric(auditRows),qc:metric(qcRows),skipped:ITEMS.length-rows.length,by}}}}
function results(){{last=score();d("resultmeta").textContent=`${{last.rows.length}} dinilai, ${{last.skipped}} dilewati, seed 42`;
d("stats").innerHTML=`<div class="stat"><div class="num ${{last.audit.rate>=.95?"good":"bad"}}">${{(last.audit.rate*100).toFixed(1)}}%</div>
<div>Konfirmasi audit</div></div><div class="stat"><div class="num ${{last.qc.rate>=.90?"good":"bad"}}">${{(last.qc.rate*100).toFixed(1)}}%</div><div>Deteksi QC</div></div>
<div class="stat"><div class="num">${{last.audit.correct}}/${{last.audit.n}}</div><div>Baris audit valid</div></div>
<div class="stat"><div class="num">${{last.qc.correct}}/${{last.qc.n}}</div><div>QC ditolak</div></div>`;
let h="<tr><th>Kategori</th><th>n</th><th>Dikonfirmasi valid</th><th>Rate</th></tr>";
Object.entries(last.by).forEach(([k,g])=>h+=`<tr><td>${{k}}</td><td>${{g.n}}</td><td>${{g.correct}}</td>
<td>${{(g.confirmation_rate*100).toFixed(1)}}%</td></tr>`);d("bycat").innerHTML=h;
const bad=last.rows.filter(x=>!x.correct);let z="<tr><th>Pertanyaan</th><th>Asal</th><th>Anda</th><th>Acuan</th></tr>";
bad.forEach(x=>z+=`<tr><td>${{escapeHtml(x.question)}}</td><td>${{x.item_origin}}</td><td>${{x.human_label}}</td><td>${{x.expected_label}}</td></tr>`);
if(!bad.length)z="<tr><td>Tidak ada ketidaksesuaian.</td></tr>";d("disagreements").innerHTML=z}}
function escapeHtml(s){{const e=document.createElement("div");e.textContent=s;return e.innerHTML}}
function download(name,text,type){{const u=URL.createObjectURL(new Blob([text],{{type}})),a=document.createElement("a");
a.href=u;a.download=name;a.click();URL.revokeObjectURL(u)}}
d("finish").onclick=()=>{{const n=answered();if(n<ITEMS.length&&!confirm(`Baru ${{n}} dari ${{ITEMS.length}} item dinilai. Lanjut?`))return;
results();show("results")}};d("back").onclick=()=>{{show("review");render()}};
d("csv").onclick=()=>{{if(!last)return;const esc=v=>'"'+String(v).replaceAll('"','""')+'"';
let s="item_id,subset,task,category,question,human_label,expected_label,correct,item_origin,qc_family,source_doc_id\\n";
last.rows.forEach(x=>s+=[x.item_id,x.subset,x.task,x.category,x.question,x.human_label,x.expected_label,x.correct,
x.item_origin,x.qc_family,x.source_doc_id].map(esc).join(",")+"\\n");
download("subset_a_blind_check.reviewed.csv",s,"text/csv;charset=utf-8")}};
d("summary").onclick=()=>{{if(!last)return;download("subset_a_blind_check.summary.json",JSON.stringify({{
generated_utc:new Date().toISOString(),dataset_version:"v2",subset:"a",seed:42,
sampling:"30 accepted rows (20% stratified) plus 10 controlled invalid QC variants",source_sha256:"{source_hash}",
n_pool:150,n_audit_sample:30,n_qc_controls:10,n_scored:last.rows.length,n_skipped:last.skipped,
audit_confirmation_rate:+last.audit.rate.toFixed(4),audit_confirmed:last.audit.correct,audit_scored:last.audit.n,
qc_detection_rate:+last.qc.rate.toFixed(4),qc_detected:last.qc.correct,qc_scored:last.qc.n,
by_category:last.by}},null,2),"application/json")}};
}})();
</script></body></html>
"""


def build(
    source: Path,
    queue_path: Path,
    key_path: Path,
    manifest_path: Path,
    html_path: Path,
) -> None:
    source_bytes = source.read_bytes()
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    rows = read_rows(source)
    if len(rows) != 150:
        raise ValueError(f"Expected 150 Subset A rows, found {len(rows)}")
    selected = sample_rows(rows)
    audit_rows = [
        {
            **row,
            "_origin": "audit_sample",
            "_expected_label": "valid",
            "_qc_family": "",
        }
        for row in selected
    ]
    controls = make_qc_controls(rows, selected)
    mixed = audit_rows + controls
    random.Random(SEED + 2).shuffle(mixed)

    keyed_rows = [
        {
            "item_id": item_id(row),
            "category": row["category"],
            "question": row["question"],
            "ground_truth_answer": row["ground_truth_answer"],
            "source_doc_id": row["source_doc_id"],
            "source_context": row["source_context"],
            "expected_label": row["_expected_label"],
            "origin": row["_origin"],
            "qc_family": row["_qc_family"],
        }
        for row in mixed
    ]
    review_rows = [
        {
            key: row[key]
            for key in (
                "item_id",
                "category",
                "question",
                "ground_truth_answer",
                "source_doc_id",
                "source_context",
            )
        }
        for row in keyed_rows
    ]
    if len({row["item_id"] for row in keyed_rows}) != len(keyed_rows):
        raise ValueError("Generated duplicate blind-check item IDs")

    write_csv(
        queue_path,
        [
            "item_id",
            "category",
            "question",
            "ground_truth_answer",
            "source_doc_id",
            "source_context",
        ],
        review_rows,
    )
    write_csv(
        key_path,
        ["item_id", "expected_label", "origin", "qc_family"],
        keyed_rows,
    )

    sample_counts = Counter(row["category"] for row in selected)
    qc_counts = Counter(row["category"] for row in controls)
    manifest = {
        "subset": "a",
        "dataset_version": "v2",
        "purpose": "blind human validation of Subset A QA-triplet quality",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_csv": str(source.relative_to(ROOT)),
        "source_sha256": source_hash,
        "source_row_count": len(rows),
        "audit_sample_row_count": len(selected),
        "audit_sample_fraction": len(selected) / len(rows),
        "qc_control_count": len(controls),
        "review_item_count": len(keyed_rows),
        "seed": SEED,
        "sampling": "20% stratified without replacement by stored category",
        "audit_sample_counts": dict(sample_counts),
        "qc_control_counts": dict(qc_counts),
        "qc_construction": {
            "answer_swap": 8,
            "false_ood": 2,
        },
        "methodological_scope": (
            "The 30 audit rows estimate human confirmation of accepted Subset A rows. "
            "The 10 controlled invalid variants measure QC detection separately. "
            "Controls are not genuine panel-rejected rows, and A item-level panel votes "
            "are unavailable."
        ),
        "review_queue": str(queue_path.relative_to(ROOT)),
        "sealed_key": str(key_path.relative_to(ROOT)),
        "browser_test": str(html_path.relative_to(ROOT)),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(render_html(keyed_rows, source_hash), encoding="utf-8")

    print(f"Source rows: {len(rows)}")
    print(f"Audit sample: {len(selected)} ({len(selected) / len(rows):.0%})")
    print(f"Audit by category: {dict(sample_counts)}")
    print(f"QC controls: {len(controls)} ({dict(qc_counts)})")
    print(f"Total review items: {len(keyed_rows)}")
    print(f"Review queue: {queue_path}")
    print(f"Sealed key: {key_path}")
    print(f"Browser test: {html_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--key", type=Path, default=DEFAULT_KEY)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--html", type=Path, default=DEFAULT_HTML)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build(args.source, args.queue, args.key, args.manifest, args.html)
