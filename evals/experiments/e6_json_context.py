"""E6 — give the LLM the document title and type, as structured context.

Today the context block is flat prose (`prompt_maker.build_context_block`):
``Sumber 1`` plus page and breadcrumbs, and **the document title is
deliberately excluded**. So when a decree about one exchange programme is
retrieved, the model has no way to know that is what it is reading — it sees
"Rp 500.000 per semester" with no indication of whose semester.

Three ways of presenting the same retrieved context:

    C1 flat            what production does now
    C2 json            JSON records carrying judul_dokumen and jenis_dokumen
    C3 json + rule     the same, plus one instruction: a Keputusan about a
                       named programme cannot answer a general question, so
                       say who it applies to or say you do not have the rule

Run on the **baseline** retrieval — the hard case, where a decree is on top —
so this measures whether generation can recover from a retrieval mistake. Then
on the best retrieval (E3+E4b) to see whether the two compose.

    python -m evals.experiments.e6_json_context

Writes results/e6_json_context.json. ~60 LLM calls.
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Dict, List, Sequence, Tuple

from app.rag.prompts import build_prompt
from app.rag.prompts.prompt_maker import DEFAULT_SYSTEM_PROMPT_ID
from app.guardrails.ram.interfaces import RetrievedContext

from evals.probes._common import RESULTS_DIR, quiet_logs
from ._bench import MODEL, QUESTIONS, Bench, doc_type
from .e2_judge import JUDGE_V2, TEMPLATE as JUDGE_TEMPLATE, parse as parse_verdict

TYPE_RULE = (
    "\n\nSetiap sumber memiliki 'jenis_dokumen'. 'Peraturan' adalah aturan umum "
    "yang berlaku bagi semua mahasiswa. 'Keputusan' adalah keputusan "
    "administratif mengenai orang atau kegiatan tertentu saja — misalnya "
    "peserta satu program pertukaran pelajar. Angka atau ketentuan di dalam "
    "sebuah 'Keputusan' HANYA berlaku bagi orang atau kegiatan yang disebut "
    "dalam dokumen itu. Jika pertanyaannya bersifat umum dan Anda hanya "
    "memiliki 'Keputusan', jangan sajikan isinya seolah-olah berlaku umum: "
    "sebutkan secara eksplisit untuk siapa ketentuan itu berlaku, atau nyatakan "
    "bahwa Anda tidak menemukan aturan umum yang mengaturnya."
)


def json_block(contexts: Sequence[RetrievedContext]) -> str:
    records = [
        {
            "sumber": i,
            "judul_dokumen": c.source_title,
            "jenis_dokumen": doc_type(c.source_title),
            "halaman": c.page,
            "bagian": " > ".join(c.breadcrumbs) if c.breadcrumbs else None,
            "isi": " ".join((c.text or "").split()),
        }
        for i, c in enumerate(contexts, 1)
    ]
    return json.dumps(records, ensure_ascii=False, indent=1)


def build_json_prompt(question: str, contexts: Sequence[RetrievedContext],
                      with_rule: bool) -> Tuple[str, str]:
    system = DEFAULT_SYSTEM_PROMPT_ID + (TYPE_RULE if with_rule else "")
    user = (
        "Konteks (JSON):\n" + json_block(contexts) +
        f"\n\nPertanyaan:\n{question}"
    )
    return system, user


# Does the answer say who a decree's terms apply to, rather than stating them
# as general fact?
_SCOPED = re.compile(
    r"peserta program|program magang|outbound|inbound|pertukaran|"
    r"hanya berlaku|khusus (?:bagi|untuk)|mahasiswa peserta|"
    r"tidak menemukan|tidak terdapat aturan umum|tidak disebutkan",
    re.I,
)


async def generate(llm, system: str, user: str) -> str:
    for attempt in range(5):
        try:
            r = await llm.chat(
                [{"role": "system", "content": system},
                 {"role": "user", "content": user}],
                temperature=0.0, max_tokens=400)
            return " ".join(r.split())
        except Exception as exc:                                  # noqa: BLE001
            if attempt == 4:
                raise
            await asyncio.sleep(6 * (attempt + 1))
    return ""


async def main() -> None:
    quiet_logs()
    from dotenv import load_dotenv

    load_dotenv(".env")
    from evals._shared.clients import get_llm_client_from_env

    print(f"E6 — structured context  [{MODEL}]\n")

    bench = Bench()
    await bench.setup()
    for q in bench.queries:
        await bench.rerank_scores(q)

    # Baseline retrieval: the hard case, a decree often on top.
    contexts_by_q: Dict[str, List[RetrievedContext]] = {}
    for q in bench.queries:
        cs = bench.candidates(q)
        sc = bench._rerank[f"plain::{q.qid}"]
        order = sorted(range(len(cs)), key=lambda i: sc[i], reverse=True)[:8]
        contexts_by_q[q.qid] = await bench.hydrate([cs[i] for i in order])

    llm = get_llm_client_from_env(model=MODEL)
    rows: List[dict] = []
    try:
        for n, q in enumerate(bench.queries):
            ctxs = contexts_by_q[q.qid]
            top_title = ctxs[0].source_title if ctxs else ""
            row = {"question": q.text, "top_title": top_title,
                   "top_type": doc_type(top_title)}

            bundle = build_prompt(user_message=q.text, contexts=ctxs,
                                  base_system_prompt=DEFAULT_SYSTEM_PROMPT_ID,
                                  use_nonce=True)
            row["C1_flat"] = await generate(llm, bundle.system_prompt, bundle.user_turn)
            await asyncio.sleep(1.5)

            for tag, rule in (("C2_json", False), ("C3_json_rule", True)):
                sysp, userp = build_json_prompt(q.text, ctxs, rule)
                row[tag] = await generate(llm, sysp, userp)
                await asyncio.sleep(1.5)

            for tag in ("C1_flat", "C2_json", "C3_json_rule"):
                row[f"{tag}_scoped"] = bool(_SCOPED.search(row[tag]))
                jr = await generate(
                    llm, JUDGE_V2,
                    JUDGE_TEMPLATE.format(question=q.text,
                                          answer=row[tag][:700], title=top_title))
                row[f"{tag}_verdict"] = parse_verdict(jr)[0]
                await asyncio.sleep(1.5)

            rows.append(row)
            print(f"  {n+1:2d}. [{row['top_type'][:4]}] {q.text[:44]}")
            for tag in ("C1_flat", "C2_json", "C3_json_rule"):
                flag = "scoped" if row[f"{tag}_scoped"] else "      "
                print(f"        {tag:13s} {flag} {row[f'{tag}_verdict'][:16]:16s} "
                      f"{row[tag][:70]}")
    finally:
        await llm.aclose()
        await bench.close()

    hard = [r for r in rows if r["top_type"] == "Keputusan"]
    summary = {
        "model": MODEL,
        "retrieval": "baseline (production ranking)",
        "questions": len(rows),
        "questions_with_decree_on_top": len(hard),
        "conditions": {},
    }
    for tag in ("C1_flat", "C2_json", "C3_json_rule"):
        summary["conditions"][tag] = {
            "scoped_or_declined_all": sum(1 for r in rows if r[f"{tag}_scoped"]),
            "scoped_or_declined_when_decree_on_top":
                sum(1 for r in hard if r[f"{tag}_scoped"]),
            "judge_ok": sum(1 for r in rows if r[f"{tag}_verdict"] == "MASUK_AKAL"),
            "judge_mismatch": sum(1 for r in rows
                                  if r[f"{tag}_verdict"] == "TIDAK_MASUK_AKAL"),
        }

    print(f"\n  of the {len(hard)} questions where a decree is on top, how often "
          f"did the answer say who it applies to (or decline)?")
    for tag in ("C1_flat", "C2_json", "C3_json_rule"):
        c = summary["conditions"][tag]
        print(f"    {tag:13s} {c['scoped_or_declined_when_decree_on_top']}/{len(hard)}"
              f"   (judge flagged {c['judge_mismatch']}/10)")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "e6_json_context.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=2, ensure_ascii=False),
        encoding="utf-8")
    print(f"\n  Written to {RESULTS_DIR / 'e6_json_context.json'}")


if __name__ == "__main__":
    asyncio.run(main())
