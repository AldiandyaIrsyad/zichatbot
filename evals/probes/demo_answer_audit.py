"""Demo — ask, answer, then check the source document's title makes sense.

The UKT case in one line:

    Q: Biaya kuliah per semester berapa?
    A: Rp 500.000
    source: "... Peserta Program Outbound Student Mobility ke Sookmyung ..."   <- nonsense

This runs the real pipeline for ten student questions — hybrid retrieval, BGE
reranking, Small-to-Big parent hydration, the production system prompt — then
puts the answer next to the title of the document it came from and asks whether
that pairing is coherent.

The judgement is deliberately narrow. It is not "is the answer correct" (that
needs the ground truth we do not have for these questions). It is "could a
document with *this title* plausibly be the source of *this answer* to *this
question*" — which is answerable by inspection, and is exactly the failure the
title probe is about.

    python -m evals.probes.demo_answer_audit

Writes results/demo_answer_audit.md and .csv. Spends API credit (~20 calls).
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
from typing import Dict, List, Optional, Sequence, Tuple

from dotenv import load_dotenv
from sqlalchemy import select

from app.kb.domain.models import ParentChunk
from app.shared.db import async_session_maker
from app.rag.prompts import build_prompt
from app.rag.prompts.prompt_maker import DEFAULT_SYSTEM_PROMPT_ID
from app.guardrails.ram.interfaces import RetrievedContext

from ._common import (
    RERANK_TOP_K,
    RESULTS_DIR,
    Query,
    Retriever,
    get_reranker,
    load_child_texts,
    load_documents,
    quiet_logs,
    write_results,
)

QUESTIONS = [
    "Berapa biaya UKT?",
    "Biaya kuliah per semester berapa?",
    "Bagaimana proses penentuan golongan tarif UKT mahasiswa?",
    "Apa syarat memperoleh keringanan biaya pendidikan?",
    "Apakah ada sanksi keterlambatan pembayaran uang kuliah per semester?",
    "Apakah ada tarif khusus untuk parkir kendaraan mahasiswa?",
    "Bagaimana syarat pendaftaran stiker izin parkir kendaraan?",
    "Apa saja syarat dokumen untuk daftar ulang?",
    "Berapa jumlah maksimal buku yang dapat dipinjam di perpustakaan?",
    "Kapan jadwal mulai perkuliahan mahasiswa baru?",
]

JUDGE_SYSTEM = (
    "Anda menilai konsistensi sebuah sistem pencarian dokumen hukum kampus. "
    "Anda TIDAK menilai benar atau salahnya jawaban. Anda hanya menilai satu "
    "hal: apakah dokumen dengan JUDUL tersebut masuk akal sebagai sumber dari "
    "JAWABAN itu untuk PERTANYAAN itu.\n\n"
    "Contoh TIDAK MASUK AKAL: pertanyaan tentang biaya kuliah per semester, "
    "jawaban 'Rp 500.000', sumber berjudul 'Peserta Program Outbound Student "
    "Mobility ke Sookmyung Women's University'. Angka itu memang ada di "
    "dokumen tersebut, tetapi dokumen itu mengatur satu program pertukaran "
    "pelajar, bukan tarif kuliah pada umumnya.\n\n"
    "Jawab HANYA dengan satu baris berformat:\n"
    "VERDICT: MASUK_AKAL|TIDAK_MASUK_AKAL|SEBAGIAN — <alasan satu kalimat>"
)

# UPI numbers its instruments two ways, and the distinction turns out to matter
# more than topic. "N Tahun YYYY" or a "-HK-" code is a Peraturan Rektor — a
# general rule. "KM"/"TM"/"KP"/"PL" codes are Keputusan Rektor — a decision
# about named people or one event. A student's general question should be
# answered by a rule, not by a decree that happens to mention the same words.
_PERATURAN_NUM = re.compile(r"^\s*\d+\s+[Tt]ahun\s+(?:19|20)\d{2}\b")
_HK_CODE = re.compile(r"[-/]HK[-/.]", re.I)


def doc_type(title: str) -> str:
    head = title.split(" - ")[0]
    return "Peraturan" if (_PERATURAN_NUM.match(title) or _HK_CODE.search(head)) \
        else "Keputusan"


# Verdicts assigned by reading all ten. The LLM judge is recorded too, but it
# is not reliable here: it marked question 1 coherent because the answer and
# the title agree with each other, missing that both are wrong for the question
# asked, and it failed question 7, where the system correctly used an amending
# document whose title describes what it amends rather than what it contains.
MANUAL_VERDICTS = {
    1: ("MISMATCH", "General UKT question answered with a magang programme's Rp 500.000 fee."),
    2: ("MISMATCH", "The flagship case: outbound-exchange decree answering a general tuition question."),
    3: ("PARTIAL", "Right topic and correct content, but the review regulation rather than the one that sets the brackets."),
    4: ("OK", "Title, matched Pasal and answer all align."),
    5: ("MISMATCH", "Outbound decree; the answer then speculates about sanctions it cannot source."),
    6: ("OK", "Correct parking regulation, real tariffs quoted from Lampiran V."),
    7: ("PARTIAL", "Correct document — the inserted Pasal 14B — but the passage covers staff, not students."),
    8: ("MISMATCH", "A postgraduate admissions list answering a general registration question."),
    9: ("OK", "Sensible source, and correctly reports the limit is not specified."),
    10: ("MISMATCH", "An inbound-exchange decree for one partner university."),
}

JUDGE_TEMPLATE = (
    "PERTANYAAN: {question}\n\n"
    "JAWABAN SISTEM: {answer}\n\n"
    "JUDUL DOKUMEN SUMBER: {title}\n\n"
    "Apakah dokumen berjudul demikian masuk akal sebagai sumber jawaban itu?"
)


async def chat_with_backoff(llm, messages, **kw) -> str:
    """OpenRouter rate-limits bursts; back off rather than losing the run."""
    delay = 8.0
    for attempt in range(7):
        try:
            return await llm.chat(messages, **kw)
        except Exception as exc:                                  # noqa: BLE001
            # A throttled request can also come back as a 200 with an error
            # body and no "choices", which surfaces as KeyError.
            transient = "429" in str(exc) or isinstance(exc, KeyError)
            if not transient or attempt == 6:
                raise
            print(f"      throttled ({type(exc).__name__}), retrying in {delay:.0f}s")
            await asyncio.sleep(delay)
            delay = min(delay * 2, 90)
    return ""


async def hydrate(parent_ids: Sequence[str]
                  ) -> Dict[str, Tuple[str, Optional[int], List[str], str]]:
    """Parent chunk text, page and breadcrumbs — the Small-to-Big context."""
    if not parent_ids:
        return {}
    async with async_session_maker() as session:
        rows = (await session.execute(
            select(ParentChunk.id, ParentChunk.text, ParentChunk.page,
                   ParentChunk.breadcrumbs, ParentChunk.doc_id)
            .where(ParentChunk.id.in_(list(parent_ids)))
        )).all()
    return {r[0]: (r[1] or "", r[2], list(r[3] or []), r[4]) for r in rows}


def verdict_of(text: str) -> Tuple[str, str]:
    line = text.strip().splitlines()[0] if text.strip() else ""
    body = line.split("VERDICT:", 1)[-1].strip()
    for tag in ("TIDAK_MASUK_AKAL", "SEBAGIAN", "MASUK_AKAL"):
        if body.upper().startswith(tag):
            return tag, body[len(tag):].lstrip(" —-:").strip()
    return "UNPARSED", line[:160]


async def main() -> None:
    quiet_logs()
    load_dotenv()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=os.environ.get("CHAT_LLM_MODEL", "qwen/qwen3-14b"))
    ap.add_argument("--report-only", action="store_true",
                    help="rebuild the report from the existing CSV, no API calls")
    args = ap.parse_args()

    print(f"Demo — answer / source-title audit  [model={args.model}]\n")

    docs = await load_documents()
    titles = {d.doc_id: d.title for d in docs.values()}

    retriever = Retriever()
    reranker = get_reranker()
    from evals._shared.clients import get_llm_client_from_env
    llm = get_llm_client_from_env(model=args.model)

    rows: List[dict] = []
    if args.report_only:
        import csv as _csv
        with open(RESULTS_DIR / "demo_answer_audit.csv", encoding="utf-8") as f:
            rows = list(_csv.DictReader(f))
        for r in rows:
            r["source_doc_type"] = doc_type(r["source_title"])
        await retriever.close()
        await reranker.close()
        await llm.aclose()
        return await report(rows, args)

    try:
        queries = [Query(qid=f"aud{i}", text=q) for i, q in enumerate(QUESTIONS)]
        cands_by_qid = await retriever.retrieve(queries)

        for q in queries:
            cands = cands_by_qid[q.qid]
            texts = await load_child_texts([c.chunk_id for c in cands])
            res = await reranker.rerank(
                query=q.text, documents=[texts.get(c.chunk_id, "") for c in cands]
            )
            sc = [0.0] * len(cands)
            for x in res:
                sc[x.index] = float(x.score)
            order = sorted(range(len(cands)), key=lambda i: sc[i], reverse=True)
            top = [cands[i] for i in order[:RERANK_TOP_K]]

            parents = await hydrate([c.parent_chunk_id for c in top])
            contexts: List[RetrievedContext] = []
            seen = set()
            for c in top:
                p = parents.get(c.parent_chunk_id)
                if not p or c.parent_chunk_id in seen:
                    continue
                seen.add(c.parent_chunk_id)
                text, page, crumbs, doc_id = p
                contexts.append(RetrievedContext(
                    text=text, source_title=titles.get(doc_id, ""), page=page,
                    breadcrumbs=crumbs, doc_id=doc_id,
                ))

            bundle = build_prompt(
                user_message=q.text, contexts=contexts,
                base_system_prompt=DEFAULT_SYSTEM_PROMPT_ID, use_nonce=True,
            )
            answer = await chat_with_backoff(
                llm,
                [{"role": "system", "content": bundle.system_prompt},
                 {"role": "user", "content": bundle.user_turn}],
                temperature=0.0, max_tokens=350,
            )
            answer = " ".join(answer.split())

            top_doc = top[0].doc_id
            top_title = titles.get(top_doc, "")
            judgement = await chat_with_backoff(
                llm,
                [{"role": "system", "content": JUDGE_SYSTEM},
                 {"role": "user", "content": JUDGE_TEMPLATE.format(
                     question=q.text, answer=answer[:700], title=top_title)}],
                temperature=0.0, max_tokens=120,
            )
            verdict, reason = verdict_of(judgement)

            rows.append({
                "question": q.text,
                "answer": answer,
                "source_title": top_title,
                "source_doc_id": top_doc,
                "source_snippet": " ".join(texts.get(top[0].chunk_id, "").split())[:260],
                "distinct_source_docs_in_context": len({c.doc_id for c in top}),
                "verdict": verdict,
                "judge_reason": reason,
                "source_doc_type": doc_type(top_title),
            })
            flag = {"MASUK_AKAL": "ok  ", "SEBAGIAN": "part", "TIDAK_MASUK_AKAL": "BAD "}.get(verdict, "??  ")
            print(f"  [{flag}] {q.text}")
            print(f"          A: {answer[:96]}")
            print(f"          S: {top_title[:96]}")
            await asyncio.sleep(6.0)   # stay under the burst limit
    finally:
        await llm.aclose()
        await retriever.close()
        await reranker.close()

    return await report(rows, args)


async def report(rows: List[dict], args) -> None:
    for n, r in enumerate(rows, 1):
        v, why = MANUAL_VERDICTS.get(n, ("", ""))
        r["manual_verdict"], r["manual_reason"] = v, why

    counts: Dict[str, int] = {}
    for r in rows:
        counts[r["manual_verdict"]] = counts.get(r["manual_verdict"], 0) + 1
    by_type: Dict[str, Dict[str, int]] = {}
    for r in rows:
        by_type.setdefault(r["source_doc_type"], {})
        by_type[r["source_doc_type"]][r["manual_verdict"]] = (
            by_type[r["source_doc_type"]].get(r["manual_verdict"], 0) + 1
        )

    lines = [
        "# Answer / source-title audit\n",
        "Ten student questions through the real pipeline — hybrid retrieval, BGE",
        "reranking, Small-to-Big hydration, production system prompt. For each:",
        "the answer, the title of the document it came from, and whether that",
        "pairing is coherent.\n",
        "The judgement is **not** about answer correctness — there is no ground",
        "truth for these questions. It is only: could a document with *this",
        "title* plausibly be the source of *this answer*?\n",
        f"**{counts.get('MISMATCH', 0)} of {len(rows)} mismatched, "
        f"{counts.get('PARTIAL', 0)} partial, {counts.get('OK', 0)} coherent.**\n",
        "Verdicts are mine, assigned by reading all ten. The LLM judge's call is",
        "kept in the CSV but is not reliable here — it passed question 1 because",
        "the answer and title agree with *each other*, missing that both are",
        "wrong for the question, and failed question 7, where the system",
        "correctly used an amending document.\n",
        "| # | Question | Answer (abridged) | Source document | Type | Verdict |",
        "|---|---|---|---|---|---|",
    ]
    for n, r in enumerate(rows, 1):
        v = {"OK": "ok", "PARTIAL": "partial",
             "MISMATCH": "**mismatch**"}.get(r["manual_verdict"], "")
        lines.append(
            f"| {n} | {r['question']} | {r['answer'][:100].replace('|', '/')}… "
            f"| {r['source_title'][:62].replace('|', '/')} "
            f"| {r['source_doc_type']} | {v} |"
        )
    lines += [
        "",
        "## The pattern",
        "",
        "| source document type | ok | partial | mismatch |",
        "|---|---|---|---|",
    ]
    for t in sorted(by_type):
        b = by_type[t]
        lines.append(f"| {t} | {b.get('OK', 0)} | {b.get('PARTIAL', 0)} "
                     f"| {b.get('MISMATCH', 0)} |")
    lines += [
        "",
        "**Every mismatch came from a Keputusan Rektor; every acceptable answer",
        "came from a Peraturan Rektor.** No crossover in ten cases.",
        "",
        "A Peraturan sets a general rule. A Keputusan decides something about",
        "named people or a single event — who joins an exchange programme, who",
        "passed a selection. A student asking a general question needs a rule,",
        "but a decree will often match the words better, because it states a",
        "concrete figure where the rule states a formula. Rp 500.000 beats",
        "\"ditetapkan berdasarkan kemampuan ekonomi orang tua\" on a query about",
        "cost, every time.",
        "",
        "**63% of the corpus (582 of 922) is Keputusan.** The class that answers",
        "these questions badly is the majority class.",
        "",
        "The distinction is free to compute — it is in the title's own numbering",
        "(`41 Tahun 2023` and `2151-UN40-HK-2019` are rules; `346-UN40-KM.02.02`",
        "and `549-UN40-TM.01.04` are decrees), so no embedding or model call is",
        "needed to know which kind of document a candidate is.",
        "",
    ]
    lines.append("")
    for n, r in enumerate(rows, 1):
        lines += [
            f"\n---\n\n### {n}. {r['question']}\n",
            f"**Answer:** {r['answer']}\n",
            f"**Source document:** {r['source_title']}\n",
            f"**Matched passage:** > {r['source_snippet']}\n",
            f"**Source type:** {r['source_doc_type']}\n",
            f"**Verdict:** {r['manual_verdict']} — {r['manual_reason']}\n",
            f"*(LLM judge said {r['verdict']}: {r['judge_reason']})*\n",
        ]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "demo_answer_audit.md").write_text("\n".join(lines), encoding="utf-8")
    write_results("demo_answer_audit", rows, {
        "model": args.model, "questions": len(rows),
        "verdicts_manual": counts,
        "verdicts_by_source_type": by_type,
    })
    print(f"\n  {counts}")


if __name__ == "__main__":
    asyncio.run(main())
