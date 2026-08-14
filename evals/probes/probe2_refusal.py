"""Probe 2 — can the LLM be made to abstain with a fixed, recognisable sentence?

Today three refusal strings exist (app/chat/config.py:139-160) and all three
come from guardrails: unsafe input, the relevance judge, or an internal error.
They are tagged on the wire as ``reason`` in {unsafe, irrelevant, internal}
(chat_service.py:570). There is **no** string and no tag for the fourth case —
retrieval returned something, the guardrails let it through, and the model had
nothing useful to work with. It then writes free-form Indonesian prose that is
indistinguishable from a real answer.

The current system prompt asks the model to *say* it lacks the information but
never dictates the wording (prompt_maker.py:27-31). This probe tests whether
mandating one exact sentence works, and — the number that actually decides it —
whether mandating it makes the model refuse when it should have answered.

    3 context conditions x 2 prompts

    relevant    the gold chunk for the question      -> must still answer
    mismatched  chunks from an unrelated document    -> should abstain
    empty       no context at all                    -> should abstain

    P0  current DEFAULT_SYSTEM_PROMPT_ID
    P1  same, with the abstention clause replaced by a mandated sentence

    python -m evals.probes.probe2_refusal [--limit N] [--model M]

Appends to results/probe2_refusal.csv as it goes, so a crash does not lose the
API spend already made. Re-running skips cells already present.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from dotenv import load_dotenv

from app.rag.prompts import build_prompt
from app.rag.prompts.prompt_maker import DEFAULT_SYSTEM_PROMPT_ID
from app.guardrails.ram.interfaces import RetrievedContext

from ._common import (
    RESULTS_DIR,
    Query,
    load_child_texts,
    load_documents,
    load_subset_a_queries,
    quiet_logs,
    write_results,
)

SENTINEL = "Maaf, saya tidak menemukan dokumen relevan."

# The clause in the production prompt that leaves the wording open.
_OPEN_CLAUSE = (
    "Gunakan informasi yang relevan dari konteks tersebut sebaik mungkin "
    "untuk menjawab, meskipun tidak lengkap, dan sebutkan secara singkat "
    "jika ada bagian yang tidak tercakup dalam konteks. Katakan bahwa Anda "
    "tidak memiliki informasi tersebut hanya jika konteks yang diberikan "
    "benar-benar tidak berkaitan dengan pertanyaan."
)

_MANDATED_CLAUSE = (
    "Gunakan informasi yang relevan dari konteks tersebut sebaik mungkin "
    "untuk menjawab, meskipun tidak lengkap, dan sebutkan secara singkat "
    "jika ada bagian yang tidak tercakup dalam konteks. Jika konteks yang "
    "diberikan benar-benar tidak berkaitan dengan pertanyaan, atau tidak ada "
    "konteks sama sekali, jawab HANYA dengan kalimat berikut secara persis "
    f"tanpa tambahan apa pun: \"{SENTINEL}\""
)

PROMPTS = {
    "P0_current": DEFAULT_SYSTEM_PROMPT_ID,
    "P1_mandated": DEFAULT_SYSTEM_PROMPT_ID.replace(_OPEN_CLAUSE, _MANDATED_CLAUSE),
}

CONDITIONS = ("relevant", "mismatched", "empty")

# Free-form abstention, for measuring what P0 does today.
_ABSTAIN_RE = re.compile(
    r"tidak\s+menemukan|tidak\s+memiliki\s+informasi|tidak\s+tercakup|"
    r"tidak\s+terdapat\s+informasi|tidak\s+ada\s+informasi|"
    r"tidak\s+dapat\s+menjawab|tidak\s+berkaitan",
    re.IGNORECASE,
)

CSV_PATH = RESULTS_DIR / "probe2_refusal.csv"
FIELDS = [
    "qid", "condition", "prompt", "question", "gold_doc_id",
    "context_doc_id", "n_contexts", "response",
    "exact_sentinel", "any_abstention",
]


def refused_outright(text: str) -> bool:
    """The whole response is a refusal, not an answer that happens to note a gap.

    ``any_abstention`` alone over-counts badly: a correct, complete answer may
    still say "bagian X tidak tercakup dalam konteks". Only a short response
    that is essentially just the refusal counts here.
    """
    stripped = text.strip()
    return is_exact(text) or (
        bool(_ABSTAIN_RE.search(stripped)) and len(stripped) < 200
    )


def is_exact(text: str) -> bool:
    """The response is the mandated sentence and essentially nothing else."""
    stripped = text.strip().strip('"').strip()
    return stripped == SENTINEL or (
        stripped.startswith(SENTINEL) and len(stripped) <= len(SENTINEL) + 8
    )


def load_done() -> set:
    if not CSV_PATH.exists():
        return set()
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        return {(r["qid"], r["condition"], r["prompt"]) for r in csv.DictReader(f)}


def append_row(row: Dict[str, object]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    exists = CSV_PATH.exists()
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if not exists:
            w.writeheader()
        w.writerow(row)


async def build_contexts(
    queries: Sequence[Query], titles: Dict[str, str], rng: random.Random
) -> Dict[str, Dict[str, List[RetrievedContext]]]:
    """Gold contexts per query, plus a mismatched set from another document."""
    from sqlalchemy import select

    from app.kb.domain.models import ParentChunk
    from app.shared.db import async_session_maker

    gold_ids = [q.gold_doc_ids[0] for q in queries]
    async with async_session_maker() as session:
        rows = (
            await session.execute(
                select(ParentChunk.doc_id, ParentChunk.text, ParentChunk.page,
                       ParentChunk.breadcrumbs)
                .where(ParentChunk.doc_id.in_(list(set(gold_ids))))
            )
        ).all()

    by_doc: Dict[str, List[RetrievedContext]] = {}
    for doc_id, text, page, breadcrumbs in rows:
        by_doc.setdefault(doc_id, []).append(
            RetrievedContext(
                text=text or "", source_title=titles.get(doc_id, ""),
                page=page, breadcrumbs=list(breadcrumbs or []), doc_id=doc_id,
            )
        )

    out: Dict[str, Dict[str, List[RetrievedContext]]] = {}
    pool = [d for d in by_doc if by_doc[d]]
    for q in queries:
        gold = q.gold_doc_ids[0]
        others = [d for d in pool if d != gold]
        wrong = rng.choice(others) if others else gold
        # The gold passage from subset_a, not an arbitrary slice of the gold
        # document: if the answer is not actually in the context, an abstention
        # is correct and the over-refusal number becomes meaningless.
        relevant = [RetrievedContext(
            text=q.gold_context, source_title=titles.get(gold, ""), doc_id=gold,
        )] if q.gold_context else by_doc.get(gold, [])[:3]
        out[q.qid] = {
            "relevant": relevant,
            "mismatched": by_doc.get(wrong, [])[:3],
            "empty": [],
        }
    return out


async def main() -> None:
    quiet_logs()
    load_dotenv()

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=40,
                    help="questions sampled from subset_a (default 40)")
    ap.add_argument("--model", default=os.environ.get("CHAT_LLM_MODEL", "qwen/qwen3-14b"))
    args = ap.parse_args()

    print(f"Probe 2 — refusal prompt  [model={args.model}]\n")

    docs = await load_documents()
    titles = {d.doc_id: d.title for d in docs.values()}

    rng = random.Random(42)
    all_q = [q for q in load_subset_a_queries() if q.gold_doc_ids]
    queries = rng.sample(all_q, min(args.limit, len(all_q)))
    contexts = await build_contexts(queries, titles, rng)

    done = load_done()
    todo = [
        (q, cond, pname)
        for q in queries for cond in CONDITIONS for pname in PROMPTS
        if (q.qid, cond, pname) not in done
    ]
    print(f"  {len(queries)} questions x {len(CONDITIONS)} conditions x "
          f"{len(PROMPTS)} prompts = {len(queries)*6} cells")
    print(f"  {len(done)} already done, {len(todo)} to run\n")

    if todo:
        from evals._shared.clients import get_llm_client_from_env

        llm = get_llm_client_from_env(model=args.model)
        try:
            for n, (q, cond, pname) in enumerate(todo, 1):
                ctxs = contexts[q.qid][cond]
                bundle = build_prompt(
                    user_message=q.text, contexts=ctxs,
                    base_system_prompt=PROMPTS[pname], use_nonce=True,
                )
                try:
                    resp = await llm.chat(
                        [{"role": "system", "content": bundle.system_prompt},
                         {"role": "user", "content": bundle.user_turn}],
                        temperature=0.0, max_tokens=400,
                    )
                except Exception as exc:                      # noqa: BLE001
                    print(f"    call failed for {q.qid}/{cond}/{pname}: {exc}")
                    continue
                append_row({
                    "qid": q.qid, "condition": cond, "prompt": pname,
                    "question": q.text, "gold_doc_id": q.gold_doc_ids[0],
                    "context_doc_id": ctxs[0].doc_id if ctxs else "",
                    "n_contexts": len(ctxs),
                    "response": " ".join(resp.split()),
                    "exact_sentinel": int(is_exact(resp)),
                    "any_abstention": int(bool(_ABSTAIN_RE.search(resp))),
                })
                if n % 20 == 0:
                    print(f"    {n}/{len(todo)}")
        finally:
            await llm.aclose()

    # --- aggregate ----------------------------------------------------------
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    cells: Dict[str, Dict[str, dict]] = {}
    for pname in PROMPTS:
        cells[pname] = {}
        for cond in CONDITIONS:
            sel = [r for r in rows if r["prompt"] == pname and r["condition"] == cond]
            if not sel:
                continue
            n = len(sel)
            cells[pname][cond] = {
                "n": n,
                "exact_sentinel": round(sum(int(r["exact_sentinel"]) for r in sel) / n, 4),
                "any_abstention": round(sum(int(r["any_abstention"]) for r in sel) / n, 4),
                "refused_outright": round(
                    sum(refused_outright(r["response"]) for r in sel) / n, 4
                ),
            }

    over_refusal = {
        pname: cells[pname].get("relevant", {}).get("refused_outright", 0.0)
        for pname in PROMPTS
    }

    summary = {
        "model": args.model,
        "questions": len(queries),
        "cells": cells,
        "over_refusal_on_relevant_context": over_refusal,
        "verdict_inputs": {
            "P1_exact_on_empty": cells.get("P1_mandated", {}).get("empty", {}).get("exact_sentinel"),
            "P1_exact_on_mismatched": cells.get("P1_mandated", {}).get("mismatched", {}).get("exact_sentinel"),
            "P0_exact_on_empty": cells.get("P0_current", {}).get("empty", {}).get("exact_sentinel"),
            "P1_minus_P0_over_refusal": round(
                over_refusal.get("P1_mandated", 0) - over_refusal.get("P0_current", 0), 4
            ),
        },
    }
    write_results("probe2_refusal", [], summary)

    print(f"\n  {'prompt':14s} {'condition':12s} {'n':>4s} {'exact':>8s} {'refused':>8s} {'anyment':>8s}")
    for pname in PROMPTS:
        for cond in CONDITIONS:
            c = cells[pname].get(cond)
            if c:
                print(f"  {pname:14s} {cond:12s} {c['n']:4d} "
                      f"{c['exact_sentinel']:8.1%} {c['refused_outright']:8.1%} "
                      f"{c['any_abstention']:8.1%}")

    print(f"\n  Over-refusal on RELEVANT context (lower is better)")
    for pname, v in over_refusal.items():
        print(f"    {pname:14s} {v:.1%}")

    print("\n  Sample responses:")
    seen = set()
    for r in rows:
        key = (r["prompt"], r["condition"])
        if key in seen:
            continue
        seen.add(key)
        print(f"\n    [{r['prompt']} / {r['condition']}] {r['response'][:150]}")


if __name__ == "__main__":
    asyncio.run(main())
