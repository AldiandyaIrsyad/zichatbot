"""Shared harness for the ten audited questions.

Every experiment scores the same ten questions the same way, so their numbers
can be compared. What varies is the *intervention*: a function that reorders
the retrieved candidates, and optionally a different way of presenting them to
the LLM.

The scoring target is deliberately not "is the answer correct" — there is no
ground truth for these questions. It is "did the answer come from a document
whose title makes sense as its source", which is the failure being fixed and is
judgeable by inspection. The ten manual verdicts in
``demo_answer_audit.MANUAL_VERDICTS`` are the reference labels.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from sqlalchemy import select

from app.kb.domain.models import ParentChunk
from app.shared.db import async_session_maker
from app.guardrails.ram.interfaces import RetrievedContext

from evals.probes._common import (
    RERANK_TOP_K,
    Candidate,
    Query,
    Retriever,
    get_reranker,
    load_child_texts,
    load_documents,
)

MODEL = "deepseek/deepseek-v4-flash-0731"

# The ten audited questions, with the document a competent librarian would
# reach for. "" means no single obviously-right document exists in the corpus,
# in which case the honest outcome is a Peraturan on the topic or an abstention.
QUESTIONS: List[Tuple[str, str]] = [
    ("Berapa biaya UKT?", "003 Tahun 2022"),
    ("Biaya kuliah per semester berapa?", "003 Tahun 2022"),
    ("Bagaimana proses penentuan golongan tarif UKT mahasiswa?", "003 Tahun 2022"),
    ("Apa syarat memperoleh keringanan biaya pendidikan?", "41 Tahun 2023"),
    ("Apakah ada sanksi keterlambatan pembayaran uang kuliah per semester?", ""),
    ("Apakah ada tarif khusus untuk parkir kendaraan mahasiswa?", "2151-UN40-HK-2019"),
    ("Bagaimana syarat pendaftaran stiker izin parkir kendaraan?", "2151-UN40-HK-2019"),
    ("Apa saja syarat dokumen untuk daftar ulang?", ""),
    ("Berapa jumlah maksimal buku yang dapat dipinjam di perpustakaan?", ""),
    ("Kapan jadwal mulai perkuliahan mahasiswa baru?", ""),
]

# UPI numbers its instruments two ways and the distinction predicts the failure:
# every mismatch in the audit came from a Keputusan, every good answer from a
# Peraturan. "N Tahun YYYY" or a "-HK-" code is a Peraturan (a general rule);
# KM / TM / KP / PL codes are Keputusan (a decision about named people).
_PERATURAN_NUM = re.compile(r"^\s*\d+\s+[Tt]ahun\s+(?:19|20)\d{2}\b")
_HK_CODE = re.compile(r"[-/]HK[-/.]", re.I)


def doc_type(title: str) -> str:
    head = title.split(" - ")[0]
    return "Peraturan" if (_PERATURAN_NUM.match(title) or _HK_CODE.search(head)) \
        else "Keputusan"


@dataclass
class Scored:
    """One candidate with the score the intervention gave it."""

    cand: Candidate
    score: float


@dataclass
class RunResult:
    question: str
    expected_prefix: str
    top_doc_id: str
    top_title: str
    top_type: str
    top_snippet: str
    contexts: List[RetrievedContext] = field(default_factory=list)
    doc_ranking: List[str] = field(default_factory=list)
    answer: str = ""
    judge_verdict: str = ""
    judge_reason: str = ""

    @property
    def found_expected(self) -> Optional[bool]:
        """Did the expected document win? None when no expectation is set."""
        if not self.expected_prefix:
            return None
        return self.top_title.startswith(self.expected_prefix)


# An intervention receives the query, its candidates, and a context bundle of
# lookups, and returns a score per candidate. Higher is better.
Intervention = Callable[[Query, Sequence[Candidate], dict], List[float]]


class Bench:
    """Retrieves once, then lets any number of interventions reorder the same
    candidates. Retrieval and rerank results are cached across experiments."""

    def __init__(self) -> None:
        self.retriever = Retriever()
        self.reranker = get_reranker()
        self.titles: Dict[str, str] = {}
        self.child_text: Dict[str, str] = {}
        self._cands: Dict[str, List[Candidate]] = {}
        self._rerank: Dict[str, List[float]] = {}
        self.queries = [
            Query(qid=f"e{i}", text=q) for i, (q, _) in enumerate(QUESTIONS)
        ]
        self.expected = {f"e{i}": exp for i, (_, exp) in enumerate(QUESTIONS)}

    async def setup(self) -> None:
        docs = await load_documents()
        self.titles = {d.doc_id: d.title for d in docs.values()}
        self._cands = await self.retriever.retrieve(self.queries)
        ids = sorted({c.chunk_id for cs in self._cands.values() for c in cs})
        self.child_text = await load_child_texts(ids)

    def candidates(self, q: Query) -> List[Candidate]:
        return self._cands[q.qid]

    async def rerank_scores(self, q: Query, key: str = "plain",
                            docs: Optional[Sequence[str]] = None) -> List[float]:
        """Cross-encoder scores in candidate order, cached per (query, key)."""
        ck = f"{key}::{q.qid}"
        if ck in self._rerank:
            return self._rerank[ck]
        cands = self.candidates(q)
        payload = list(docs) if docs is not None else [
            self.child_text.get(c.chunk_id, "") for c in cands
        ]
        res = await self.reranker.rerank(query=q.text, documents=payload)
        scores = [0.0] * len(cands)
        for r in res:
            if 0 <= r.index < len(scores):
                scores[r.index] = float(r.score)
        self._rerank[ck] = scores
        return scores

    async def hydrate(self, cands: Sequence[Candidate]) -> List[RetrievedContext]:
        """Small-to-Big: the parent chunks behind the winning children."""
        pids = [c.parent_chunk_id for c in cands]
        if not pids:
            return []
        async with async_session_maker() as session:
            rows = (await session.execute(
                select(ParentChunk.id, ParentChunk.text, ParentChunk.page,
                       ParentChunk.breadcrumbs, ParentChunk.doc_id)
                .where(ParentChunk.id.in_(list(pids)))
            )).all()
        by_id = {r[0]: r for r in rows}
        out, seen = [], set()
        for c in cands:
            row = by_id.get(c.parent_chunk_id)
            if not row or c.parent_chunk_id in seen:
                continue
            seen.add(c.parent_chunk_id)
            out.append(RetrievedContext(
                text=row[1] or "", source_title=self.titles.get(row[4], ""),
                page=row[2], breadcrumbs=list(row[3] or []), doc_id=row[4],
            ))
        return out

    async def run(self, name: str, intervention: Intervention,
                  top_k: int = RERANK_TOP_K) -> List[RunResult]:
        """Score every question under one intervention. No LLM calls."""
        results: List[RunResult] = []
        for q in self.queries:
            cands = self.candidates(q)
            ctx = {
                "bench": self,
                "titles": self.titles,
                "child_text": self.child_text,
                "rerank": await self.rerank_scores(q),
            }
            scores = await intervention(q, cands, ctx) if _is_async(intervention) \
                else intervention(q, cands, ctx)
            order = sorted(range(len(cands)), key=lambda i: scores[i], reverse=True)
            top = [cands[i] for i in order[:top_k]]
            ranking: List[str] = []
            for i in order:
                if cands[i].doc_id not in ranking:
                    ranking.append(cands[i].doc_id)
            title = self.titles.get(top[0].doc_id, "")
            results.append(RunResult(
                question=q.text,
                expected_prefix=self.expected[q.qid],
                top_doc_id=top[0].doc_id,
                top_title=title,
                top_type=doc_type(title),
                top_snippet=" ".join(
                    self.child_text.get(top[0].chunk_id, "").split())[:240],
                contexts=await self.hydrate(top),
                doc_ranking=ranking,
            ))
        return results

    async def close(self) -> None:
        await self.retriever.close()
        await self.reranker.close()


def _is_async(fn) -> bool:
    import inspect

    return inspect.iscoroutinefunction(fn)


def baseline(q: Query, cands: Sequence[Candidate], ctx: dict) -> List[float]:
    """What production does today: rerank on child text alone."""
    return ctx["rerank"]


def summarise(name: str, results: Sequence[RunResult]) -> dict:
    """Headline numbers: how often a decree won, how often the expected
    document won among the questions that have one."""
    keputusan = sum(1 for r in results if r.top_type == "Keputusan")
    scored = [r for r in results if r.found_expected is not None]
    correct = sum(1 for r in scored if r.found_expected)
    return {
        "intervention": name,
        "questions": len(results),
        "top_source_keputusan": keputusan,
        "top_source_peraturan": len(results) - keputusan,
        "expected_doc_available": len(scored),
        "expected_doc_first": correct,
    }


def table(rows: Sequence[dict]) -> str:
    head = ("| intervention | decree on top | expected doc first |\n"
            "|---|---|---|\n")
    body = "".join(
        f"| {r['intervention']} | {r['top_source_keputusan']}/{r['questions']} "
        f"| {r['expected_doc_first']}/{r['expected_doc_available']} |\n"
        for r in rows
    )
    return head + body
