"""Shared harness for the post-prasidang probes.

The app on :8000 is not assumed to be running, so these probes talk to Qdrant,
Postgres and Infinity directly — the ``exp2a_chunking`` pattern rather than the
HTTP pattern used by ``exp2_retrieval``.

Anything that costs GPU time or API credit is cached to ``cache/`` and appended
as it is produced, so a crash or a rerun is free.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from sqlalchemy import select

from app.kb.config import get_bge_m3_settings, get_infinity_settings, get_qdrant_settings
from app.kb.domain.models import ChildChunk, PDFDocument
from app.shared.db import async_session_maker

def quiet_logs() -> None:
    """Drop the app's INFO chatter — one line per rerank call drowns the report."""
    import logging

    logging.basicConfig(level=logging.WARNING, force=True)
    for name in ("app", "httpx", "httpcore", "FlagEmbedding", "sentence_transformers"):
        logging.getLogger(name).setLevel(logging.WARNING)
    try:
        import structlog

        structlog.configure(
            wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING)
        )
    except ImportError:
        pass


HERE = Path(__file__).resolve().parent          # evals/probes/
DATA_DIR = HERE.parent / "data"                 # evals/data/
CACHE_DIR = HERE / "cache"
RESULTS_DIR = HERE / "results"
STUDENT_CSV = DATA_DIR / "LLM_Generated_studentQ.csv"
SUBSET_A = DATA_DIR / "subset_a.csv"

# Mirrors app/kb/application/search_service.py so variant A reproduces production.
INITIAL_SEARCH_TOP_K = 50
RERANK_TOP_K = 8


# =============================================================================
# DOCUMENT METADATA
# =============================================================================


@dataclass(frozen=True)
class DocMeta:
    """What can be recovered about a document without a schema change.

    ``pdf_documents`` has no ``tahun``, ``nomor`` or ``status`` column, so every
    field here is parsed out of the title prefix or the ``description``
    provenance string.
    """

    doc_id: str
    title: str
    nomor: Optional[str] = None
    tahun: Optional[int] = None
    tahun_source: str = "none"          # "title" | "description" | "none"
    is_perubahan: bool = False
    perubahan_ordinal: Optional[str] = None   # Pertama / Kedua / Ketiga / ...
    is_pencabutan: bool = False
    target_ref: Optional[str] = None    # the document this one amends, as written


# "26 Tahun 2025 - Pengelolaan ..."  /  "0118-UN40-HK-2015 - Tata Naskah ..."
_TITLE_TAHUN_RE = re.compile(r"^\s*(\d+)\s+Tahun\s+((?:19|20)\d{2})\b")
_TITLE_CODE_RE = re.compile(r"^\s*([\w.]+)[-/]UN40[\w.\-/]*[-/]((?:19|20)\d{2})\b")
# "Sourced from JDIH UPI. Code: 1282/UN40/KM.02.02/2026"
_DESC_CODE_RE = re.compile(r"Code:\s*([^\s,;]+)")
_DESC_YEAR_RE = re.compile(r"/((?:19|20)\d{2})\s*$")

_PERUBAHAN_RE = re.compile(
    r"\bPerubahan(?:\s+(Pertama|Kedua|Ketiga|Keempat|Kelima|Keenam|Ketujuh))?\s+Atas\b",
    re.IGNORECASE,
)
_PENCABUTAN_RE = re.compile(r"\bPencabutan\b", re.IGNORECASE)
_TARGET_RE = re.compile(
    r"Atas\s+(.{0,160}?)(?:\s+[Tt]entang\b|$)",
    re.DOTALL,
)


def parse_doc_meta(doc_id: str, title: str, description: str = "") -> DocMeta:
    """Recover nomor / tahun / amendment role from a document's title.

    The year is taken from the **prefix before the first " - "**, never from a
    global regex: subject text carries its own years ("...Laporan Akhir Tahun
    UPI Tahun 2025" sits on a 2026 document), so a global match picks the wrong
    one. The ``description`` provenance code is the fallback and is populated
    for the whole corpus.
    """
    title = (title or "").strip()
    prefix = title.split(" - ", 1)[0] if " - " in title else title

    nomor: Optional[str] = None
    tahun: Optional[int] = None
    tahun_source = "none"

    m = _TITLE_TAHUN_RE.match(prefix)
    if m:
        nomor, tahun, tahun_source = m.group(1), int(m.group(2)), "title"
    else:
        m = _TITLE_CODE_RE.match(prefix)
        if m:
            nomor, tahun, tahun_source = m.group(1), int(m.group(2)), "title"

    if tahun is None and description:
        code = _DESC_CODE_RE.search(description)
        if code:
            nomor = nomor or code.group(1)
            year = _DESC_YEAR_RE.search(code.group(1))
            if year:
                tahun, tahun_source = int(year.group(1)), "description"

    perubahan = _PERUBAHAN_RE.search(title)
    target = None
    if perubahan:
        t = _TARGET_RE.search(title[perubahan.start():])
        if t:
            target = t.group(1).strip(" ,.")

    return DocMeta(
        doc_id=doc_id,
        title=title,
        nomor=nomor,
        tahun=tahun,
        tahun_source=tahun_source,
        is_perubahan=bool(perubahan),
        perubahan_ordinal=perubahan.group(1).capitalize() if perubahan and perubahan.group(1) else None,
        is_pencabutan=bool(_PENCABUTAN_RE.search(title)),
        target_ref=target,
    )


async def load_documents() -> Dict[str, DocMeta]:
    """Every document in the corpus, keyed by doc_id, with parsed metadata."""
    async with async_session_maker() as session:
        rows = (
            await session.execute(
                select(PDFDocument.id, PDFDocument.title, PDFDocument.description)
            )
        ).all()
    return {
        r[0]: parse_doc_meta(r[0], r[1] or "", r[2] or "")
        for r in rows
    }


async def load_child_texts(chunk_ids: Sequence[str]) -> Dict[str, str]:
    """Child chunk text by id, the same source the production reranker sees."""
    if not chunk_ids:
        return {}
    out: Dict[str, str] = {}
    async with async_session_maker() as session:
        for i in range(0, len(chunk_ids), 500):
            batch = list(chunk_ids[i : i + 500])
            rows = (
                await session.execute(
                    select(ChildChunk.id, ChildChunk.text).where(ChildChunk.id.in_(batch))
                )
            ).all()
            out.update({r[0]: r[1] for r in rows})
    return out


# =============================================================================
# QUERY SETS
# =============================================================================


@dataclass(frozen=True)
class Query:
    qid: str
    text: str
    gold_doc_ids: Tuple[str, ...] = ()
    register: str = ""
    category: str = ""
    gold_context: str = ""


def load_student_questions() -> List[Query]:
    """The staged student questions, one Query per (row, register)."""
    import csv

    out: List[Query] = []
    with open(STUDENT_CSV, newline="", encoding="utf-8-sig") as f:
        for i, row in enumerate(csv.DictReader(f)):
            for col in ("question_1", "question_2", "question_3"):
                text = (row.get(col) or "").strip()
                if text:
                    out.append(Query(qid=f"s{i:03d}_{col[-1]}", text=text, register=col))
    return out


def load_subset_a_queries(limit: Optional[int] = None) -> List[Query]:
    """Subset A as Query objects. Out-of-domain rows (gold "NONE") are dropped —
    they carry no retrieval target, so they cannot score a ranking variant."""
    from evals._shared.dataset import load_subset_a

    rows = load_subset_a(str(SUBSET_A))
    out = [
        Query(
            qid=f"a{i:04d}",
            text=r.question,
            gold_doc_ids=tuple(r.gold_doc_ids),
            category=r.category,
            gold_context=r.source_context or "",
        )
        for i, r in enumerate(rows)
        if r.gold_doc_ids
    ]
    return out[:limit] if limit else out


# =============================================================================
# RETRIEVAL (cached)
# =============================================================================


@dataclass
class Candidate:
    chunk_id: str
    parent_chunk_id: str
    doc_id: str
    score: float


@dataclass
class QueryCandidates:
    qid: str
    text: str
    candidates: List[Candidate] = field(default_factory=list)


class Retriever:
    """BGE-M3 + Qdrant hybrid search, matching the production call exactly.

    Results are cached per query so probe 4 (and any rerun of probe 1) costs
    nothing. The cache key is the query text, not the qid, so the same question
    appearing in two sets is embedded once.
    """

    def __init__(self, cache_name: str = "candidates.jsonl") -> None:
        self._cache_path = CACHE_DIR / cache_name
        self._cache: Dict[str, List[Candidate]] = {}
        self._embedder = None
        self._store = None
        self._load_cache()

    def _load_cache(self) -> None:
        if not self._cache_path.exists():
            return
        with open(self._cache_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                self._cache[rec["text"]] = [Candidate(**c) for c in rec["candidates"]]
        print(f"  cache: {len(self._cache)} queries already retrieved")

    def _append_cache(self, text: str, cands: List[Candidate]) -> None:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(self._cache_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(
                {"text": text, "candidates": [asdict(c) for c in cands]},
                ensure_ascii=False,
            ) + "\n")

    def _ensure_clients(self) -> None:
        if self._embedder is not None:
            return
        from app.kb.infra.bge_m3_embeddings import BGEM3Embeddings
        from app.kb.infra.qdrant_store import QdrantStore

        ecfg = get_bge_m3_settings()
        qcfg = get_qdrant_settings()
        self._embedder = BGEM3Embeddings(
            model_name=ecfg.model, device=ecfg.device,
            use_fp16=ecfg.use_fp16, batch_size=ecfg.batch_size,
        )
        self._store = QdrantStore(qcfg.host, qcfg.port, qcfg.collection_name)

    async def retrieve(self, queries: Sequence[Query],
                       top_k: int = INITIAL_SEARCH_TOP_K) -> Dict[str, List[Candidate]]:
        """Candidates per query text, using the cache where possible."""
        missing = [q for q in queries if q.text not in self._cache]
        if missing:
            self._ensure_clients()
            print(f"  retrieving {len(missing)} uncached queries ...")
            for i, q in enumerate(missing, 1):
                emb = (await self._embedder.embed_texts([q.text]))[0]
                results = await self._store.hybrid_search(
                    dense_vector=emb.dense,
                    sparse_indices=emb.sparse_indices,
                    sparse_values=emb.sparse_values,
                    top_k=top_k,
                )
                cands = [
                    Candidate(r.chunk_id, r.parent_chunk_id, r.doc_id, float(r.score))
                    for r in results
                ]
                self._cache[q.text] = cands
                self._append_cache(q.text, cands)
                if i % 25 == 0:
                    print(f"    {i}/{len(missing)}")
        return {q.qid: self._cache[q.text] for q in queries}

    async def embed(self, texts: Sequence[str]) -> List[List[float]]:
        """Dense vectors only — used for title and query similarity."""
        self._ensure_clients()
        out: List[List[float]] = []
        batch = 64
        for i in range(0, len(texts), batch):
            embs = await self._embedder.embed_texts(list(texts[i : i + batch]))
            out.extend(e.dense for e in embs)
        return out

    async def close(self) -> None:
        if self._embedder is not None:
            await self._embedder.close()


def cached_dense(name: str):
    """Load a cached (ids, matrix) dense-vector pair, or None."""
    import numpy as np

    path = CACHE_DIR / f"{name}.npz"
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=True)
    return list(data["ids"]), data["vecs"]


def save_dense(name: str, ids: Sequence[str], vecs) -> None:
    import numpy as np

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE_DIR / f"{name}.npz", ids=np.array(list(ids), dtype=object),
             vecs=np.asarray(vecs, dtype="float32"))


# =============================================================================
# RERANKING
# =============================================================================


def get_reranker():
    from app.kb.infra.infinity_reranker import InfinityReranker

    icfg = get_infinity_settings()
    return InfinityReranker(base_url=icfg.base_url, model=icfg.reranker_model)


# =============================================================================
# SCORING HELPERS
# =============================================================================


def doc_ranking(candidates: Iterable[Candidate]) -> List[str]:
    """Collapse a chunk ranking to a document ranking, first occurrence wins."""
    seen: List[str] = []
    for c in candidates:
        if c.doc_id not in seen:
            seen.append(c.doc_id)
    return seen


def hit_at_k(ranked_docs: Sequence[str], gold: Sequence[str], k: int) -> int:
    return int(any(d in gold for d in ranked_docs[:k]))


def reciprocal_rank(ranked_docs: Sequence[str], gold: Sequence[str]) -> float:
    for i, d in enumerate(ranked_docs, 1):
        if d in gold:
            return 1.0 / i
    return 0.0


def cosine(a, b) -> float:
    import numpy as np

    a = np.asarray(a, dtype="float32")
    b = np.asarray(b, dtype="float32")
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b / denom) if denom else 0.0


def minmax(values: Sequence[float]) -> List[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        return [0.5] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


# =============================================================================
# OUTPUT
# =============================================================================


def write_results(name: str, rows: List[Dict[str, Any]], summary: Dict[str, Any]) -> None:
    """Per-row CSV plus aggregate JSON, the evals/data/results convention."""
    from evals._shared.csv_export import write_results_csv

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if rows:
        write_results_csv(str(RESULTS_DIR / f"{name}.csv"), rows)
    path = RESULTS_DIR / f"{name}.summary.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  Summary written to {path}")
