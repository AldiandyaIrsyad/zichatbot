"""Phase 1: Generate adjacent-conflict queries targeting confusable document pairs.

Identifies document pairs in the 300-doc manifest that share topic keywords but
differ in year/number (e.g., two SK Rektor about "Pengelola Jurnal" from
different faculties). Uses Qwen3-14B to generate discriminating questions that
are answerable from exactly one document of the pair.

Usage:
    python -m evals.exp2a_chunking.generate_adjacent_queries \\
        --manifest evals/data/rq1_manifest.csv \\
        --output evals/data/subset_a_adjacent.csv \\
        --max-pairs 20
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import re
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import structlog

from app.chat.config import get_chat_config
from app.chat.infra.llm_connection import LLMConnection
from evals.exp2a_chunking.run import load_manifest, load_document_texts

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Pair detection
# ---------------------------------------------------------------------------

# Strip leading regulation numbers/years to get the "topic core"
_NUM_PREFIX_RE = re.compile(
    r"^[\d\w./-]+\s*[-–—]\s*", re.UNICODE
)
_YEAR_RE = re.compile(r"\b(20\d{2}|19\d{2})\b")


def _topic_key(title: str) -> str:
    """Normalise a manifest title to a comparable topic key.

    Removes leading regulation numbers, years, and lowercases. Two titles
    with the same topic key but different years/numbers are confusable.
    """
    t = _NUM_PREFIX_RE.sub("", title.strip())
    t = _YEAR_RE.sub("", t)
    t = re.sub(r"[^a-z0-9\s]", " ", t.lower())
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _title_similarity(a: str, b: str) -> float:
    """SequenceMatcher ratio between two normalised topic keys."""
    return SequenceMatcher(None, a, b).ratio()


def find_confusable_pairs(
    manifest: List[Dict[str, str]],
    min_similarity: float = 0.65,
    max_pairs: int = 20,
) -> List[Tuple[Dict[str, str], Dict[str, str], float]]:
    """Find document pairs with similar topics but different identifiers.

    Returns (doc_a, doc_b, similarity) sorted by descending similarity,
    capped at ``max_pairs``.
    """
    gold_docs = [r for r in manifest if r["role"] == "gold"]
    keyed: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for doc in gold_docs:
        keyed[_topic_key(doc["title"])].append(doc)

    pairs: List[Tuple[Dict[str, str], Dict[str, str], float]] = []
    seen_ids: set = set()

    # Exact topic-key matches first (different year/number, same topic)
    for key, docs in keyed.items():
        if len(docs) < 2:
            continue
        for i in range(len(docs)):
            for j in range(i + 1, len(docs)):
                pair_id = frozenset({docs[i]["doc_id"], docs[j]["doc_id"]})
                if pair_id in seen_ids:
                    continue
                seen_ids.add(pair_id)
                sim = _title_similarity(docs[i]["title"], docs[j]["title"])
                if sim >= min_similarity:
                    pairs.append((docs[i], docs[j], sim))

    # Fuzzy matches across different topic keys
    all_keys = list(keyed.keys())
    for i in range(len(all_keys)):
        for j in range(i + 1, len(all_keys)):
            sim = _title_similarity(all_keys[i], all_keys[j])
            if sim < min_similarity:
                continue
            for da in keyed[all_keys[i]]:
                for db in keyed[all_keys[j]]:
                    pair_id = frozenset({da["doc_id"], db["doc_id"]})
                    if pair_id in seen_ids:
                        continue
                    seen_ids.add(pair_id)
                    title_sim = _title_similarity(da["title"], db["title"])
                    if title_sim >= min_similarity:
                        pairs.append((da, db, title_sim))

    pairs.sort(key=lambda x: -x[2])
    return pairs[:max_pairs]


# ---------------------------------------------------------------------------
# LLM query generation
# ---------------------------------------------------------------------------

GENERATION_PROMPT = """\
Anda adalah pembuat soal evaluasi untuk sistem temu kembali informasi (retrieval) \
dokumen hukum universitas.

Diberikan dua dokumen yang sangat mirip topiknya tetapi berbeda isinya:

DOKUMEN A (ID: {doc_a_id}):
Judul: {title_a}
Cuplikan isi:
{snippet_a}

DOKUMEN B (ID: {doc_b_id}):
Judul: {title_b}
Cuplikan isi:
{snippet_b}

Tugas: Buat SATU pertanyaan dalam Bahasa Indonesia yang:
1. Hanya bisa dijawab dengan benar dari DOKUMEN A (bukan B).
2. Jika seseorang salah mengambil DOKUMEN B, jawabannya akan salah.
3. Spesifik — menyebutkan detail unik (nomor, nama, tanggal, fakultas) dari DOKUMEN A.
4. Natural — seperti pertanyaan yang akan diajukan mahasiswa atau staf.

Balas HANYA dalam format JSON (tanpa markdown fence):
{{"question": "...", "answer": "...", "discriminating_detail": "..."}}

"discriminating_detail" = fakta spesifik dari DOKUMEN A yang membedakan dari B.
"""


async def generate_query_for_pair(
    llm: LLMConnection,
    model: str,
    doc_a: Dict[str, str],
    doc_b: Dict[str, str],
    text_a: str,
    text_b: str,
) -> Optional[Dict[str, str]]:
    """Generate one discriminating question for a document pair.

    Returns dict with keys: question, answer, discriminating_detail.
    Returns None on any failure (LLM error, parse error, empty result).
    """
    snippet_a = text_a[:2000]
    snippet_b = text_b[:2000]

    prompt = GENERATION_PROMPT.format(
        doc_a_id=doc_a["doc_id"],
        title_a=doc_a["title"],
        snippet_a=snippet_a,
        doc_b_id=doc_b["doc_id"],
        title_b=doc_b["title"],
        snippet_b=snippet_b,
    )

    try:
        raw = await llm.generate(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
            temperature=0.3,
        )
    except Exception as exc:
        logger.warning("adjacent_gen.llm_failed", error=str(exc),
                       doc_a=doc_a["doc_id"][:8], doc_b=doc_b["doc_id"][:8])
        return None

    try:
        # Strip markdown fences if present
        raw = raw.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
        parsed = json.loads(raw)
        q = parsed.get("question", "").strip()
        a = parsed.get("answer", "").strip()
        d = parsed.get("discriminating_detail", "").strip()
        if not q or not a:
            return None
        return {"question": q, "answer": a, "discriminating_detail": d}
    except (json.JSONDecodeError, KeyError, Exception) as exc:
        logger.warning("adjacent_gen.parse_failed", error=str(exc))
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


FIELDNAMES = [
    "question", "category", "ground_truth_answer", "source_doc_id",
    "source_context", "adjacent_distractor_id", "discriminating_detail",
    "pair_similarity",
]


def _load_completed_pairs(output_csv: str) -> set:
    """Return set of (source_doc_id, adjacent_distractor_id) already in the CSV."""
    done: set = set()
    p = Path(output_csv)
    if not p.is_file():
        return done
    try:
        with p.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                key = (row.get("source_doc_id", ""), row.get("adjacent_distractor_id", ""))
                done.add(key)
    except Exception:
        pass
    return done


async def _process_one_pair(
    idx: int,
    doc_a: Dict[str, str],
    doc_b: Dict[str, str],
    sim: float,
    texts: Dict[str, str],
    llm: LLMConnection,
    model: str,
    sem: asyncio.Semaphore,
) -> Optional[Dict[str, str]]:
    """Process a single pair with semaphore-bounded concurrency. Never raises."""
    text_a = texts.get(doc_a["doc_id"], "")
    text_b = texts.get(doc_b["doc_id"], "")
    if not text_a or not text_b:
        print(f"  SKIP pair {idx}: missing text")
        return None

    async with sem:
        try:
            result = await generate_query_for_pair(llm, model, doc_a, doc_b, text_a, text_b)
        except Exception as exc:
            logger.warning("adjacent_gen.pair_failed", idx=idx, error=str(exc))
            return None

    if result is None:
        print(f"  FAIL pair {idx}: LLM/parse error")
        return None

    row = {
        "question": result["question"],
        "category": "adjacent-conflict",
        "ground_truth_answer": result["answer"],
        "source_doc_id": doc_a["doc_id"],
        "source_context": text_a[:3000],
        "adjacent_distractor_id": doc_b["doc_id"],
        "discriminating_detail": result["discriminating_detail"],
        "pair_similarity": f"{sim:.3f}",
    }
    print(f"  OK pair {idx}: {result['question'][:80]}...")
    return row


async def async_main(args: Any) -> None:
    manifest = load_manifest(args.manifest)
    pairs = find_confusable_pairs(
        manifest, min_similarity=args.min_similarity, max_pairs=args.max_pairs
    )
    print(f"Found {len(pairs)} confusable pairs")
    for a, b, sim in pairs:
        print(f"  [{sim:.2f}] {a['title'][:60]}  ↔  {b['title'][:60]}")

    if not pairs:
        print("No pairs found — exiting.")
        return

    # Resume: skip pairs already generated
    completed = _load_completed_pairs(args.output) if args.resume else set()
    if completed:
        before = len(pairs)
        pairs = [
            (a, b, s) for a, b, s in pairs
            if (a["doc_id"], b["doc_id"]) not in completed
        ]
        print(f"Resuming: {before - len(pairs)} pairs already done, {len(pairs)} remaining")

    if not pairs:
        print("Nothing to generate.")
        return

    # Load document texts
    all_ids = list({doc["doc_id"] for doc_a, doc_b, _ in pairs for doc in (doc_a, doc_b)})
    texts, _ = await load_document_texts(all_ids)

    # LLM setup
    cfg = get_chat_config()
    llm = LLMConnection(base_url=cfg.llm_base_url, api_key=cfg.llm_api_key)
    model = args.model or cfg.llm_model
    sem = asyncio.Semaphore(args.concurrency)

    # Open CSV in append mode for resume, write mode otherwise
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_header = not (args.resume and completed)
    csv_f = out.open("a" if (args.resume and completed) else "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_f, fieldnames=FIELDNAMES)
    if write_header:
        writer.writeheader()

    rows: List[Dict[str, str]] = []
    try:
        # Batch: process all pairs concurrently (bounded by semaphore)
        tasks = [
            _process_one_pair(idx, doc_a, doc_b, sim, texts, llm, model, sem)
            for idx, (doc_a, doc_b, sim) in enumerate(pairs)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for idx, r in enumerate(results):
            if isinstance(r, Exception):
                logger.warning("adjacent_gen.task_exception", idx=idx, error=str(r))
                continue
            if r is not None:
                rows.append(r)
                writer.writerow(r)
                csv_f.flush()  # incremental flush — crash loses at most 1 row
    finally:
        csv_f.close()
        await llm.close()

    print(f"\nWrote {len(rows)} adjacent-conflict queries to {out}")
    print(f"Spot-check recommended: verify gold ≠ distractor for each row.")


def main() -> None:
    p = argparse.ArgumentParser(description="Generate adjacent-conflict queries")
    p.add_argument("--manifest", default="evals/data/rq1_manifest.csv")
    p.add_argument("--output", default="evals/data/subset_a_adjacent.csv")
    p.add_argument("--max-pairs", type=int, default=20)
    p.add_argument("--min-similarity", type=float, default=0.65)
    p.add_argument("--model", default=None, help="Override LLM model name")
    p.add_argument("--concurrency", type=int, default=4,
                   help="Max concurrent LLM calls (default: 4)")
    p.add_argument("--resume", action="store_true",
                   help="Skip pairs already in the output CSV")
    args = p.parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
