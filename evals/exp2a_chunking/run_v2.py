"""Exp2a-v2: 2×2 chunking × reranker evaluation with LLM-judged answer quality.

Supplements the original exp2a (kept immutable) with:
  - Reranker-on condition (2×2: {fixed, hierarchical} × {reranker off/on})
  - Context sufficiency scoring (answer term recall in budgeted context)
  - Adjacent-conflict discrimination rate
  - LLM answer generation + judge (Qwen3-14B via Ollama)
  - Per-category breakdown including adjacent-conflict queries

Fail-safe: every external call (Qdrant, reranker, LLM) is wrapped in
try/except.  A failed query is logged and skipped; partial results are
flushed to disk after every embedding batch so a crash never loses more
than one batch of work.

Batching: LLM generation + judging for all 4 conditions of a single
query run concurrently via asyncio.gather (bounded by --llm-concurrency).
Reranker calls for the two conditions sharing the same retrieval result
also run concurrently.

Usage:
    # Full run (retrieval + LLM generation + judging):
    python -m evals.exp2a_chunking.run_v2 --evaluate

    # Retrieval-only (skip LLM calls):
    python -m evals.exp2a_chunking.run_v2 --evaluate --skip-llm

    # With adjacent-conflict queries:
    python -m evals.exp2a_chunking.run_v2 --evaluate \\
        --adjacent-dataset evals/data/subset_a_adjacent.csv

    # Resume after crash (skips questions already in the output CSV):
    python -m evals.exp2a_chunking.run_v2 --evaluate --resume
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import structlog
from qdrant_client import AsyncQdrantClient
from transformers import AutoTokenizer

from app.chat.config import get_chat_config
from app.chat.infra.llm_connection import LLMConnection
from app.kb.config import get_bge_m3_settings, get_infinity_settings, get_qdrant_settings
from app.kb.infra.bge_m3_embeddings import BGEM3Embeddings
from app.kb.infra.infinity_reranker import InfinityReranker
from evals._shared.dataset import SubsetARow, load_subset_a
from evals.exp2a_chunking.run import (
    FIXED_COLLECTION,
    HIER_COLLECTION,
    load_document_texts,
    load_manifest,
    mcnemar_exact,
    paired_bootstrap,
    query_points,
)

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONDITIONS = ("fixed", "hierarchical")
RERANKER_MODES = ("off", "on")
RERANK_TOP_K = 8
TOKEN_BUDGET = 8000
LLM_MODEL_DEFAULT = "qwen3:14b"
LLM_MAX_TOKENS = 1024
JUDGE_MAX_TOKENS = 256
DEFAULT_LLM_CONCURRENCY = 4

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _terms(text: str) -> set:
    return {x.casefold() for x in re.findall(r"[A-Za-zÀ-ÿ0-9]+", text) if len(x) > 2}


def _term_recall(needle: str, haystack: str) -> float:
    a, b = _terms(needle), _terms(haystack)
    return len(a & b) / len(a) if a else 0.0


def _doc_aggregated_ranking(points: Sequence[Any]) -> List[str]:
    scores: Dict[str, float] = {}
    for p in points:
        doc = str((p.payload or {}).get("doc_id", ""))
        if doc:
            scores[doc] = max(scores.get(doc, float("-inf")), float(p.score))
    return [d for d, _ in sorted(scores.items(), key=lambda x: (-x[1], x[0]))]


def _rr_at_k(ranking: Sequence[str], gold: Sequence[str], k: int = 5) -> float:
    gold_set = set(gold)
    for i, d in enumerate(ranking[:k], 1):
        if d in gold_set:
            return 1.0 / i
    return 0.0


def _rank_of(ranking: Sequence[str], doc_id: str) -> Optional[int]:
    for i, d in enumerate(ranking):
        if d == doc_id:
            return i
    return None


# ---------------------------------------------------------------------------
# Reranker (fail-safe)
# ---------------------------------------------------------------------------


async def rerank_points(
    reranker: InfinityReranker,
    query: str,
    points: Sequence[Any],
    top_k: int = RERANK_TOP_K,
) -> List[Any]:
    """Rerank points via cross-encoder. Falls back to original order on error."""
    if not points:
        return []
    try:
        texts = [str((p.payload or {}).get("text", "")) for p in points]
        results = await reranker.rerank(query, texts, top_k=top_k)
        reranked = sorted(results, key=lambda r: -r.score)
        return [points[r.index] for r in reranked if r.index < len(points)]
    except Exception as exc:
        logger.warning("rerank.failed_fallback_to_original", error=str(exc), n_points=len(points))
        return list(points[:top_k])


# ---------------------------------------------------------------------------
# LLM prompts + fail-safe calls
# ---------------------------------------------------------------------------

ANSWER_PROMPT = """\
Anda adalah asisten penjawab pertanyaan berbasis dokumen hukum universitas.

Berdasarkan KONTEKS berikut, jawab PERTANYAAN dengan singkat dan akurat.
Jika konteks tidak cukup untuk menjawab, tulis: INSUFFICIENT

KONTEKS:
{context}

PERTANYAAN: {question}

JAWABAN:"""

JUDGE_PROMPT = """\
Anda adalah penilai jawaban. Bandingkan JAWABAN MODEL dengan KUNCI JAWABAN.

KUNCI JAWABAN: {gold_answer}

JAWABAN MODEL: {model_answer}

Berikan verdict:
- CORRECT: jawaban model secara substansi benar dan sesuai kunci
- PARTIAL: jawaban model sebagian benar tetapi tidak lengkap
- INCORRECT: jawaban model salah atau tidak relevan
- REFUSED: jawaban model menyatakan tidak bisa menjawab / INSUFFICIENT

Balas HANYA satu kata: CORRECT, PARTIAL, INCORRECT, atau REFUSED"""


async def _safe_generate(
    llm: LLMConnection, model: str, prompt: str, max_tokens: int
) -> str:
    """Call LLM.generate with try/except. Returns ERROR: string on failure."""
    try:
        return await llm.generate(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.0,
        )
    except Exception as exc:
        logger.warning("llm.generate_failed", error=str(exc), model=model)
        return f"ERROR: {exc}"


def _parse_verdict(raw: str) -> str:
    verdict = raw.strip().upper()
    if verdict in ("CORRECT", "PARTIAL", "INCORRECT", "REFUSED"):
        return verdict
    for v in ("CORRECT", "PARTIAL", "INCORRECT", "REFUSED"):
        if v in verdict:
            return v
    if verdict.startswith("ERROR:"):
        return verdict
    return "UNKNOWN"


# ---------------------------------------------------------------------------
# Adjacent-conflict dataset loader
# ---------------------------------------------------------------------------


def load_adjacent_dataset(path: str) -> Tuple[List[SubsetARow], Dict[str, str]]:
    """Returns (rows, distractor_map) where distractor_map maps question → distractor_id."""
    rows: List[SubsetARow] = []
    distractor_map: Dict[str, str] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            q = row["question"].strip()
            rows.append(
                SubsetARow(
                    question=q,
                    category="adjacent-conflict",
                    ground_truth_answer=row.get("ground_truth_answer", "").strip(),
                    source_doc_id=row.get("source_doc_id", "").strip(),
                    source_context=row.get("source_context", "").strip(),
                )
            )
            dist_id = row.get("adjacent_distractor_id", "").strip()
            if dist_id:
                distractor_map[q] = dist_id
    return rows, distractor_map


# ---------------------------------------------------------------------------
# Resume support
# ---------------------------------------------------------------------------


def _load_completed_questions(output_csv: str) -> Set[str]:
    """Read already-written questions from a partial output CSV."""
    done: Set[str] = set()
    p = Path(output_csv)
    if not p.is_file():
        return done
    try:
        with p.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                q = row.get("question", "").strip()
                if q:
                    done.add(q)
    except Exception:
        pass
    return done


# ---------------------------------------------------------------------------
# Context packing
# ---------------------------------------------------------------------------


def _pack_context(
    points: Sequence[Any],
    condition: str,
    tokenizer: Any,
    parent_text: Dict[str, str],
    token_budget: int,
) -> Tuple[List[str], int, str]:
    used = 0
    ranking: List[str] = []
    seen_docs: set = set()
    seen_contexts: set = set()
    texts: List[str] = []
    for point in points:
        payload = point.payload or {}
        doc_id = str(payload.get("doc_id", ""))
        context_id = str(payload.get("parent_chunk_id", point.id))
        if context_id in seen_contexts:
            continue
        text = (
            parent_text.get(context_id, str(payload.get("text", "")))
            if condition == "hierarchical"
            else str(payload.get("text", ""))
        )
        n_tokens = len(tokenizer.encode(text, add_special_tokens=False))
        if used and used + n_tokens > token_budget:
            continue
        used += min(n_tokens, token_budget)
        seen_contexts.add(context_id)
        texts.append(text)
        if doc_id and doc_id not in seen_docs:
            ranking.append(doc_id)
            seen_docs.add(doc_id)
        if used >= token_budget:
            break
    return ranking, used, "\n\n".join(texts)


# ---------------------------------------------------------------------------
# Per-query processing (all 4 conditions, LLM calls batched via gather)
# ---------------------------------------------------------------------------


async def _process_query(
    row: SubsetARow,
    emb: Any,
    client: AsyncQdrantClient,
    collection_map: Dict[str, str],
    reranker: InfinityReranker,
    tokenizer: Any,
    parent_text: Dict[str, str],
    distractor_map: Dict[str, str],
    llm: Optional[LLMConnection],
    model_name: str,
    llm_sem: asyncio.Semaphore,
) -> Dict[str, Any]:
    """Process one query across all 4 conditions. Never raises."""
    rec: Dict[str, Any] = {
        "question": row.question,
        "category": row.category,
        "gold_doc_ids": "|".join(row.gold_doc_ids),
    }

    for cond in CONDITIONS:
        collection = collection_map[cond]

        # --- Retrieval (fail-safe) ---
        try:
            points = await query_points(client, collection, emb, limit=50)
        except Exception as exc:
            logger.error("retrieval.failed", condition=cond, error=str(exc))
            points = []

        # --- Reranker off + on run concurrently ---
        async def _eval_rmode(rmode: str, pts: Sequence[Any] = points) -> Dict[str, Any]:
            prefix = f"{cond}_{rmode}"
            out: Dict[str, Any] = {}

            if rmode == "on":
                ranked = await rerank_points(reranker, row.question, pts)
            else:
                ranked = list(pts)

            agg_ranking = _doc_aggregated_ranking(ranked)
            budget_ranking, used_tokens, context_text = _pack_context(
                ranked, cond, tokenizer, parent_text, TOKEN_BUDGET
            )

            h5 = int(any(g in budget_ranking[:5] for g in row.gold_doc_ids))
            rr5 = _rr_at_k(budget_ranking, row.gold_doc_ids)
            agg_h5 = int(any(g in agg_ranking[:5] for g in row.gold_doc_ids))
            agg_rr5 = _rr_at_k(agg_ranking, row.gold_doc_ids)
            ctx_suff = _term_recall(row.ground_truth_answer, context_text)

            out[f"{prefix}_doc_ids"] = "|".join(budget_ranking)
            out[f"{prefix}_tokens"] = used_tokens
            out[f"{prefix}_hit5"] = h5
            out[f"{prefix}_rr5"] = rr5
            out[f"{prefix}_agg_hit5"] = agg_h5
            out[f"{prefix}_agg_rr5"] = agg_rr5
            out[f"{prefix}_ctx_sufficiency"] = round(ctx_suff, 4)

            # Adjacent discrimination
            if row.category == "adjacent-conflict":
                distractor = distractor_map.get(row.question)
                if distractor and row.gold_doc_ids:
                    gold_rank = _rank_of(budget_ranking, row.gold_doc_ids[0])
                    dist_rank = _rank_of(budget_ranking, distractor)
                    out[f"{prefix}_adjacent_correct"] = int(
                        gold_rank is not None
                        and (dist_rank is None or gold_rank < dist_rank)
                    )

            # LLM generation + judging (bounded by semaphore)
            if llm is not None:
                async with llm_sem:
                    answer = await _safe_generate(
                        llm, model_name,
                        ANSWER_PROMPT.format(context=context_text[:12000], question=row.question),
                        LLM_MAX_TOKENS,
                    )
                async with llm_sem:
                    raw_verdict = await _safe_generate(
                        llm, model_name,
                        JUDGE_PROMPT.format(gold_answer=row.ground_truth_answer, model_answer=answer),
                        JUDGE_MAX_TOKENS,
                    )
                out[f"{prefix}_answer"] = answer[:500]
                out[f"{prefix}_verdict"] = _parse_verdict(raw_verdict)

            return out

        # Run reranker-off and reranker-on concurrently for this condition
        try:
            results = await asyncio.gather(
                _eval_rmode("off"),
                _eval_rmode("on"),
                return_exceptions=True,
            )
            for r in results:
                if isinstance(r, Exception):
                    logger.error("condition.failed", condition=cond, error=str(r))
                elif isinstance(r, dict):
                    rec.update(r)
        except Exception as exc:
            logger.error("query.gather_failed", condition=cond, error=str(exc))

    return rec


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------


async def evaluate(
    manifest_path: str,
    dataset_path: str,
    adjacent_path: Optional[str],
    output_csv: str,
    hier_collection: str,
    fixed_collection: str,
    skip_llm: bool,
    llm_model: Optional[str],
    llm_concurrency: int,
    resume: bool,
) -> Dict[str, Any]:
    # --- Load data ---
    manifest = load_manifest(manifest_path)
    manifest_ids = {r["doc_id"] for r in manifest}
    dataset = [r for r in load_subset_a(dataset_path) if r.category != "out-of-domain"]
    if len(dataset) != 115:
        raise AssertionError(f"Expected 115 queries, got {len(dataset)}")

    distractor_map: Dict[str, str] = {}
    if adjacent_path and Path(adjacent_path).is_file():
        adjacent, distractor_map = load_adjacent_dataset(adjacent_path)
        dataset = dataset + adjacent
        print(f"Loaded {len(adjacent)} adjacent-conflict queries (total: {len(dataset)})")

    # --- Resume: skip already-completed questions ---
    completed: Set[str] = set()
    if resume:
        completed = _load_completed_questions(output_csv)
        if completed:
            print(f"Resuming: {len(completed)} questions already done, skipping them")
            dataset = [r for r in dataset if r.question not in completed]
            print(f"  remaining: {len(dataset)}")

    if not dataset:
        print("Nothing to evaluate.")
        return {}

    # --- Infrastructure ---
    qcfg = get_qdrant_settings()
    client = AsyncQdrantClient(host=qcfg.host, port=qcfg.port)
    ecfg = get_bge_m3_settings()
    tokenizer = AutoTokenizer.from_pretrained(ecfg.model, local_files_only=True)
    embedder = BGEM3Embeddings(
        model_name=ecfg.model, device=ecfg.device,
        use_fp16=ecfg.use_fp16, batch_size=ecfg.batch_size,
    )

    icfg = get_infinity_settings()
    reranker = InfinityReranker(base_url=icfg.base_url, model=icfg.reranker_model)

    _, parent_text = await load_document_texts(list(manifest_ids))

    llm: Optional[LLMConnection] = None
    model_name = llm_model or LLM_MODEL_DEFAULT
    if not skip_llm:
        cfg = get_chat_config()
        llm = LLMConnection(base_url=cfg.llm_base_url, api_key=cfg.llm_api_key)
        model_name = llm_model or cfg.llm_model or LLM_MODEL_DEFAULT

    llm_sem = asyncio.Semaphore(llm_concurrency)
    collection_map = {"fixed": fixed_collection, "hierarchical": hier_collection}

    # --- Accumulators ---
    rows: List[Dict[str, Any]] = []
    acc: Dict[Tuple[str, str], Dict[str, list]] = {}
    for cond in CONDITIONS:
        for rmode in RERANKER_MODES:
            acc[(cond, rmode)] = {
                "hit5": [], "rr5": [], "agg_hit5": [], "agg_rr5": [],
                "context_sufficiency": [], "adjacent_correct": [], "verdicts": [],
            }

    # --- CSV writer (append mode for resume) ---
    out = Path(output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_header = not (resume and out.is_file() and completed)
    csv_f = out.open("a" if (resume and completed) else "w", newline="", encoding="utf-8")
    csv_writer: Optional[csv.DictWriter] = None

    try:
        for start in range(0, len(dataset), ecfg.batch_size):
            batch = dataset[start : start + ecfg.batch_size]

            # Embed batch (fail-safe: skip batch on failure)
            try:
                embeddings = await embedder.embed_texts([r.question for r in batch])
            except Exception as exc:
                logger.error("embed.batch_failed", error=str(exc), batch_start=start)
                continue

            for row, emb in zip(batch, embeddings):
                try:
                    rec = await _process_query(
                        row, emb, client, collection_map, reranker,
                        tokenizer, parent_text, distractor_map,
                        llm, model_name, llm_sem,
                    )
                except Exception as exc:
                    logger.error("query.failed", question=row.question[:60], error=str(exc))
                    rec = {"question": row.question, "category": row.category,
                           "gold_doc_ids": "|".join(row.gold_doc_ids), "error": str(exc)}

                rows.append(rec)

                # Accumulate metrics
                for cond in CONDITIONS:
                    for rmode in RERANKER_MODES:
                        key = (cond, rmode)
                        prefix = f"{cond}_{rmode}"
                        a = acc[key]
                        a["hit5"].append(rec.get(f"{prefix}_hit5", 0))
                        a["rr5"].append(rec.get(f"{prefix}_rr5", 0.0))
                        a["agg_hit5"].append(rec.get(f"{prefix}_agg_hit5", 0))
                        a["agg_rr5"].append(rec.get(f"{prefix}_agg_rr5", 0.0))
                        a["context_sufficiency"].append(rec.get(f"{prefix}_ctx_sufficiency", 0.0))
                        if f"{prefix}_adjacent_correct" in rec:
                            a["adjacent_correct"].append(rec[f"{prefix}_adjacent_correct"])
                        if f"{prefix}_verdict" in rec:
                            a["verdicts"].append(rec[f"{prefix}_verdict"])

                # Flush to CSV incrementally
                if csv_writer is None:
                    # Build complete fieldnames upfront so adjacent-conflict
                    # extra fields don't break the writer later.
                    base_fields = ["question", "category", "gold_doc_ids"]
                    cond_fields: List[str] = []
                    for c in CONDITIONS:
                        for rm in RERANKER_MODES:
                            pfx = f"{c}_{rm}"
                            cond_fields += [
                                f"{pfx}_doc_ids", f"{pfx}_tokens",
                                f"{pfx}_hit5", f"{pfx}_rr5",
                                f"{pfx}_agg_hit5", f"{pfx}_agg_rr5",
                                f"{pfx}_ctx_sufficiency",
                                f"{pfx}_adjacent_correct",
                                f"{pfx}_answer", f"{pfx}_verdict",
                            ]
                    all_fields = base_fields + cond_fields + ["error"]
                    csv_writer = csv.DictWriter(csv_f, fieldnames=all_fields, extrasaction="ignore")
                    if write_header:
                        csv_writer.writeheader()
                csv_writer.writerow(rec)
                csv_f.flush()

            done = min(start + len(batch), len(dataset))
            print(f"  evaluated {done}/{len(dataset)}", flush=True)

    finally:
        csv_f.close()
        await embedder.close()
        await client.close()
        if llm is not None:
            await llm.close()

    # --- Summary ---
    summary = _build_summary(acc, rows)
    out.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


# ---------------------------------------------------------------------------
# Summary builder
# ---------------------------------------------------------------------------


def _build_summary(
    acc: Dict[Tuple[str, str], Dict[str, list]],
    rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"query_n": len(rows), "conditions": {}}

    for cond in CONDITIONS:
        for rmode in RERANKER_MODES:
            a = acc[(cond, rmode)]
            n = len(a["hit5"])
            if n == 0:
                continue
            entry: Dict[str, Any] = {
                "hit_rate_at_5": sum(a["hit5"]) / n,
                "mrr_at_5": sum(a["rr5"]) / n,
                "agg_hit_rate_at_5": sum(a["agg_hit5"]) / n,
                "agg_mrr_at_5": sum(a["agg_rr5"]) / n,
                "mean_context_sufficiency": statistics.mean(a["context_sufficiency"]),
            }
            if a["adjacent_correct"]:
                entry["adjacent_discrimination_rate"] = (
                    sum(a["adjacent_correct"]) / len(a["adjacent_correct"])
                )
                entry["adjacent_n"] = len(a["adjacent_correct"])
            if a["verdicts"]:
                vc = Counter(a["verdicts"])
                total_v = len(a["verdicts"])
                entry["answer_correct_rate"] = vc.get("CORRECT", 0) / total_v
                entry["answer_partial_rate"] = vc.get("PARTIAL", 0) / total_v
                entry["answer_incorrect_rate"] = vc.get("INCORRECT", 0) / total_v
                entry["answer_refused_rate"] = vc.get("REFUSED", 0) / total_v
                entry["verdict_distribution"] = dict(vc)
            summary["conditions"][f"{cond}_reranker_{rmode}"] = entry

    # Pairwise comparisons
    for rmode in RERANKER_MODES:
        fk, hk = ("fixed", rmode), ("hierarchical", rmode)
        if not acc[fk]["hit5"] or not acc[hk]["hit5"]:
            continue
        summary[f"fixed_vs_hierarchical_reranker_{rmode}"] = {
            "mcnemar_hit5": mcnemar_exact(
                [bool(x) for x in acc[fk]["hit5"]],
                [bool(x) for x in acc[hk]["hit5"]],
            ),
            "bootstrap_mrr5": paired_bootstrap(acc[fk]["rr5"], acc[hk]["rr5"]),
        }

    # Per-category breakdown
    categories = sorted({r["category"] for r in rows})
    summary["per_category"] = {}
    for cat in categories:
        cat_rows = [r for r in rows if r["category"] == cat]
        cat_summary: Dict[str, Any] = {"n": len(cat_rows)}
        for cond in CONDITIONS:
            for rmode in RERANKER_MODES:
                prefix = f"{cond}_{rmode}"
                h5_key = f"{prefix}_hit5"
                rr5_key = f"{prefix}_rr5"
                if not cat_rows or h5_key not in cat_rows[0]:
                    continue
                vals_h = [r.get(h5_key, 0) for r in cat_rows]
                vals_r = [r.get(rr5_key, 0.0) for r in cat_rows]
                cat_summary[f"{prefix}_hr5"] = sum(vals_h) / len(vals_h)
                cat_summary[f"{prefix}_mrr5"] = sum(vals_r) / len(vals_r)
        summary["per_category"][cat] = cat_summary

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description="Exp2a-v2: 2×2 chunking × reranker evaluation")
    p.add_argument("--manifest", default="evals/data/rq1_manifest.csv")
    p.add_argument("--dataset", default="evals/data/subset_a.csv")
    p.add_argument("--adjacent-dataset", default=None)
    p.add_argument("--output", default="evals/data/results/exp2a_v2_chunking.csv")
    p.add_argument("--hier-collection", default=HIER_COLLECTION)
    p.add_argument("--fixed-collection", default=FIXED_COLLECTION)
    p.add_argument("--evaluate", action="store_true")
    p.add_argument("--skip-llm", action="store_true")
    p.add_argument("--llm-model", default=None)
    p.add_argument("--llm-concurrency", type=int, default=DEFAULT_LLM_CONCURRENCY,
                   help="Max concurrent LLM calls (default: 4)")
    p.add_argument("--resume", action="store_true",
                   help="Skip questions already in the output CSV")
    args = p.parse_args()
    if not args.evaluate:
        p.error("pass --evaluate to run")
    asyncio.run(
        evaluate(
            manifest_path=args.manifest,
            dataset_path=args.dataset,
            adjacent_path=args.adjacent_dataset,
            output_csv=args.output,
            hier_collection=args.hier_collection,
            fixed_collection=args.fixed_collection,
            skip_llm=args.skip_llm,
            llm_model=args.llm_model,
            llm_concurrency=args.llm_concurrency,
            resume=args.resume,
        )
    )


if __name__ == "__main__":
    main()
