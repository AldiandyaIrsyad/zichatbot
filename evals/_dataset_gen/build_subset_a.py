"""Build Subset A — RAG QA Triplets.

Generates RAG QA triplets (question, category, ground_truth_answer,
source_doc_id, source_context) from the knowledge base using the
Generator-Evaluator architecture.

Pipeline:
    1. Load KB documents from the running application
    2. Generator produces draft QA triplets from each document
    3. Panel validates: is the question answerable from the context? Is the
       ground_truth_answer hallucination-free?
    4. Accept if ≥4/5 panel members agree
    5. Output CSV with columns: question, category, ground_truth_answer,
       source_doc_id, source_context
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

import httpx
import structlog

from evals._dataset_gen.config import DatasetGenSettings, get_dataset_gen_settings
from evals._dataset_gen.checkpoint import IncrementalCSVWriter, resume_rows
from evals._dataset_gen.concordance import BlindInjectionTracker
from evals._dataset_gen.generator import DatasetGenerator
from evals._dataset_gen.panel import EvaluatorPanel, PanelUnavailableError
from evals._dataset_gen.provenance import append_panel_vote_record, write_provenance

logger = structlog.get_logger(__name__)

CATEGORIES = ["factual", "procedural", "multi-hop", "out-of-domain"]

GENERATOR_SYSTEM_PROMPT = """\
You are a dataset generator for a RAG evaluation benchmark about the internal \
legal/regulatory documents of Universitas Pendidikan Indonesia (UPI), published \
via its legal documentation portal JDIH (Jaringan Dokumentasi dan Informasi \
Hukum): Peraturan Rektor, SK Rektor, Statuta UPI, keputusan Senat Akademik / \
Majelis Wali Amanat (MWA), and internal pedoman. Generate high-quality QA \
triplets in Indonesian.

Quality rules:
- Every in-domain answer must be fully entailed by the supplied document text.
- Copy source_context verbatim; never reconstruct, normalize, or invent it.
- factual uses one explicit fact; procedural requires stated steps/conditions; multi-hop
  requires combining at least two distinct statements present in the supplied text.
- Questions must be self-contained, unique, and must not depend on an unspecified
  "document above". Do not infer policy intent or unstated consequences.
- For out-of-domain, use exactly NONE for answer and context and ask a plausible
  legal/institutional question whose scope is explicitly not UPI internal regulation.


For each item, output a JSON object on its own line (JSONL format) with these fields:
- "question": A question in Indonesian
- "category": One of: factual, procedural, multi-hop, out-of-domain
- "ground_truth_answer": The correct answer based on the context (or "NONE" for out-of-domain)
- "source_context": The verbatim paragraph from the source document (or "NONE" for out-of-domain)

Do not include markdown code fences. Output one JSON object per line.
"""

VALIDATION_PROMPT = """\
You are evaluating a generated QA triplet for a RAG benchmark.

Question: {question}
Category: {category}
Ground Truth Answer: {ground_truth_answer}
Source Context: {source_context}

If Category is "out-of-domain": Ground Truth Answer and Source Context are
EXPECTED to be exactly "NONE" — this is CORRECT by design, not a defect. Do
NOT penalize "NONE"/"NONE" for out-of-domain items, and do NOT require the
question to be answerable from this knowledge base. Evaluate only:
1. Is the question clear (as a standalone question)?
2. Is the question truly outside the domain of UPI's internal legal/regulatory
   documents (JDIH)?

If Category is NOT "out-of-domain", evaluate:
1. Is the question clear and answerable from the source_context?
2. Is every material part of the ground_truth_answer explicitly supported by the verbatim source_context?
3. Does the category match the evidence structure (single fact, stated procedure, or multiple distinct statements)?

Answer YES only if every applicable criterion above is satisfied; otherwise answer NO.
Return one aggregate vote only, with exactly 'YES' or 'NO' and no per-criterion answers.
"""


async def fetch_kb_documents(api_url: str) -> List[Dict[str, Any]]:
    """Fetch all active PDF documents from the KB API."""
    async with httpx.AsyncClient(base_url=api_url, timeout=30.0) as client:
        response = await client.get("/api/admin/pdfs")
        response.raise_for_status()
        docs = response.json()
        return [d for d in docs if d.get("active", True)]


async def fetch_document_text(api_url: str, doc_id: str, doc_title: str) -> str:
    """Fetch text content for a document by searching its chunks.

    Uses the document's own title as the search query (domain-agnostic) and
    post-filters by ``doc_id`` to ensure only this document's chunks are
    joined. ``top_k`` is set to the API maximum (100) to maximise coverage.
    Returns empty string on failure.
    """
    if not doc_title:
        return ""
    async with httpx.AsyncClient(base_url=api_url, timeout=30.0) as client:
        response = await client.get(
            "/api/kb/search",
            params={"q": doc_title, "top_k": 100},
        )
        if response.status_code != 200:
            return ""
        results = response.json()
        return "\n\n".join(
            r.get("text", "") for r in results if r.get("doc_id") == doc_id
        )


async def build_subset_a(
    settings: DatasetGenSettings,
    api_url: str,
    output_path: str,
    count: int,
    seed: int = 42,
    max_items_per_doc: int = 1,
    resume: bool = False,
) -> None:
    """Build Subset A (RAG QA triplets) and save to CSV.

    Args:
        seed: RNG seed for the document shuffle — recorded in the provenance
            sidecar so a run can be reproduced.
        max_items_per_doc: Cap on accepted doc-bound items per source document.
            Defaults to 1, i.e. one question per document. See the
            document-coverage note in the loop below for why.
        resume: Continue an interrupted run, keeping the rows already written
            to ``output_path`` and rebuilding the per-category and
            per-document counters from them.
    """
    if not settings.openrouter_api_key:
        logger.error("datagen.subset_a.missing_api_key")
        sys.exit(1)

    # 1. Fetch KB documents
    logger.info("datagen.subset_a.fetching_docs")
    docs = await fetch_kb_documents(api_url)
    if not docs:
        logger.error("datagen.subset_a.no_active_docs")
        sys.exit(1)
    logger.info("datagen.subset_a.docs_found", count=len(docs))

    # Shuffle so document selection does not depend on the KB API's return
    # order. Seeded so a run stays reproducible.
    #
    # ``max_items_per_doc`` defaults to 1 — one question per source document.
    # The reason is statistical rather than aesthetic: Exp2 makes a
    # DOCUMENT-level claim ("how often does retrieval surface the right
    # document?"), and two questions drawn from the same document are not two
    # independent observations of that. A document that is hard to retrieve —
    # poor chunking, scan-only pages, an undistinctive title — tends to make
    # both of its questions miss together, so the effective sample size for
    # Hit Rate@k tracks the number of distinct documents, not the number of
    # questions. One question per document makes those two counts equal. It
    # also doubles corpus coverage for the same question budget, and gives a
    # sampling rule that needs no defending ("each question is drawn from a
    # distinct document") rather than an arbitrary cap.
    docs = list(docs)
    random.Random(seed).shuffle(docs)

    # 2. Initialize generator and panel
    generator = DatasetGenerator(settings)
    panel = EvaluatorPanel(settings)
    blind_tracker = BlindInjectionTracker()

    accepted_items: List[Dict[str, str]] = list(
        resume_rows(output_path, ["question", "category", "ground_truth_answer", "source_doc_id", "source_context"]) if resume else []
    )
    total_generated = 0
    total_rejected = 0

    # Per-category targets (total 150). Scaled proportionally when --count
    # differs from that total.
    #
    # Sizing rationale (95% Wilson intervals): out-of-domain and multi-hop are
    # the proportions on small slices that drive the widest intervals, so they
    # are sized to roughly ±14pp. factual/procedural feed BERTScore, whose
    # interval is already narrow at n≈87, so they need not grow. The binding
    # constraints are all proportions on small slices, not the continuous
    # metrics.
    CATEGORY_TARGETS = {"factual": 40, "procedural": 40, "multi-hop": 35, "out-of-domain": 35}
    _target_total = sum(CATEGORY_TARGETS.values())
    scale = count / _target_total if count != _target_total else 1.0
    category_targets = {cat: max(1, int(t * scale)) for cat, t in CATEGORY_TARGETS.items()}

    # Track accepted counts per category. Seeded from any resumed rows so a
    # continued run tops up the shortfall rather than restarting the targets.
    accepted_per_category: Dict[str, int] = {cat: 0 for cat in CATEGORIES}
    for _row in accepted_items:
        _cat = _row.get("category", "")
        if _cat in accepted_per_category:
            accepted_per_category[_cat] += 1
    # Track accepted counts per source document, to spread questions across the
    # corpus instead of exhausting the targets on the first few documents.
    # Only doc-bound categories count against the cap — out-of-domain items are
    # not attributable to a source document (source_doc_id is "NONE").
    accepted_per_doc: Dict[str, int] = {}
    for _row in accepted_items:
        _doc = _row.get("source_doc_id", "")
        if _doc and _doc != "NONE":
            accepted_per_doc[_doc] = accepted_per_doc.get(_doc, 0) + 1

    # Rows are written to the CSV as they are accepted, so an interruption
    # (outage, Ctrl-C, closed laptop) leaves a valid partial dataset on disk
    # that --resume can continue from, instead of discarding the API spend
    # behind it.
    writer_ctx = IncrementalCSVWriter(output_path, ["question", "category", "ground_truth_answer", "source_doc_id", "source_context"], resume=resume)
    try:
        with writer_ctx as row_writer:
            # Iterate documents round-robin, generating per-category items until targets met
            doc_cycle = 0
            max_cycles = 5  # safety bound to avoid infinite loops if acceptance rate is low
            while (
                any(accepted_per_category[c] < category_targets[c] for c in CATEGORIES)
                and len(accepted_items) < count
                and doc_cycle < max_cycles
            ):
                # The cap loosens each cycle, so a corpus too small (or with too low
                # an acceptance rate) to hit the targets at ``max_items_per_doc``
                # still converges on a later pass rather than stalling.
                doc_cap = max_items_per_doc * (doc_cycle + 1)

                for doc in docs:
                    if all(accepted_per_category[c] >= category_targets[c] for c in CATEGORIES):
                        break

                    doc_id = doc.get("id", "")
                    doc_title = doc.get("title", "Unknown")

                    # Skip documents already at the cap before paying for their text
                    # fetch. Out-of-domain items don't need document text at all, so
                    # they are only generated from documents still under the cap —
                    # they'd otherwise re-fetch every capped document on later cycles.
                    if accepted_per_doc.get(doc_id, 0) >= doc_cap:
                        continue

                    logger.info(
                        "datagen.subset_a.processing_doc",
                        cycle=doc_cycle + 1,
                        doc_id=doc_id,
                        doc_title=doc_title,
                    )

                    # Fetch document text
                    doc_text = await fetch_document_text(api_url, doc_id, doc_title)
                    if not doc_text:
                        logger.warning(
                            "datagen.subset_a.no_text_for_doc",
                            doc_id=doc_id,
                            doc_title=doc_title,
                        )
                        continue

                    # Generate drafts for each category that still needs items
                    for category in CATEGORIES:
                        remaining = category_targets[category] - accepted_per_category[category]
                        if remaining <= 0:
                            continue

                        # Doc-bound categories also respect the per-document cap, so
                        # a single document cannot supply a large share of the set.
                        doc_bound = category != "out-of-domain"
                        if doc_bound:
                            doc_remaining = doc_cap - accepted_per_doc.get(doc_id, 0)
                            if doc_remaining <= 0:
                                continue
                            remaining = min(remaining, doc_remaining)

                        items_per_category = 5  # minimum batch; accepted rows remain capped per document
                        logger.info(
                            "datagen.subset_a.generating",
                            category=category,
                            batch_size=items_per_category,
                            target=category_targets[category],
                            have=accepted_per_category[category],
                        )

                        seed_prompt = (
                            f"Based on this UPI internal legal document (JDIH):\n\n{doc_text[:3000]}\n\n"
                            f"Generate {items_per_category} {category} questions in Indonesian. "
                            f"Category '{category}' means: "
                        )
                        if category == "factual":
                            seed_prompt += "questions answerable from a single paragraph."
                        elif category == "procedural":
                            seed_prompt += "questions about steps, processes, or procedures."
                        elif category == "multi-hop":
                            seed_prompt += "questions requiring synthesis of multiple paragraphs."
                        else:  # out-of-domain
                            seed_prompt += (
                                "questions OUTSIDE the domain of UPI internal legal documents "
                                "(e.g. general national law, other institutions, or unrelated topics)."
                            )

                        try:
                            drafts = await generator.generate(
                                seed_prompt=seed_prompt,
                                count=items_per_category,
                                system_prompt=GENERATOR_SYSTEM_PROMPT,
                            )
                        except Exception as e:
                            logger.error("datagen.subset_a.generator_error", error=str(e), exc_info=True)
                            continue

                        total_generated += len(drafts)

                        # Validate each draft
                        for draft in drafts:
                            if accepted_per_category[category] >= category_targets[category]:
                                break
                            if len(accepted_items) >= count:
                                break
                            if doc_bound and accepted_per_doc.get(doc_id, 0) >= doc_cap:
                                break

                            if not isinstance(draft.parsed, dict):
                                continue

                            item = draft.parsed
                            question = item.get("question", "").strip()
                            if not question:
                                continue

                            # Validate with panel
                            validation_context = VALIDATION_PROMPT.format(
                                question=question,
                                category=item.get("category", category),
                                ground_truth_answer=item.get("ground_truth_answer", ""),
                                source_context=item.get("source_context", ""),
                            )

                            try:
                                verdict = await panel.evaluate(
                                    prompt="Is this QA triplet valid and high-quality?",
                                    context=validation_context,
                                )
                            except PanelUnavailableError:
                                # The API is down, not this candidate. Let it propagate so the
                                # run stops with its output intact instead of burning the batch
                                # budget marking every item rejected; --resume continues it.
                                raise
                            except Exception as e:
                                logger.error("datagen.subset_a.panel_error", error=str(e), exc_info=True)
                                continue

                            append_panel_vote_record(
                                output_path, "a",
                                {"question": question, "category": item.get("category", category), "source_doc_id": doc_id},
                                verdict, verdict.accepted,
                            )
                            if verdict.accepted:
                                row = {
                                    "question": question,
                                    "category": item.get("category", category),
                                    "ground_truth_answer": item.get("ground_truth_answer", ""),
                                    "source_doc_id": doc_id if category != "out-of-domain" else "NONE",
                                    "source_context": item.get("source_context", ""),
                                }
                                accepted_items.append(row)
                                row_writer.append(row)
                                accepted_per_category[category] += 1
                                if doc_bound:
                                    accepted_per_doc[doc_id] = accepted_per_doc.get(doc_id, 0) + 1
                                # Track 5/5-unanimous items for blind injection
                                if verdict.yes_count == len(verdict.votes):
                                    blind_tracker.add_candidate({**row, "_panel_yes": verdict.yes_count})
                                logger.info(
                                    "datagen.subset_a.accepted",
                                    category=category,
                                    accepted=accepted_per_category[category],
                                    target=category_targets[category],
                                )
                            else:
                                total_rejected += 1
                                logger.info(
                                    "datagen.subset_a.rejected",
                                    yes=verdict.yes_count,
                                    total=verdict.no_count + verdict.yes_count,
                                )
                doc_cycle += 1

    finally:
        await generator.aclose()
        await panel.aclose()

    # Write blind-injection sidecar
    blind_tracker.write_sidecar(
        output_path.replace(".csv", "_blind_injection.csv"),
        fieldnames=["question", "category", "ground_truth_answer", "source_doc_id", "source_context"],
    )

    # The CSV is already complete: rows were flushed as they were accepted
    # (see IncrementalCSVWriter above), which is what makes an interrupted run
    # resumable.

    write_provenance(
        output_path,
        subset="a",
        settings=settings,
        row_count=len(accepted_items),
        extra={
            "seed": seed,
            "max_items_per_doc": max_items_per_doc,
            "kb_document_count": len(docs),
            "distinct_source_documents": len(accepted_per_doc),
            "target_count": count,
            "category_targets": category_targets,
            "generated": total_generated,
            "rejected": total_rejected,
        },
    )

    logger.info(
        "datagen.subset_a.complete",
        generated=total_generated,
        accepted=len(accepted_items),
        rejected=total_rejected,
        distinct_docs=len(accepted_per_doc),
        output=output_path,
    )


def main() -> None:
    """Entry point for Subset A generation."""
    parser = argparse.ArgumentParser(
        description="Build Subset A (RAG QA Triplets) using Generator-Evaluator architecture."
    )
    parser.add_argument(
        "--api-url",
        default="http://localhost:8000",
        help="Base URL of the running application",
    )
    parser.add_argument(
        "--output",
        default="evals/data/subset_a.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=150,
        help="Target number of accepted items. Note the per-category targets are "
        "scaled by int() truncation, so the delivered total lands slightly under "
        "this (150 -> 150, but e.g. 100 -> 98).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue an interrupted run: keep the rows already in --output and "
        "generate only the shortfall. Rows are written as they are accepted, so a "
        "run stopped by an outage or Ctrl-C can always be continued this way.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for the document shuffle (recorded in the .meta.json sidecar)",
    )
    parser.add_argument(
        "--max-items-per-doc",
        type=int,
        default=1,
        help="Cap on accepted doc-bound items per source document (default 1, i.e. one "
        "question per document) so the set spans the corpus and each question is an "
        "independent observation for Exp2's document-level retrieval metrics. The cap "
        "loosens by this amount on each additional pass, so a corpus too small to fill "
        "the targets at this cap still converges.",
    )
    args = parser.parse_args()

    settings = get_dataset_gen_settings()
    asyncio.run(
        build_subset_a(
            settings,
            args.api_url,
            args.output,
            args.count,
            seed=args.seed,
            max_items_per_doc=args.max_items_per_doc,
            resume=args.resume,
        )
    )


if __name__ == "__main__":
    main()
