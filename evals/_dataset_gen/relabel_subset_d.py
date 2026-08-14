"""Re-split and re-label an existing Subset D-family CSV.

Corrective tool for the citation-marker sentence-splitting bug in
``build_subset_d.split_sentences()`` (splitting on periods inside inline
citation markers like "M.Ag." corrupted ``sentence_text``).

Since ``full_response`` and ``retrieved_context`` are already stored
correctly (verbatim) per question in the existing CSV, this re-derives
sentences from them with the fixed splitter and re-runs ONLY the panel
labeling step — it does NOT re-invoke the live chat pipeline.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import structlog

from evals._dataset_gen.build_subset_d import LABELS, VALIDATION_PROMPT, split_sentences
from evals._dataset_gen.concordance import BlindInjectionTracker
from evals._dataset_gen.config import DatasetGenSettings, get_dataset_gen_settings
from evals._dataset_gen.panel import EvaluatorPanel

logger = structlog.get_logger(__name__)

FIELDNAMES = [
    "question_id", "question", "full_response", "sentence_id",
    "sentence_text", "retrieved_context", "label", "verifier_note",
]


def load_unique_questions(path: str) -> List[Tuple[str, str, str, str]]:
    """Recover one (question_id, question, full_response, retrieved_context) tuple per question."""
    seen: Dict[str, Tuple[str, str, str, str]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            qid = row["question_id"]
            if qid not in seen:
                seen[qid] = (qid, row["question"], row["full_response"], row["retrieved_context"])
    return list(seen.values())


async def relabel(
    settings: DatasetGenSettings,
    input_path: str,
    output_path: str,
) -> None:
    """Re-split and re-label an existing Subset D-family CSV."""
    if not settings.openrouter_api_key:
        logger.error("datagen.relabel_subset_d.missing_api_key")
        sys.exit(1)

    questions = load_unique_questions(input_path)
    logger.info("datagen.relabel_subset_d.loaded_questions", count=len(questions))

    panel = EvaluatorPanel(settings)
    blind_tracker = BlindInjectionTracker()

    accepted_items: List[Dict[str, Any]] = []
    total_accepted = 0
    total_rejected = 0

    try:
        for question_id, question, full_response, retrieved_context in questions:
            sentences = split_sentences(full_response)
            logger.info(
                "datagen.relabel_subset_d.decomposed",
                question_id=question_id,
                sentence_count=len(sentences),
            )

            for sent_idx, sentence in enumerate(sentences):
                validation_context = VALIDATION_PROMPT.format(
                    question=question,
                    full_response=full_response,
                    sentence_id=sent_idx,
                    sentence_text=sentence,
                    retrieved_context=retrieved_context,
                )
                try:
                    verdict = await panel.evaluate_label(
                        prompt="Assign the correct label to this sentence.",
                        context=validation_context,
                        valid_labels=LABELS,
                    )
                except Exception as e:
                    logger.error(
                        "datagen.relabel_subset_d.panel_error",
                        question_id=question_id,
                        sentence_id=sent_idx,
                        error=str(e),
                        exc_info=True,
                    )
                    continue

                if verdict.accepted and verdict.accepted_label:
                    label = verdict.accepted_label
                    vote_summary = ", ".join(f"{v.label or 'none'}" for v in verdict.votes)
                    row = {
                        "question_id": question_id,
                        "question": question,
                        "full_response": full_response,
                        "sentence_id": sent_idx,
                        "sentence_text": sentence,
                        "retrieved_context": retrieved_context,
                        "label": label,
                        "verifier_note": f"Panel majority ({verdict.label_counts}; votes: {vote_summary})",
                    }
                    accepted_items.append(row)
                    total_accepted += 1
                    top_count = max(verdict.label_counts.values()) if verdict.label_counts else 0
                    if top_count == len(verdict.votes):
                        blind_tracker.add_candidate({**row, "_panel_label": label})
                    logger.info(
                        "datagen.relabel_subset_d.sentence_accepted",
                        question_id=question_id,
                        sentence_id=sent_idx,
                        label=label,
                    )
                else:
                    total_rejected += 1
                    logger.info(
                        "datagen.relabel_subset_d.sentence_rejected",
                        question_id=question_id,
                        sentence_id=sent_idx,
                        label_counts=dict(verdict.label_counts),
                    )
    finally:
        await panel.aclose()

    blind_tracker.write_sidecar(
        output_path.replace(".csv", "_blind_injection.csv"),
        fieldnames=FIELDNAMES,
    )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(accepted_items)

    logger.info(
        "datagen.relabel_subset_d.complete",
        questions_processed=len(questions),
        sentences_accepted=total_accepted,
        sentences_rejected=total_rejected,
        output=output_path,
    )


def main() -> None:
    """Entry point for Subset D-family relabeling."""
    parser = argparse.ArgumentParser(
        description="Re-split and re-label an existing Subset D-family CSV using the fixed sentence splitter."
    )
    parser.add_argument("--input", required=True, help="Path to the existing (buggy) CSV")
    parser.add_argument("--output", required=True, help="Path to write the corrected CSV to")
    args = parser.parse_args()

    settings = get_dataset_gen_settings()
    asyncio.run(relabel(settings, args.input, args.output))


if __name__ == "__main__":
    main()
