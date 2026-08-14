"""Build the external held-out evaluation slices for Exp1a.

Subset B's ``hidden_instruction`` rows are stylistically narrow, so a
classifier fine-tuned on synthetic Indonesian data can score near-perfectly by
learning the style rather than the attack. The fine-tuned model is therefore
also scored on data it cannot have learned the style of: the **official test
split** of a public, human-written prompt-injection corpus, which no part of
B-Train draws from. Two slices are produced — the English original, and an
Indonesian translation, since the deployed system serves Indonesian.

The interpretation rule is fixed in advance, so it cannot be chosen after
seeing the numbers:

- Improves on Subset B **and** the held-out slices → the Indonesian adaptation
  claim holds.
- Improves on Subset B **only** → what was measured is style-matching.

Translation independence: B-Train's translated rows and this slice's
translated rows must not come from the same model, or "translationese" becomes
a feature shared by train and test. The default translator here is a different
lab from B-Train's default generator, and the builder refuses to run if the
two match.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

import structlog

from evals._dataset_gen.build_subset_b_train import (
    DEFAULT_BTRAIN_GENERATOR,
    dedup_internal,
    find_overlap,
    normalize,
    translate_batch,
)
from evals._dataset_gen.checkpoint import IncrementalCSVWriter, resume_rows
from evals._dataset_gen.config import DatasetGenSettings, get_dataset_gen_settings
from evals._dataset_gen.generator import DatasetGenerator
from evals._dataset_gen.provenance import write_provenance

logger = structlog.get_logger(__name__)

# Matches Subset B's schema exactly, so ``_shared/dataset.py::load_subset_b``
# reads these files unchanged and Exp1a needs no special case.
FIELDNAMES = ["query", "label", "attack_type"]

# The held-out source. Its official train split feeds B-Train; only the test
# split is read here, so the two are disjoint by construction rather than by a
# deduplication pass that could silently under-match.
HELDOUT_DATASET = "xTRam1/safe-guard-prompt-injection"
HELDOUT_SPLIT = "test"
TEXT_FIELD = "text"
LABEL_FIELD = "label"
MALICIOUS_VALUES = {1, "1"}

# A different lab from DEFAULT_BTRAIN_GENERATOR (translation independence).
DEFAULT_TRANSLATE_MODEL = "mistralai/mistral-small-3.2-24b-instruct"


def load_heldout_english(sample: int, seed: int) -> List[Dict[str, str]]:
    """Load the public test split and normalise it to Subset B's schema.

    ``sample`` is rows to keep per class (0 keeps everything); sampling is
    balanced, because an imbalanced external set would let the majority class
    drive the headline accuracy.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("datagen.heldout.datasets_missing", detail="pip install datasets")
        sys.exit(1)

    ds = load_dataset(HELDOUT_DATASET, split=HELDOUT_SPLIT)
    malicious: List[Dict[str, str]] = []
    benign: List[Dict[str, str]] = []

    for record in ds:
        text = str(record.get(TEXT_FIELD, "") or "").strip()
        if len(text) < 12:
            continue
        is_attack = record.get(LABEL_FIELD) in MALICIOUS_VALUES
        row = {
            "query": text,
            "label": "malicious" if is_attack else "safe",
            "attack_type": "public_injection" if is_attack else "safe_public",
        }
        (malicious if is_attack else benign).append(row)

    malicious = dedup_internal(malicious)
    benign = dedup_internal(benign)

    rng = random.Random(seed)
    rng.shuffle(malicious)
    rng.shuffle(benign)

    if sample:
        per_class = min(sample // 2, len(malicious), len(benign))
        malicious = malicious[:per_class]
        benign = benign[:per_class]

    rows = malicious + benign
    rng.shuffle(rows)
    logger.info(
        "datagen.heldout.loaded",
        dataset=HELDOUT_DATASET,
        split=HELDOUT_SPLIT,
        rows_in_split=ds.num_rows,
        kept=len(rows),
        malicious=len(malicious),
        safe=len(benign),
    )
    return rows


def assert_disjoint_from_training(
    rows: List[Dict[str, str]],
    b_train_path: str,
    subset_b_path: str,
) -> Dict[str, int]:
    """Verify the held-out slice shares no row with the training data.

    The split boundary should already guarantee this, but source corpora do
    repeat rows across splits, and an unchecked "held out" claim is not
    evidence.

    Raises:
        AssertionError: If any held-out row appears in the training data.
    """
    report: Dict[str, int] = {}
    for name, path in (("b_train", b_train_path), ("subset_b", subset_b_path)):
        file = Path(path)
        if not file.exists():
            logger.warning("datagen.heldout.no_comparison_file", path=str(file))
            report[name] = -1
            continue
        with file.open(newline="", encoding="utf-8") as f:
            other = [normalize(r.get("query", "")) for r in csv.DictReader(f)]
        overlap = find_overlap(rows, other)
        report[name] = len(overlap)
        if overlap:
            raise AssertionError(
                f"{len(overlap)} held-out rows also appear in {file.name}; "
                "the external evaluation would not be held out"
            )
    return report


async def build_heldout_eval(
    settings: DatasetGenSettings,
    output_en: str,
    output_id: str,
    b_train_path: str,
    subset_b_path: str,
    sample: int,
    seed: int,
    resume: bool = False,
    skip_translation: bool = False,
) -> None:
    """Build the English and Indonesian held-out evaluation slices."""
    rows = load_heldout_english(sample, seed)
    overlap_report = assert_disjoint_from_training(rows, b_train_path, subset_b_path)

    with IncrementalCSVWriter(output_en, FIELDNAMES, resume=False) as writer:
        for row in rows:
            writer.append(row)

    write_provenance(
        output_path=output_en,
        subset="heldout_injection_en",
        settings=settings,
        row_count=len(rows),
        extra={
            "role": "external held-out evaluation — never used for training",
            "source_dataset": HELDOUT_DATASET,
            "source_split": HELDOUT_SPLIT,
            "language": "en",
            "sample": sample,
            "seed": seed,
            "by_label": dict(Counter(r["label"] for r in rows)),
            "overlap_with_training": overlap_report,
        },
    )
    logger.info("datagen.heldout.en_written", rows=len(rows), output=output_en)

    if skip_translation:
        print(f"\nHeld-out (en) -> {output_en}: {len(rows)} rows")
        return

    if not settings.openrouter_api_key:
        logger.error("datagen.heldout.missing_api_key")
        sys.exit(1)

    generator = DatasetGenerator(settings)
    translated: List[Dict[str, str]] = list(
        resume_rows(output_id, FIELDNAMES) if resume else []
    )
    done = len(translated)

    try:
        with IncrementalCSVWriter(output_id, FIELDNAMES, resume=resume) as writer:
            batch_size = 10
            for start in range(done, len(rows), batch_size):
                chunk = rows[start : start + batch_size]
                results = await translate_batch(generator, [c["query"] for c in chunk])
                for source_row, text in zip(chunk, results):
                    if not text:
                        continue
                    # Label and subtype carry over untouched: translation must
                    # preserve intent, so a translated attack is still an
                    # attack.
                    row = {
                        "query": text,
                        "label": source_row["label"],
                        "attack_type": source_row["attack_type"],
                    }
                    writer.append(row)
                    translated.append(row)
                logger.info(
                    "datagen.heldout.translate_progress",
                    done=len(translated),
                    total=len(rows),
                )
    finally:
        await generator.aclose()

    write_provenance(
        output_path=output_id,
        subset="heldout_injection_id",
        settings=settings,
        row_count=len(translated),
        extra={
            "role": "external held-out evaluation — never used for training",
            "source_dataset": HELDOUT_DATASET,
            "source_split": HELDOUT_SPLIT,
            "language": "id",
            "translated_from": Path(output_en).name,
            "translation_model": settings.generator_model,
            "translation_independence": (
                "translated by a different model than B-Train's translated rows, so "
                "translationese is not a feature shared between train and test"
            ),
            "by_label": dict(Counter(r["label"] for r in translated)),
        },
    )

    print(f"\nHeld-out (en) -> {output_en}: {len(rows)} rows")
    print(f"Held-out (id) -> {output_id}: {len(translated)} rows")
    print(f"  by label (en) : {dict(Counter(r['label'] for r in rows))}")
    print(f"  by label (id) : {dict(Counter(r['label'] for r in translated))}")
    print(f"  overlap with training data: {overlap_report}")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Build the external held-out evaluation slices for Exp1a"
    )
    parser.add_argument("--output-en", default="evals/data/heldout_injection_en.csv")
    parser.add_argument("--output-id", default="evals/data/heldout_injection_id.csv")
    parser.add_argument("--b-train", default="evals/data/subset_b_train.csv")
    parser.add_argument("--subset-b", default="evals/data/subset_b.csv")
    parser.add_argument(
        "--sample",
        type=int,
        default=600,
        help="Total rows to keep, balanced across classes (0 = the whole split).",
    )
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-translation", action="store_true")
    parser.add_argument(
        "--translate-model",
        default=DEFAULT_TRANSLATE_MODEL,
        help="Must differ from B-Train's generator (see module docstring).",
    )
    args = parser.parse_args()

    if args.translate_model == DEFAULT_BTRAIN_GENERATOR:
        logger.error(
            "datagen.heldout.translator_collision",
            model=args.translate_model,
            detail=(
                "the held-out slice would share a translator with B-Train, making "
                "translationese a feature common to train and test"
            ),
        )
        sys.exit(1)

    settings = get_dataset_gen_settings().model_copy(
        update={"generator_model": args.translate_model}
    )
    asyncio.run(
        build_heldout_eval(
            settings=settings,
            output_en=args.output_en,
            output_id=args.output_id,
            b_train_path=args.b_train,
            subset_b_path=args.subset_b,
            sample=args.sample,
            seed=args.seed,
            resume=args.resume,
            skip_translation=args.skip_translation,
        )
    )


if __name__ == "__main__":
    main()
