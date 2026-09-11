"""Fine-tune mmBERT-small on IndoNLI train (3-way NLI).

Trains ``jhu-clsp/mmBERT-small`` (a 140M multilingual encoder) for the
entailment/neutral/contradiction task, then scores it on IndoNLI ``val`` before
and after so the adaptation effect is visible. The resulting checkpoint is
served by ``services/nli`` and selected at runtime with
``CHAT_NLI_MODEL_KIND=mmbert``.

IndoNLI label encoding is pinned to ``{0: entailment, 1: neutral, 2: contradiction}``
(matching indo-roberta-indonli's ``label_0/1/2`` convention), asserted after
training — an inverted label order would silently flip the RAM verdict while
every training metric still looked correct.

Usage:
    python -m evals._train.train_nli_mmbert \\
        --output models/mmbert_nli_id --epochs 3 --batch-size 16 --device cuda
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import structlog
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from evals._shared.dataset import IndoNLIRow, load_indonli
from evals._shared.metrics import compute_multiclass_metrics

logger = structlog.get_logger(__name__)

BASE_MODEL = "jhu-clsp/mmBERT-small"

# Pinned 3-way label order, matching indo-roberta-indonli and the RAM service.
ID2LABEL = {0: "entailment", 1: "neutral", 2: "contradiction"}
LABEL2ID = {"entailment": 0, "neutral": 1, "contradiction": 2}
NLI_LABELS = ["entailment", "neutral", "contradiction"]

# mmBERT's context is 8192, but IndoNLI premises/hypotheses are short sentences;
# 256 covers effectively all rows and keeps batches compact on a single GPU.
TRAIN_MAX_LENGTH = 256
EVAL_MAX_LENGTH = 512


def _label_id(label: str) -> int:
    return LABEL2ID[label]


def build_model(token: str | None = None):
    """Load mmBERT-small with a 3-way classification head attached."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=token)
    # mmBERT uses the Gemma-2 tokenizer, which may not define a pad token.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
        token=token,
    )
    model.config.id2label = ID2LABEL
    model.config.label2id = LABEL2ID

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "train.model_ready",
        total_params=sum(p.numel() for p in model.parameters()),
        trainable_params=trainable,
        id2label=ID2LABEL,
    )
    return tokenizer, model


def _encode(tokenizer, rows: list[IndoNLIRow], max_length: int) -> dict:
    premises = [r.premise for r in rows]
    hypotheses = [r.hypothesis for r in rows]
    return tokenizer(
        premises,
        hypotheses,
        truncation=True,
        max_length=max_length,
        padding=True,
        return_tensors="pt",
    )


def score_model(model, tokenizer, rows: list[IndoNLIRow], device: str, batch_size: int = 64) -> list[str]:
    """Return the predicted label for each row."""
    model.eval()
    predictions: list[str] = []
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            encoded = _encode(tokenizer, batch, EVAL_MAX_LENGTH).to(device)
            logits = model(**encoded).logits
            pred_ids = logits.argmax(-1).cpu().tolist()
            predictions.extend(ID2LABEL[i] for i in pred_ids)
    return predictions


def evaluate(model, tokenizer, rows: list[IndoNLIRow], device: str, tag: str) -> dict:
    """Score IndoNLI val and return 3-way metrics."""
    truths = [r.label for r in rows]
    predictions = score_model(model, tokenizer, rows, device)
    m = compute_multiclass_metrics(predictions, truths, NLI_LABELS)
    result = {
        "tag": tag,
        "n": len(rows),
        "accuracy": m.accuracy,
        "macro_precision": m.macro_precision,
        "macro_recall": m.macro_recall,
        "macro_f1": m.macro_f1,
        "cohen_kappa": m.cohen_kappa,
    }
    print(
        f"  {tag:12s} n={result['n']:5d}  acc={result['accuracy']:.4f}  "
        f"macro_F1={result['macro_f1']:.4f}  kappa={result['cohen_kappa']:.4f}"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune mmBERT-small on IndoNLI train")
    parser.add_argument("--output", default="models/mmbert_nli_id")
    parser.add_argument("--results", default="evals/data/results/nli_mmbert_finetune.json")
    parser.add_argument("--data-dir", default="evals/data")
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    import os

    token = os.getenv("HF_TOKEN") or None
    device = args.device

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_rows = load_indonli("train", data_dir=args.data_dir)
    val_rows = load_indonli("val", data_dir=args.data_dir)
    logger.info(
        "train.start", device=device, train_rows=len(train_rows), val_rows=len(val_rows),
        transformers=__import__("transformers").__version__,
    )

    tokenizer, model = build_model(token)
    model.to(device)

    print("\n" + "=" * 78)
    print("  BASELINE (off the shelf, on IndoNLI val)")
    print("=" * 78)
    baseline = evaluate(model, tokenizer, val_rows, device, "off-the-shelf")

    # --- training -----------------------------------------------------------
    from datasets import Dataset
    from transformers import DataCollatorWithPadding, Trainer, TrainingArguments

    def to_dataset(rows: list[IndoNLIRow]) -> Dataset:
        return Dataset.from_dict(
            {
                "premise": [r.premise for r in rows],
                "hypothesis": [r.hypothesis for r in rows],
                "labels": [_label_id(r.label) for r in rows],
            }
        )

    def tokenize_fn(examples):
        return tokenizer(
            examples["premise"],
            examples["hypothesis"],
            truncation=True,
            max_length=TRAIN_MAX_LENGTH,
        )

    def compute_metrics(prediction) -> Dict[str, float]:
        predicted = prediction.predictions.argmax(-1)
        truth = prediction.label_ids
        acc = accuracy_score(truth, predicted)
        p, r, f1, _ = precision_recall_fscore_support(truth, predicted, average="macro", zero_division=0)
        return {"accuracy": acc, "macro_precision": p, "macro_recall": r, "macro_f1": f1}

    train_dataset = to_dataset(train_rows).map(tokenize_fn, batched=True, remove_columns=["premise", "hypothesis"])
    val_dataset = to_dataset(val_rows).map(tokenize_fn, batched=True, remove_columns=["premise", "hypothesis"])

    training_args = TrainingArguments(
        output_dir=str(Path(args.output) / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=64,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        logging_steps=25,
        fp16=(device == "cuda"),
        seed=args.seed,
        report_to="none",
        dataloader_num_workers=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )
    print("\n" + "=" * 78)
    print(f"  TRAINING  ({len(train_rows)} rows, {args.epochs} epochs, device={device})")
    print("=" * 78)
    trainer.train()

    print("\n" + "=" * 78)
    print("  FINE-TUNED (IndoNLI val)")
    print("=" * 78)
    finetuned = evaluate(model, tokenizer, val_rows, device, "fine-tuned")

    # --- save + assert label order -----------------------------------------
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output))
    tokenizer.save_pretrained(str(output))

    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    id2label = {int(k): v for k, v in config.get("id2label", {}).items()}
    if id2label.get(0) != "entailment" or id2label.get(2) != "contradiction":
        raise AssertionError("LABEL ORDER INVERTED — do not deploy this checkpoint")
    logger.info("train.label_order_verified", id2label=id2label)

    results_path = Path(args.results)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(
        json.dumps(
            {
                "base_model": BASE_MODEL,
                "transformers_version": __import__("transformers").__version__,
                "torch_version": torch.__version__,
                "device": device,
                "seed": args.seed,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "train_max_length": TRAIN_MAX_LENGTH,
                "eval_max_length": EVAL_MAX_LENGTH,
                "train_rows": len(train_rows),
                "val_rows": len(val_rows),
                "baseline": baseline,
                "finetuned": finetuned,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"\n  model   -> {output}")
    print(f"  results -> {results_path}")
    print("\n  To serve it: docker compose up -d nli (CHAT_NLI_MODEL_KIND=mmbert).")


if __name__ == "__main__":
    main()
