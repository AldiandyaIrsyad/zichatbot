"""Fine-tune the IVM safety classifier on Subset B-Train, locally.

Runs the whole experiment in one process: score the off-the-shelf model, train,
score the fine-tuned model on the same sets with the same code, and print the
comparison plus the pre-registered verdict.

Held-out discipline
-------------------
Subset B and both external slices are scored, never trained on, and the overlap
assertion runs again here rather than being trusted from the build step.

The verdict rule is applied as written, decided before any number existed:
improvement on Subset B *and* on held-out data supports the adaptation claim;
improvement on Subset B alone is style-matching against the generator's
templates and must be reported as such.

Usage:
    python -m evals._train.train_prompt_guard \\
        --output models/prompt_guard_id [--epochs 3] [--batch-size 16]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import structlog
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

logger = structlog.get_logger(__name__)

BASE_MODEL = "meta-llama/Llama-Prompt-Guard-2-86M"

# Training truncation. B-Train's median row is 29 tokens and its 95th percentile
# is 81, but its 99th is the full 512 — and with dynamic padding a single long
# row inflates its whole batch to 512, where DeBERTa-v2's disentangled attention
# is quadratic and expensive. Capping at 256 covers ~97% of rows untouched and
# is the difference between training in VRAM and paging over PCIe.
#
# Inference is unaffected: the served model still accepts the full 512.
TRAIN_MAX_LENGTH = 256

# Evaluation truncates at the model's full context, matching the deployed guard.
# Training and evaluation limits are deliberately independent: capping training
# length is a memory decision, but capping EVALUATION length would change the
# reported numbers, and the figures must describe the system as actually served.
EVAL_MAX_LENGTH = 512

# Pinned, and asserted after training. The deployed client treats index 1 as the
# malicious class; if a fine-tune swaps these the guard inverts — passing
# attacks and blocking legitimate queries — while every training metric still
# looks correct.
ID2LABEL = {0: "BENIGN", 1: "MALICIOUS"}
LABEL2ID = {"BENIGN": 0, "MALICIOUS": 1}

# Matches the deployed guard, so the reported figures describe the system as it
# will actually behave rather than an argmax that nothing uses.
SECURITY_THRESHOLD = 0.75


@dataclass
class EvalResult:
    """Metrics for one model on one dataset."""

    model: str
    dataset: str
    n: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    fpr: float


def normalize(text: str) -> str:
    """Fold a query for duplicate detection (mirrors the dataset builders)."""
    folded = unicodedata.normalize("NFKC", str(text or "").lower())
    folded = re.sub(r"[^\w\s]", " ", folded)
    return re.sub(r"\s+", " ", folded).strip()


def assert_held_out(train: pd.DataFrame, eval_sets: Sequence[Tuple[pd.DataFrame, str]]) -> None:
    """Re-verify that no evaluation row appears in the training data.

    The builders already assert this, but a claim worth making is worth checking
    where it is used — a stale CSV or a wrong path would otherwise go unnoticed.

    Raises:
        SystemExit: If any evaluation row is present in the training set.
    """
    train_norm = {normalize(q) for q in train["query"]}
    leaked = False
    for frame, name in eval_sets:
        overlap = sum(1 for q in frame["query"] if normalize(q) in train_norm)
        logger.info(
            "train.held_out_check", dataset=name, rows=len(frame), shared_with_train=overlap
        )
        if overlap:
            leaked = True
    if leaked:
        logger.error(
            "train.held_out_violation",
            detail="evaluation rows appear in the training data; results would be invalid",
        )
        sys.exit(1)


def load_frames(data_dir: Path) -> Tuple[pd.DataFrame, List[Tuple[pd.DataFrame, str]], Optional[pd.DataFrame]]:
    """Load the training set, the evaluation sets, and the policy annotation.

    Returns:
        (training frame, [(frame, name), ...], optional override annotation).
    """
    train = pd.read_csv(data_dir / "subset_b_train.csv")
    eval_sets = [
        (pd.read_csv(data_dir / "subset_b.csv"), "Subset B"),
        (pd.read_csv(data_dir / "heldout_injection_en.csv"), "held-out EN"),
        (pd.read_csv(data_dir / "heldout_injection_id.csv"), "held-out ID"),
    ]
    slices_path = data_dir / "subset_b_slices.csv"
    slices = pd.read_csv(slices_path) if slices_path.exists() else None
    return train, eval_sets, slices


def build_model(token: Optional[str] = None, freeze_embeddings: bool = True):
    """Load the base classifier with the label order pinned.

    Args:
        token: Hugging Face token (the base checkpoint is gated).
        freeze_embeddings: Whether to hold the word-embedding table fixed.

    Returns:
        A tuple of (tokenizer, model).
    """
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=token)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
        token=token,
    )
    assert model.config.id2label[0] == "BENIGN"
    assert model.config.id2label[1] == "MALICIOUS"

    if freeze_embeddings:
        # The 251k-token multilingual vocabulary table is 192.8M of this
        # model's 278.8M parameters — 69% — and freezing it leaves exactly the
        # 86M encoder the model is named for.
        #
        # This is a modelling decision before it is a memory one. The table
        # already represents Indonesian from mDeBERTa pretraining; what has to
        # change to recognise Indonesian attacks is the encoder and the
        # classifier head. Re-estimating 192M embedding parameters from 2.5k
        # examples mostly invites overfitting.
        #
        # It also removes their gradients and AdamW moments, ~2.3 GB, which is
        # the difference between training in VRAM and paging over PCIe.
        for parameter in model.deberta.embeddings.word_embeddings.parameters():
            parameter.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "train.model_ready",
        total_params=sum(p.numel() for p in model.parameters()),
        trainable_params=trainable,
        embeddings_frozen=freeze_embeddings,
    )
    return tokenizer, model


def score_malicious(model, tokenizer, texts: Sequence[str], batch_size: int = 64) -> np.ndarray:
    """Return P(malicious) for each text."""
    model.eval()
    scores: List[float] = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            encoded = tokenizer(
                list(texts[start : start + batch_size]),
                truncation=True,
                max_length=EVAL_MAX_LENGTH,
                padding=True,
                return_tensors="pt",
            ).to(model.device)
            probabilities = torch.softmax(model(**encoded).logits, dim=-1)
            scores.extend(probabilities[:, LABEL2ID["MALICIOUS"]].cpu().tolist())
    return np.array(scores)


def evaluate(model, tokenizer, frame: pd.DataFrame, dataset: str, tag: str) -> EvalResult:
    """Score one dataset at the deployed threshold."""
    truth = (frame["label"] == "malicious").astype(int).to_numpy()
    predicted = (
        score_malicious(model, tokenizer, frame["query"].astype(str).tolist())
        >= SECURITY_THRESHOLD
    ).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        truth, predicted, average="binary", zero_division=0
    )
    negatives = int((truth == 0).sum())
    fpr = float(((predicted == 1) & (truth == 0)).sum() / negatives) if negatives else float("nan")

    result = EvalResult(
        model=tag,
        dataset=dataset,
        n=len(frame),
        accuracy=float(accuracy_score(truth, predicted)),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        fpr=fpr,
    )
    print(
        f"  {tag:14s} {dataset:14s} n={result.n:5d}  acc={result.accuracy:.4f}  "
        f"P={result.precision:.4f}  R={result.recall:.4f}  F1={result.f1:.4f}  FPR={result.fpr:.4f}"
    )
    return result


def report_per_subtype(model, tokenizer, frame: pd.DataFrame) -> List[Dict[str, object]]:
    """Break Subset B down by attack subtype.

    The aggregate can improve while the subtype that actually fails stays
    broken, and ``hidden_instruction`` is both the subtype this fine-tune
    targets and the one carrying the reported failure.
    """
    rows: List[Dict[str, object]] = []
    print("\n  Per subtype (fine-tuned, Subset B):")
    print(f"    {'subtype':<22} {'n':>5} {'acc':>8} {'FPR':>8}")
    for subtype, group in frame.groupby("attack_type"):
        truth = (group["label"] == "malicious").astype(int).to_numpy()
        predicted = (
            score_malicious(model, tokenizer, group["query"].astype(str).tolist())
            >= SECURITY_THRESHOLD
        ).astype(int)
        negatives = int((truth == 0).sum())
        # Undefined for a single-class subtype; printing 0.0000 there would look
        # like a measurement rather than the absence of one.
        fpr = float(((predicted == 1) & (truth == 0)).sum() / negatives) if negatives else float("nan")
        accuracy = float(accuracy_score(truth, predicted))
        rows.append({"attack_type": subtype, "n": len(group), "accuracy": accuracy, "fpr": fpr})
        print(f"    {subtype:<22} {len(group):>5} {accuracy:>8.4f} {fpr:>8.4f}")
    return rows


def report_policy_scope(
    model, tokenizer, frame: pd.DataFrame, slices: Optional[pd.DataFrame]
) -> List[Dict[str, object]]:
    """Split malicious rows by whether they attempt an instruction override.

    Prompt Guard 2 flags a prompt only when it explicitly tries to override
    prior instructions; this system's policy is broader. Reported together, the
    two are indistinguishable.
    """
    if slices is None:
        print("\n  (no override annotation found — skipping the policy-scope split)")
        return []

    # Subset B repeats a few queries, so the annotation must be deduplicated or
    # the merge multiplies rows.
    annotation = slices.drop_duplicates(subset="query")[["query", "override_present"]]
    merged = frame.merge(annotation, on="query", how="left")
    assert len(merged) == len(frame), "merge changed the row count"

    malicious = merged[merged.label == "malicious"].copy()
    malicious["override_present"] = malicious["override_present"].fillna("unknown").astype(str)

    rows: List[Dict[str, object]] = []
    print("\n  Malicious rows by policy scope (fine-tuned):")
    for value, group in malicious.groupby("override_present"):
        detected = (
            score_malicious(model, tokenizer, group["query"].astype(str).tolist())
            >= SECURITY_THRESHOLD
        ).mean()
        rows.append({"override_present": value, "n": len(group), "detected": float(detected)})
        print(f"    override_present={value:8s} n={len(group):4d}  detected={detected:.4f}")
    return rows


def print_verdict(baseline: List[EvalResult], finetuned: List[EvalResult]) -> str:
    """Apply the pre-registered interpretation rule and return the verdict."""
    before = {r.dataset: r for r in baseline}

    print("\n" + "=" * 78)
    print("  BEFORE / AFTER")
    print("=" * 78)
    print(f"  {'dataset':<14} {'recall':>18} {'FPR':>18} {'accuracy':>18}")
    for after in finetuned:
        b = before[after.dataset]
        print(
            f"  {after.dataset:<14} "
            f"{b.recall:>8.4f}->{after.recall:<8.4f} "
            f"{b.fpr:>8.4f}->{after.fpr:<8.4f} "
            f"{b.accuracy:>8.4f}->{after.accuracy:<8.4f}"
        )

    # Recall alone is trivially maximised by predicting "malicious" for
    # everything, so a recall-only rule would call a collapsed classifier a
    # success — reporting victory for a model whose false-positive rate had
    # risen from 0.0000 toward 0.7333. The rule therefore requires a *balanced*
    # improvement: F1 up and accuracy not down.
    improved = {
        r.dataset: (r.f1 > before[r.dataset].f1 and r.accuracy >= before[r.dataset].accuracy)
        for r in finetuned
    }
    collapsed = [r.dataset for r in finetuned if r.recall > 0.99 and r.fpr > 0.5]
    if collapsed:
        print("\n  ⚠ NEAR-CONSTANT CLASSIFIER on: " + ", ".join(collapsed))
        print("    Recall ~1.0 with FPR > 0.5 means the model flags almost everything.")
        print("    This is a collapse, not a detection improvement.")
    held_out_improved = improved.get("held-out EN", False) or improved.get("held-out ID", False)

    print()
    if improved.get("Subset B") and held_out_improved:
        verdict = "adaptation_supported"
        print("  VERDICT: improves on Subset B AND on held-out data.")
        print("           The Indonesian adaptation claim holds.")
    elif improved.get("Subset B"):
        verdict = "style_matching"
        print("  VERDICT: improves on Subset B ONLY.")
        print("           This is style-matching against the generator's templates.")
        print("           Report it as such — the adaptation claim is not supported.")
    else:
        verdict = "no_improvement"
        print("  VERDICT: no recall improvement on Subset B. Report the negative result.")

    worst_fpr = max(r.fpr - before[r.dataset].fpr for r in finetuned)
    if worst_fpr > 0.01:
        print(
            f"\n  NOTE: false-positive rate rose by up to {worst_fpr:+.4f}. The off-the-shelf "
            "guard\n        never blocked a legitimate JDIH query; if that property is gone, say "
            "so —\n        for this system a false block is the more visible failure."
        )
    return verdict


def main() -> None:
    """CLI entry point."""
    global TRAIN_MAX_LENGTH

    parser = argparse.ArgumentParser(description="Fine-tune the IVM safety classifier locally")
    parser.add_argument("--data-dir", default="evals/data")
    parser.add_argument("--output", default="models/prompt_guard_id")
    parser.add_argument("--results", default="evals/data/results/prompt_guard_finetune.json")
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=256, help="Training truncation only; evaluation always uses the full 512.")
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument(
        "--train-embeddings",
        action="store_true",
        help=(
            "Also update the 251k-token embedding table (192.8M of 278.8M params). "
            "Off by default: it invites overfitting on a 2.5k-row set and needs "
            "~2.3 GB more VRAM for its gradients and optimizer moments."
        ),
    )
    args = parser.parse_args()

    import os

    token = os.getenv("HF_TOKEN") or None
    data_dir = Path(args.data_dir)

    TRAIN_MAX_LENGTH = args.max_length

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_df, eval_sets, slices = load_frames(data_dir)
    assert_held_out(train_df, eval_sets)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(
        "train.start",
        device=device,
        train_rows=len(train_df),
        transformers=__import__("transformers").__version__,
    )

    tokenizer, model = build_model(token, freeze_embeddings=not args.train_embeddings)
    model.to(device)

    # --- baseline, measured before any weight changes -----------------------
    print("\n" + "=" * 78)
    print("  BASELINE (off the shelf)")
    print("=" * 78)
    baseline = [evaluate(model, tokenizer, f, name, "off-the-shelf") for f, name in eval_sets]

    # --- training -----------------------------------------------------------
    from datasets import Dataset

    shuffled = train_df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    cut = int(len(shuffled) * args.val_fraction)
    val_df, fit_df = shuffled[:cut], shuffled[cut:]

    def to_dataset(frame: pd.DataFrame) -> "Dataset":
        dataset = Dataset.from_pandas(
            pd.DataFrame(
                {
                    "text": frame["query"].astype(str),
                    "labels": (frame["label"] == "malicious").astype(int),
                }
            ),
            preserve_index=False,
        )
        tokenized = dataset.map(
            lambda batch: tokenizer(batch["text"], truncation=True, max_length=TRAIN_MAX_LENGTH),
            batched=True,
            remove_columns=["text"],
        )
        return tokenized

    def compute_metrics(prediction) -> Dict[str, float]:
        predicted = prediction.predictions.argmax(-1)
        truth = prediction.label_ids
        p, r, f1, _ = precision_recall_fscore_support(
            truth, predicted, average="binary", zero_division=0
        )
        return {"accuracy": accuracy_score(truth, predicted), "precision": p, "recall": r, "f1": f1}

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
        metric_for_best_model="f1",
        logging_steps=25,
        # Mixed precision only on CUDA; fp16 on CPU is slower, not faster.
        fp16=(device == "cuda"),
        seed=args.seed,
        report_to="none",
        # Dynamic padding already keeps batches short (median row is 29 tokens),
        # so grouping by length buys little and costs reproducibility.
        dataloader_num_workers=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=to_dataset(fit_df),
        eval_dataset=to_dataset(val_df),
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )
    print("\n" + "=" * 78)
    print(f"  TRAINING  ({len(fit_df)} rows, {args.epochs} epochs, device={device})")
    print("=" * 78)
    trainer.train()

    # --- fine-tuned evaluation, same function and thresholds ----------------
    print("\n" + "=" * 78)
    print("  FINE-TUNED")
    print("=" * 78)
    finetuned = [evaluate(model, tokenizer, f, name, "fine-tuned") for f, name in eval_sets]

    subset_b = eval_sets[0][0]
    per_subtype = report_per_subtype(model, tokenizer, subset_b)
    policy_scope = report_policy_scope(model, tokenizer, subset_b, slices)
    verdict = print_verdict(baseline, finetuned)

    # --- save ---------------------------------------------------------------
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output))
    tokenizer.save_pretrained(str(output))

    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    if not (config["id2label"]["0"] == "BENIGN" and config["id2label"]["1"] == "MALICIOUS"):
        raise AssertionError("LABEL ORDER INVERTED — do not deploy this checkpoint")
    logger.info("train.label_order_verified", id2label=config["id2label"])

    results_path = Path(args.results)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(
        json.dumps(
            {
                "base_model": BASE_MODEL,
                "transformers_version": __import__("transformers").__version__,
                "torch_version": torch.__version__,
                "device": device,
                "threshold": SECURITY_THRESHOLD,
                "seed": args.seed,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "train_max_length": TRAIN_MAX_LENGTH,
                "eval_max_length": EVAL_MAX_LENGTH,
                "learning_rate": args.learning_rate,
                "train_rows": len(fit_df),
                "embeddings_frozen": not args.train_embeddings,
                "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "val_rows": len(val_df),
                "verdict": verdict,
                "baseline": [vars(r) for r in baseline],
                "finetuned": [vars(r) for r in finetuned],
                "per_subtype": per_subtype,
                "policy_scope": policy_scope,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"\n  model   -> {output}")
    print(f"  results -> {results_path}")
    print("\n  To serve it: set INFINITY_PROMPT_GUARD_MODEL to this path (mounted into the")
    print("  container) or push it to the Hub, then recreate the prompt-guard service.")


if __name__ == "__main__":
    main()
