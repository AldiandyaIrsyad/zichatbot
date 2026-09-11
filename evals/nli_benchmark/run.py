"""NLI model benchmark on IndoNLI (test_lay + test_expert).

Scores three NLI backends against the external Indonesian NLI benchmark and
reports each split separately:

- ``indo_roberta`` — StevenLimcorn/indo-roberta-indonli (3-way, current default)
- ``mmbert`` — the IndoNLI fine-tune of jhu-clsp/mmBERT-small (3-way)
- ``zeroshot`` — MoritzLaurer/bge-m3-zeroshot-v2.0-c (binary entail/non-entail)

Models load in-process (no serving containers needed), so the benchmark runs on
a machine with ``torch`` + ``transformers`` (GPU recommended; the 0.6B zero-shot
baseline is slow on CPU). 3-way models report macro P/R/F1 + accuracy + Kappa +
confusion; the binary baseline reports entailment-vs-non-entailment binary
metrics.

Usage:
    python -m evals.nli_benchmark.run --models zeroshot,mmbert,indo_roberta --device cuda
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Dict, List, Tuple

import structlog

from app.guardrails.nli import LabelSpace, NLIModelKind, NLIModelSpec, build_nli_model
from evals._shared.dataset import IndoNLIRow, load_indonli
from evals._shared.metrics import (
    BinaryMetrics,
    MultiClassMetrics,
    compute_binary_metrics,
    compute_multiclass_metrics,
)

logger = structlog.get_logger(__name__)

NLI_LABELS = ["entailment", "neutral", "contradiction"]

# model key → (model_id, label_space, max_total_tokens)
_MODEL_DEFAULTS: Dict[str, Tuple[str, LabelSpace, int]] = {
    "zeroshot": ("MoritzLaurer/bge-m3-zeroshot-v2.0-c", LabelSpace.BINARY, 512),
    "mmbert": ("models/mmbert_nli_id", LabelSpace.THREE_WAY, 512),
    "indo_roberta": ("StevenLimcorn/indo-roberta-indonli", LabelSpace.THREE_WAY, 500),
}


def _build_local(model_key: str, args: argparse.Namespace):
    model_id = getattr(args, f"{model_key}_model")
    default_id, label_space, max_tokens = _MODEL_DEFAULTS[model_key]
    if not model_id:
        model_id = default_id
    spec = NLIModelSpec(
        kind=NLIModelKind.INDO_ROBERTA,  # kind is irrelevant for the local backend
        model_id=model_id,
        label_space=label_space,
        device=args.device,
        max_total_tokens=max_tokens,
        max_hypothesis_tokens=150,
    )
    return build_nli_model(spec, backend="local")


def _three_way_metrics(predictions: List[str], truths: List[str]) -> dict:
    m: MultiClassMetrics = compute_multiclass_metrics(predictions, truths, NLI_LABELS)
    return {
        "accuracy": m.accuracy,
        "macro_precision": m.macro_precision,
        "macro_recall": m.macro_recall,
        "macro_f1": m.macro_f1,
        "cohen_kappa": m.cohen_kappa,
        "per_class": {k: {"precision": v[0], "recall": v[1], "f1": v[2]} for k, v in m.per_class.items()},
        "confusion": m.confusion,
        "labels": m.labels,
    }


def _binary_metrics(predictions: List[str], truths: List[str]) -> dict:
    """Collapse to entailment vs non-entailment for the binary baseline."""
    pred_bool = [p == "entailment" for p in predictions]
    truth_bool = [t == "entailment" for t in truths]
    m: BinaryMetrics = compute_binary_metrics(pred_bool, truth_bool)
    return {
        "accuracy": m.accuracy,
        "precision": m.precision,
        "recall": m.recall,
        "f1": m.f1,
        "fpr": m.fpr,
        "n": m.total,
    }


async def score_model(client, rows: List[IndoNLIRow]) -> List[str]:
    """Run ``check`` on every row, returning the predicted labels in order."""
    predictions: List[str] = []
    for i, row in enumerate(rows, 1):
        result = await client.check(premise=row.premise, hypothesis=row.hypothesis)
        predictions.append(result.label)
        if i % 200 == 0:
            print(f"    {i}/{len(rows)}", flush=True)
    return predictions


async def main() -> None:
    parser = argparse.ArgumentParser(description="NLI model benchmark on IndoNLI")
    parser.add_argument(
        "--models", default="zeroshot,mmbert,indo_roberta",
        help="Comma-separated model keys: zeroshot,mmbert,indo_roberta",
    )
    parser.add_argument("--splits", default="test_lay,test_expert",
                        help="Comma-separated IndoNLI splits to score")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=0, help="Cap rows per split (0 = all)")
    parser.add_argument("--output", default="evals/data/results/nli_benchmark.json")
    parser.add_argument("--data-dir", default="evals/data")
    parser.add_argument("--mmbert-model", default="")
    parser.add_argument("--zeroshot-model", default="")
    parser.add_argument("--indo_roberta-model", default="")
    args = parser.parse_args()

    model_keys = [k.strip() for k in args.models.split(",") if k.strip()]
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    summary: dict = {"models": {}, "splits": {}}
    for model_key in model_keys:
        print(f"\n=== {model_key} ===")
        client = _build_local(model_key, args)
        model_summary: dict = {}
        try:
            for split in splits:
                rows = load_indonli(split, data_dir=args.data_dir)
                if args.limit:
                    rows = rows[: args.limit]
                print(f"  split={split} n={len(rows)}")
                predictions = await score_model(client, rows)
                truths = [r.label for r in rows]

                label_space = _MODEL_DEFAULTS[model_key][1]
                if label_space == LabelSpace.BINARY:
                    model_summary[split] = _binary_metrics(predictions, truths)
                else:
                    model_summary[split] = _three_way_metrics(predictions, truths)

                # Per-split CSV of predictions for spot-checking / analysis.
                csv_path = (
                    Path(args.output).parent / "nli_benchmark"
                    / f"{model_key}_{split}.csv"
                )
                csv_path.parent.mkdir(parents=True, exist_ok=True)
                with csv_path.open("w", encoding="utf-8") as fh:
                    fh.write("pair_id,premise,hypothesis,true,pred\n")
                    for row, pred in zip(rows, predictions):
                        fh.write(
                            f"{row.pair_id},{json.dumps(row.premise, ensure_ascii=False)},"
                            f"{json.dumps(row.hypothesis, ensure_ascii=False)},{row.label},{pred}\n"
                        )
        finally:
            await client.close()
        summary["models"][model_key] = model_summary

    # Per-split comparison table.
    for split in splits:
        summary["splits"][split] = {
            model_key: summary["models"][model_key].get(split)
            for model_key in model_keys
        }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"\n  results -> {out}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
