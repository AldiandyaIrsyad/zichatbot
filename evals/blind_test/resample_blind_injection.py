"""Resample blind-injection sidecars for subsets B and C.

Problem: The original 20%-of-5/5-unanimous rule produced too few items
for small subsets (B=3, C=4). This script resamples with a minimum of 20
items per subset to ensure meaningful concordance verification.

Strategy:
  - Subset C: has panel_yes/panel_size columns → sample from 5/5-unanimous items.
    If fewer than 20 unanimous items exist, take all of them.
  - Subset B: no panel columns in final CSV → all 160 rows passed ≥4/5 panel,
    so sample 20 from all accepted rows.

Usage:
    python -m evals.blind_test.resample_blind_injection

Outputs:
    evals/data/subset_b_blind_injection.csv  (overwritten)
    evals/data/subset_c_blind_injection.csv  (overwritten)
"""

import csv
import random
from pathlib import Path

SEED = 42
MIN_TARGET = 20

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def resample_subset_b():
    """Resample subset B blind injection: 20 from all accepted rows."""
    src = DATA_DIR / "subset_b.csv"
    dst = DATA_DIR / "subset_b_blind_injection.csv"

    with src.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    rng = random.Random(SEED)
    n = min(MIN_TARGET, len(rows))
    selected = rng.sample(rows, n)

    with dst.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected)

    print(f"Subset B: {len(rows)} total → {n} blind-injected → {dst.name}")


def resample_subset_c():
    """Resample subset C blind injection: up to 20 from 5/5-unanimous items."""
    src = DATA_DIR / "subset_c.csv"
    dst = DATA_DIR / "subset_c_blind_injection.csv"

    with src.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    # Filter to 5/5 unanimous
    unanimous = [r for r in rows if r.get("panel_yes") == "5" and r.get("panel_size") == "5"]

    rng = random.Random(SEED)
    n = min(MIN_TARGET, len(unanimous))
    selected = rng.sample(unanimous, n)

    with dst.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected)

    print(f"Subset C: {len(rows)} total, {len(unanimous)} unanimous → {n} blind-injected → {dst.name}")


if __name__ == "__main__":
    resample_subset_b()
    resample_subset_c()
    print("\nDone. Review the new sidecars with evals/blind_test/blind_injection_review.html")
