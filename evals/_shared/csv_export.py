"""Raw per-row result export for evaluation scripts.

Every exp*_*.run script prints only an aggregate report to stdout, so no
per-row data survives the process exiting — which leaves every per-row claim
("the SLM caught this attack", "RAM found this sentence unsupported")
unverifiable without rerunning the whole experiment. ``write_results_csv``
gives every script a one-line way to persist its per-row (input, ground truth,
prediction, scores) data for independent inspection.
"""

from __future__ import annotations

import csv
import os
from typing import Any, Dict, List


def write_results_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    """Write per-row evaluation results to a CSV file.

    Parent directories of ``path`` are created if needed. The header is taken
    from the first row's keys; every row must have the same keys.
    """
    if not rows:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Raw results written to {path} ({len(rows)} rows)")
