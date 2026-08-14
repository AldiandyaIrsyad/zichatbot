"""Provenance sidecars for generated evaluation subsets.

Every generated subset gets a ``<output>.meta.json`` recording which panel,
generator, and settings produced it, so "all subsets were validated by the
same ≥4/5 panel" is a checkable claim rather than one asserted from memory.

The sidecar is written next to the CSV and is cheap enough to produce on every
run, including aborted ones.
"""

from __future__ import annotations

import dataclasses
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import structlog

from evals._dataset_gen.config import DatasetGenSettings

logger = structlog.get_logger(__name__)


def _git_sha() -> Optional[str]:
    """Return the current git commit SHA, or None outside a repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None if result.returncode == 0 else None


def write_provenance(
    output_path: str,
    subset: str,
    settings: DatasetGenSettings,
    row_count: int,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """Write a ``.meta.json`` sidecar describing how a subset was generated.

    ``settings`` is recorded verbatim, so a sidecar reflects the live ``.env``
    rather than the code defaults.
    """
    meta: Dict[str, Any] = {
        "subset": subset,
        "dataset_version": "v2",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "row_count": row_count,
        "generator": {
            "model": settings.generator_model,
            "temperature": settings.generator_temperature,
        },
        "panel": {
            "models": settings.panel_model_list,
            "size": len(settings.panel_model_list),
            "acceptance_threshold": settings.acceptance_threshold,
            "temperature": settings.panel_temperature,
            "strict_principal_rows": True,
            "infrastructure_errors_are_votes": False,
        },
        "output_csv": str(Path(output_path).name),
        "cost_ledger": settings.cost_ledger_path,
        "budget_usd": settings.budget_usd,
    }
    if extra:
        meta.update(extra)

    sidecar = Path(output_path).with_suffix(".meta.json")
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    logger.info(
        "datagen.provenance_written",
        path=str(sidecar),
        subset=subset,
        panel_size=len(settings.panel_model_list),
        rows=row_count,
    )
    return sidecar


def append_panel_vote_record(
    output_path: str,
    subset: str,
    candidate: Dict[str, Any],
    verdict: Any,
    admitted: bool,
) -> Path:
    """Append a durable candidate-to-panel-decision record before row commit.

    Unlike the sampled blind-injection sidecar, this JSONL is append-only and
    records every evaluated candidate across resume sessions. It intentionally
    stores rejected/contested candidates as well as admitted rows.
    """
    path = Path(output_path).with_suffix(".panel_votes.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "subset": subset,
        "candidate": candidate,
        "admitted": admitted,
        "verdict": dataclasses.asdict(verdict),
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
    return path
