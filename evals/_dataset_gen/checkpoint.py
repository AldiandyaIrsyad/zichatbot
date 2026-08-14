"""Crash-safe incremental CSV writing for dataset generation.

Writes each row to the output CSV the moment the panel accepts it, and
flushes, so the file on disk is always a valid, complete-as-of-now dataset.
Combined with ``--resume``, an interrupted run continues from where it stopped
instead of starting over: the builders reconstruct their per-category and
per-document counters from the rows already on disk (every counter they keep
is derivable from the CSV's own columns).

Not covered by resume: the blind-injection sidecar
(``concordance.BlindInjectionTracker``) only sees candidates from the session
that writes it, so a resumed run produces a sidecar drawn from the resuming
session's candidates alone. That file is a quality-control sample rather than
part of the dataset, so this is a documented limitation, not a correctness
problem — but it is why ``resume_rows`` logs a warning.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, TextIO

import structlog

logger = structlog.get_logger(__name__)


def resume_rows(output_path: str, fieldnames: Iterable[str]) -> List[Dict[str, str]]:
    """Read back rows already written by an interrupted run.

    A file whose header doesn't match ``fieldnames`` is treated as unusable
    rather than silently merged. Returns an empty list if the file is missing,
    empty, or written under a different schema.
    """
    path = Path(output_path)
    if not path.exists() or path.stat().st_size == 0:
        return []

    expected = list(fieldnames)
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != expected:
            logger.warning(
                "datagen.resume.schema_mismatch",
                path=str(path),
                found=reader.fieldnames,
                expected=expected,
                detail="ignoring existing file; it was written under a different schema",
            )
            return []
        rows = [dict(r) for r in reader]

    if rows:
        logger.warning(
            "datagen.resume.loaded",
            path=str(path),
            rows=len(rows),
            detail=(
                "resuming an earlier run; the blind-injection sidecar will only "
                "reflect candidates seen in this session"
            ),
        )
    return rows


class IncrementalCSVWriter:
    """Append-as-you-go CSV writer that flushes after every row.

    Used as a context manager::

        with IncrementalCSVWriter(path, fieldnames, resume=True) as w:
            w.append(row)

    When ``resume`` is True, appends to an existing file (header already
    present) instead of truncating it. When False the file is overwritten — a
    fresh run should not silently inherit rows generated under an older panel
    or prompt.
    """

    def __init__(self, output_path: str, fieldnames: Iterable[str], resume: bool = False) -> None:
        self._path = Path(output_path)
        self._fieldnames = list(fieldnames)
        self._resume = resume
        self._handle: Optional[TextIO] = None
        self._writer: Optional[csv.DictWriter] = None
        self.rows_written = 0

    def __enter__(self) -> "IncrementalCSVWriter":
        self._path.parent.mkdir(parents=True, exist_ok=True)
        appending = self._resume and self._path.exists() and self._path.stat().st_size > 0
        self._handle = self._path.open("a" if appending else "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(
            self._handle, fieldnames=self._fieldnames, extrasaction="ignore"
        )
        if not appending:
            self._writer.writeheader()
            self._handle.flush()
        return self

    def __exit__(self, *exc: Any) -> None:
        if self._handle is not None:
            self._handle.flush()
            self._handle.close()
            self._handle = None
            self._writer = None

    def append(self, row: Dict[str, Any]) -> None:
        """Write one row and flush it to disk immediately.

        Flushing per row is what makes an interrupted run recoverable; the cost
        is irrelevant next to the panel calls.
        """
        if self._writer is None or self._handle is None:
            raise RuntimeError("IncrementalCSVWriter used outside its context manager")
        self._writer.writerow(row)
        self._handle.flush()
        self.rows_written += 1
