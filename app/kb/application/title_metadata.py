"""Derive document metadata from a JDIH document title.

UPI decree titles carry their issuing year inside the document number that
prefixes the subject: ``"1313-UN40-KM.02.02-2026 - Peserta Program …"`` or
``"003 Tahun 2022 - Kelompok Kemampuan Ekonomi …"``. That is the only date
signal available offline, so it backs ``released_date`` when no date was
supplied at upload time.

Pure stdlib — usable from the app, a backfill script, or a test.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Optional

# The subject follows the document number after the first " - " separator.
# Parsing is confined to that prefix on purpose: subjects routinely mention
# *academic* years ("Semester Ganjil Tahun Akademik 2025/2026") that are not
# the issuing year and would otherwise win.
_NUMBER_BLOCK_SEP = " - "

_YEAR = re.compile(r"\b((?:19|20)\d{2})\b")

# Nothing in this corpus predates the university's digital records, and a year
# beyond next year is a parse error rather than a real release.
_MIN_YEAR = 1950


def parse_released_year(title: str, max_year: Optional[int] = None) -> Optional[int]:
    """Extract the issuing year from a document title, or None.

    Reads only the document-number prefix (see ``_NUMBER_BLOCK_SEP``). When the
    prefix holds several years — ``"41 Tahun 2021"`` inside a longer number —
    the last one wins, since the year is conventionally the final component of
    an Indonesian decree number.
    """
    if not title or not title.strip():
        return None

    prefix = title.split(_NUMBER_BLOCK_SEP, 1)[0]
    matches = _YEAR.findall(prefix)
    if not matches:
        return None

    year = int(matches[-1])
    ceiling = max_year if max_year is not None else datetime.now().year + 1
    if year < _MIN_YEAR or year > ceiling:
        return None
    return year


def parse_released_date(title: str, max_year: Optional[int] = None) -> Optional[datetime]:
    """Year-precision release date derived from ``title``, or None.

    The title carries no month or day, so the result is 1 January of the parsed
    year. That is precise enough for recency ranking
    (``kb/application/retrieval_strategies.py`` scores by age in years) but must
    not be presented to a user as the actual date of issue.

    Returned **timezone-aware in UTC**. ``pdf_documents.released_date`` is a
    ``timestamptz``: a naive midnight is interpreted in the session's zone, so
    on a UTC+7 host it lands at 17:00 on 31 December of the *previous* year and
    every document reads as a year older than it is — which the answer prompt
    then repeats back, since it is told to weigh each source's date.
    """
    year = parse_released_year(title, max_year=max_year)
    return datetime(year, 1, 1, tzinfo=timezone.utc) if year else None
