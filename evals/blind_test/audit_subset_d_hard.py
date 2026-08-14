"""Audit Subset D-Hard and sibling CSVs for structural corruption.

DIAGNOSE-ONLY: no files are modified. Reports:
  1. File metadata (size, line count, encoding)
  2. Line-ending analysis (CRLF vs LF)
  3. Null byte scan
  4. CSV parse with per-row field counts (flags rows != 8 columns)
  5. Citation-marker leak detection (sentence_text containing "")* or *(Supported)
  6. question_id sequence analysis (missing IDs, prefix transitions)
  7. Sibling comparison across the d-family

Usage:
    python -m evals.blind_test.audit_subset_d_hard
"""

from __future__ import annotations

import csv
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Python's csv module needs a larger field size limit for large cells
csv.field_size_limit(sys.maxsize)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

D_FAMILY_FILES = [
    "subset_d.csv",
    "subset_d_blind_injection.csv",
    "subset_d_hard.csv",
    "subset_d_hard_blind_injection.csv",
]

EXPECTED_FIELDS = [
    "question_id",
    "question",
    "full_response",
    "sentence_id",
    "sentence_text",
    "retrieved_context",
    "label",
    "verifier_note",
]

# Patterns that indicate citation-marker leakage into sentence_text
CITATION_LEAK_PATTERNS = [
    re.compile(r'"\)\*'),
    re.compile(r'\*\(Supported:'),
    re.compile(r'Evidence:""'),
    re.compile(r'DocID:'),
    re.compile(r'; Page \d+;'),
]


def file_metadata(path: Path) -> dict:
    """Collect basic file metadata."""
    stat = path.stat()
    with path.open("rb") as f:
        raw = f.read()
    line_count = raw.count(b"\n")
    crlf_count = raw.count(b"\r\n")
    lf_only = line_count - crlf_count
    null_count = raw.count(b"\x00")
    return {
        "path": str(path),
        "size_bytes": stat.st_size,
        "physical_lines": line_count + (1 if raw and not raw.endswith(b"\n") else 0),
        "crlf_lines": crlf_count,
        "lf_only_lines": lf_only,
        "null_bytes": null_count,
    }


def audit_csv_parse(path: Path) -> dict:
    """Parse CSV and report field-count anomalies + citation leaks."""
    rows = []
    field_count_anomalies = []
    citation_leak_rows = []
    header = None

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return {"error": "empty file"}

        for row_idx, row in enumerate(reader, start=2):
            rows.append(row)
            if len(row) != len(EXPECTED_FIELDS):
                field_count_anomalies.append({
                    "csv_row": row_idx,
                    "field_count": len(row),
                    "first_50_chars": (row[0][:50] if row else ""),
                })

            # Check sentence_text (col 4, index 4) for citation leak
            if len(row) >= 5:
                sentence_text = row[4]
                for pattern in CITATION_LEAK_PATTERNS:
                    if pattern.search(sentence_text):
                        citation_leak_rows.append({
                            "csv_row": row_idx,
                            "question_id": row[0] if row else "",
                            "sentence_id": row[3] if len(row) > 3 else "",
                            "sentence_text_preview": sentence_text[:120],
                            "matched_pattern": pattern.pattern,
                        })
                        break

    return {
        "header": header,
        "header_matches": header == EXPECTED_FIELDS,
        "total_data_rows": len(rows),
        "field_count_anomalies": field_count_anomalies,
        "citation_leak_rows": citation_leak_rows,
    }


def audit_question_id_sequence(path: Path) -> dict:
    """Analyze question_id prefixes and sequence continuity."""
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        ids = [row.get("question_id", "") for row in reader]

    # Group by prefix (e.g., qh-reweighted-, qh-detail-, q-)
    prefix_groups: dict[str, list[str]] = defaultdict(list)
    for qid in ids:
        # Extract prefix: everything up to the last dash-number
        match = re.match(r"^(.+?)-(\d+)$", qid)
        if match:
            prefix_groups[match.group(1)].append(qid)
        else:
            prefix_groups[qid].append(qid)

    # Check for missing IDs within each prefix group
    missing_ids = {}
    for prefix, group_ids in prefix_groups.items():
        numbers = sorted(int(re.search(r"(\d+)$", qid).group(1)) for qid in group_ids)
        if numbers:
            expected = set(range(numbers[0], numbers[-1] + 1))
            missing = sorted(expected - set(numbers))
            if missing:
                missing_ids[prefix] = missing

    return {
        "total_ids": len(ids),
        "prefix_groups": {k: len(v) for k, v in prefix_groups.items()},
        "prefix_order": list(prefix_groups.keys()),
        "missing_ids_by_prefix": missing_ids,
    }


def print_section(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def main() -> None:
    print_section("SUBSET D-FAMILY CSV AUDIT (DIAGNOSE ONLY)")

    for filename in D_FAMILY_FILES:
        path = DATA_DIR / filename
        if not path.exists():
            print(f"\n[SKIP] {filename} — not found")
            continue

        print_section(f"FILE: {filename}")

        # 1. File metadata
        meta = file_metadata(path)
        print(f"\n  [1] FILE METADATA")
        print(f"      Size:           {meta['size_bytes']:,} bytes")
        print(f"      Physical lines: {meta['physical_lines']:,}")
        print(f"      CRLF lines:     {meta['crlf_lines']:,}")
        print(f"      LF-only lines:  {meta['lf_only_lines']:,}")
        print(f"      Null bytes:     {meta['null_bytes']}")

        # 2. CSV parse + field count + citation leak
        parse_result = audit_csv_parse(path)
        print(f"\n  [2] CSV PARSE")
        print(f"      Header: {parse_result.get('header', 'N/A')}")
        print(f"      Header matches expected: {parse_result.get('header_matches', False)}")
        print(f"      Total data rows: {parse_result.get('total_data_rows', 0)}")

        anomalies = parse_result.get("field_count_anomalies", [])
        print(f"      Field-count anomalies: {len(anomalies)}")
        for a in anomalies[:10]:
            print(f"        Row {a['csv_row']}: {a['field_count']} fields — {a['first_50_chars']!r}")
        if len(anomalies) > 10:
            print(f"        ... and {len(anomalies) - 10} more")

        leaks = parse_result.get("citation_leak_rows", [])
        print(f"\n  [3] CITATION-MARKER LEAK (sentence_text corruption)")
        print(f"      Corrupted rows: {len(leaks)}")
        for leak in leaks:
            print(f"        Row {leak['csv_row']}: qid={leak['question_id']} sid={leak['sentence_id']}")
            print(f"          sentence_text: {leak['sentence_text_preview']!r}")
            print(f"          matched: {leak['matched_pattern']}")

        # 3. Question ID sequence
        seq = audit_question_id_sequence(path)
        print(f"\n  [4] QUESTION_ID SEQUENCE")
        print(f"      Total IDs: {seq['total_ids']}")
        print(f"      Prefix groups (in file order):")
        for prefix, count in seq["prefix_groups"].items():
            print(f"        {prefix}: {count} rows")
        missing = seq.get("missing_ids_by_prefix", {})
        if missing:
            print(f"      MISSING IDs:")
            for prefix, ids in missing.items():
                print(f"        {prefix}: missing {ids}")
        else:
            print(f"      No missing IDs detected.")

    print_section("AUDIT COMPLETE")


if __name__ == "__main__":
    main()
