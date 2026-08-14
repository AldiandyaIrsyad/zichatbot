#!/usr/bin/env python3
"""Bulk-upload PDFs from scrapers/data2.csv into the running KB app.

Reads the scraped JDIH corpus (default: scrapers/data2.csv + scrapers/dataset/),
uploads each PDF via the real /api/admin/pdfs endpoint, then polls ingestion
status until it completes, retrying failed ingestions a bounded number of
times via /reingest. Progress is persisted to a JSON state file after every
document, so the script is safely resumable: Ctrl-C or a crash loses at most
the document currently in flight, and re-running skips anything already
"completed" (never re-uploads — this app has no server-side dedup, so
re-uploading the same PDF creates a duplicate).

Concurrency is deliberately conservative by default (3 documents in flight
at once): the box has a single 8GB GPU already shared by Infinity (reranker
+ prompt-guard + NLI) and the in-process BGE-M3 embedder, a 15-connection
DB pool, and both Unstructured Cloud and OpenRouter have real rate limits.
The server already retries transient failures in its own external calls
(Unstructured/VLM/embedding) — this script's retry layer is one level up:
whole-document reingest after a document ends up "failed".

Usage:
    # Preflight: check the server is up and preview what would happen
    .venv/bin/python scripts/bulk_upload_pdfs.py --dry-run

    # Try a small batch first
    .venv/bin/python scripts/bulk_upload_pdfs.py --limit 5

    # The real run (resumable — safe to re-run after Ctrl-C or a crash)
    .venv/bin/python scripts/bulk_upload_pdfs.py

    # Tune concurrency up if your Unstructured Cloud / OpenRouter tier allows it
    .venv/bin/python scripts/bulk_upload_pdfs.py --concurrency 6
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import re
import signal
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import httpx

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = REPO_ROOT / "scrapers" / "data2.csv"
DEFAULT_PDF_ROOT = REPO_ROOT / "scrapers"
DEFAULT_STATE_FILE = REPO_ROOT / "scripts" / "upload_state.json"
DEFAULT_API_BASE = "http://localhost:8000"

# How often to poll ingestion-status for a single in-flight document.
POLL_INTERVAL_SEC = 5.0
# Give up waiting on one document after this long (Unstructured Cloud's own
# job timeout is 15 min server-side; this adds buffer for VLM + embedding).
MAX_INGEST_WAIT_SEC = 25 * 60
# How many times to reingest a document that ended up "failed" before giving
# up on it permanently.
DEFAULT_MAX_REINGEST_ATTEMPTS = 2
# Retries for the upload POST itself (talking to the local server — should
# rarely be needed, but the DB pool or disk can transiently be busy).
UPLOAD_RETRY_ATTEMPTS = 3


def sanitize_title(text: str) -> str:
    """The server builds a filesystem path as f"{prefix}_{title}_{filename}"
    with no sanitization — a raw "/" in the title (common in these documents'
    Code field, e.g. "1313/UN40/KM.02.02/2026") would be interpreted as a
    path separator and break the upload. Strip anything filesystem-unsafe."""
    text = re.sub(r"[/\\]", "-", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:200]


@dataclass
class DocState:
    pdf_path: str
    code: str
    title: str
    status: str = "pending"  # pending -> uploaded -> completed | failed
    doc_id: Optional[str] = None
    attempts: int = 0
    last_error: Optional[str] = None
    updated_at: float = field(default_factory=time.time)


class StateStore:
    """JSON-backed state, keyed by pdf_path (unique per row; Code has dupes
    in this corpus). Written after every document so progress survives a
    crash or Ctrl-C."""

    def __init__(self, path: Path):
        self.path = path
        self._data: dict[str, dict] = {}
        if path.exists():
            self._data = json.loads(path.read_text())

    def get(self, pdf_path: str) -> Optional[DocState]:
        raw = self._data.get(pdf_path)
        return DocState(**raw) if raw else None

    def save(self, state: DocState) -> None:
        state.updated_at = time.time()
        self._data[state.pdf_path] = asdict(state)
        self._flush()

    def _flush(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._data, indent=2, ensure_ascii=False))
        tmp.replace(self.path)

    def summary(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for row in self._data.values():
            counts[row["status"]] = counts.get(row["status"], 0) + 1
        return counts


def load_rows(csv_path: Path, pdf_root: Path) -> list[DocState]:
    rows: list[DocState] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            pdf_path = r["pdf_path"]
            full_path = pdf_root / pdf_path
            if not full_path.is_file():
                print(f"  WARNING: missing file, skipping: {pdf_path}", file=sys.stderr)
                continue
            title = sanitize_title(f"{r['Code']} - {r['Title']}")
            rows.append(DocState(pdf_path=pdf_path, code=r["Code"], title=title))
    return rows


async def upload_one(
    client: httpx.AsyncClient,
    pdf_root: Path,
    state: DocState,
    description: str,
) -> str:
    """POST the file, return the new doc_id. Retries transient local-server
    errors; reopens the file fresh on every attempt (a retry after a failed
    send can't reuse an already-consumed file stream)."""
    full_path = pdf_root / state.pdf_path
    last_exc: Optional[Exception] = None
    for attempt in range(1, UPLOAD_RETRY_ATTEMPTS + 1):
        try:
            with full_path.open("rb") as fh:
                resp = await client.post(
                    "/api/admin/pdfs",
                    data={"title": state.title, "description": description},
                    files={"file": (full_path.name, fh, "application/pdf")},
                )
            resp.raise_for_status()
            return resp.json()["id"]
        except Exception as exc:  # noqa: BLE001 - deliberately broad, this is a CLI retry loop
            last_exc = exc
            if attempt < UPLOAD_RETRY_ATTEMPTS:
                wait = 2 ** attempt
                print(f"  upload attempt {attempt} failed for {state.pdf_path}: {exc} — retrying in {wait}s")
                await asyncio.sleep(wait)
    raise RuntimeError(f"upload failed after {UPLOAD_RETRY_ATTEMPTS} attempts: {last_exc}")


async def poll_until_done(client: httpx.AsyncClient, doc_id: str) -> tuple[str, Optional[str]]:
    """Poll /ingestion-status until completed/failed or MAX_INGEST_WAIT_SEC
    elapses. Returns (status, error_message)."""
    deadline = time.monotonic() + MAX_INGEST_WAIT_SEC
    while time.monotonic() < deadline:
        resp = await client.get(f"/api/admin/pdfs/{doc_id}/ingestion-status")
        if resp.status_code == 200:
            data = resp.json()
            status = data.get("status")
            if status in ("completed", "failed"):
                return status, data.get("error_message")
        await asyncio.sleep(POLL_INTERVAL_SEC)
    return "timeout", f"ingestion did not finish within {MAX_INGEST_WAIT_SEC}s"


async def reingest(client: httpx.AsyncClient, doc_id: str) -> None:
    resp = await client.post(f"/api/admin/pdfs/{doc_id}/reingest")
    resp.raise_for_status()


async def process_one(
    client: httpx.AsyncClient,
    pdf_root: Path,
    store: StateStore,
    state: DocState,
    max_reingest_attempts: int,
    index: int,
    total: int,
) -> None:
    prefix = f"[{index}/{total}]"
    try:
        if state.status == "pending":
            print(f"{prefix} uploading: {state.pdf_path}")
            description = f"Sourced from JDIH UPI. Code: {state.code}"
            state.doc_id = await upload_one(client, pdf_root, state, description)
            state.status = "uploaded"
            store.save(state)

        # status == "uploaded" here whether we just uploaded it or we're
        # resuming a previous run that uploaded but never saw completion.
        while True:
            assert state.doc_id is not None
            print(f"{prefix} waiting for ingestion: {state.pdf_path} (doc_id={state.doc_id})")
            status, error = await poll_until_done(client, state.doc_id)

            if status == "completed":
                state.status = "completed"
                state.last_error = None
                store.save(state)
                print(f"{prefix} DONE: {state.pdf_path}")
                return

            state.attempts += 1
            state.last_error = error
            if state.attempts > max_reingest_attempts:
                state.status = "failed"
                store.save(state)
                print(f"{prefix} FAILED (giving up after {state.attempts} attempts): {state.pdf_path}: {error}")
                return

            print(f"{prefix} status={status} error={error!r} — reingesting (attempt {state.attempts}/{max_reingest_attempts})")
            store.save(state)
            await asyncio.sleep(min(30, 2 ** state.attempts))
            await reingest(client, state.doc_id)

    except Exception as exc:  # noqa: BLE001
        state.status = "failed"
        state.last_error = str(exc)
        store.save(state)
        print(f"{prefix} FAILED (exception): {state.pdf_path}: {exc}")


async def run(args: argparse.Namespace) -> None:
    store = StateStore(args.state_file)
    rows = load_rows(args.csv, args.pdf_root)
    if args.limit:
        rows = rows[: args.limit]

    # Resume: fold in any prior state for each row so completed/failed docs
    # are skipped and in-flight ("uploaded") docs resume polling instead of
    # re-uploading.
    for i, row in enumerate(rows):
        prior = store.get(row.pdf_path)
        if prior:
            rows[i] = prior

    todo = [r for r in rows if r.status not in ("completed",)]
    already_done = len(rows) - len(todo)
    print(f"Loaded {len(rows)} documents from {args.csv} ({already_done} already completed, {len(todo)} to process)")

    if args.dry_run:
        print("--dry-run: not uploading anything. Sample of what would run:")
        for r in todo[:10]:
            print(f"  [{r.status}] {r.title}  <-  {r.pdf_path}")
        if len(todo) > 10:
            print(f"  ... and {len(todo) - 10} more")
        return

    if not todo:
        print("Nothing to do.")
        return

    async with httpx.AsyncClient(
        base_url=args.api_base, timeout=httpx.Timeout(120.0, connect=10.0)
    ) as client:
        try:
            resp = await client.get("/api/admin/pdfs")
            resp.raise_for_status()
        except Exception as exc:
            print(f"ERROR: can't reach the API at {args.api_base}: {exc}", file=sys.stderr)
            print("Is the server running? e.g. .venv/bin/python -m uvicorn app.main:fastapi_app", file=sys.stderr)
            sys.exit(1)

        sem = asyncio.Semaphore(args.concurrency)

        async def bound(i: int, r: DocState) -> None:
            async with sem:
                await process_one(client, args.pdf_root, store, r, args.max_reingest_attempts, i, len(todo))

        tasks = [asyncio.create_task(bound(i + 1, r)) for i, r in enumerate(todo)]

        # Ctrl-C: let in-flight tasks finish their current step (state is
        # saved after every step already) rather than hard-killing mid-upload.
        stop = asyncio.Event()

        def on_sigint():
            print("\nInterrupt received — finishing in-flight documents, then stopping. Re-run to resume.")
            stop.set()
            for t in tasks:
                t.cancel()

        loop = asyncio.get_running_loop()
        loop.add_signal_handler(signal.SIGINT, on_sigint)

        await asyncio.gather(*tasks, return_exceptions=True)

    summary = store.summary()
    print("\n=== Summary ===")
    for status, count in sorted(summary.items()):
        print(f"  {status}: {count}")
    failed = [DocState(**v) for v in json.loads(args.state_file.read_text()).values() if v["status"] == "failed"]
    if failed:
        print(f"\n{len(failed)} document(s) permanently failed:")
        for f in failed:
            print(f"  - {f.pdf_path}: {f.last_error}")
        print(f"\nRe-run this script to retry them (they're not marked 'completed').")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Path to the scraped corpus CSV")
    parser.add_argument("--pdf-root", type=Path, default=DEFAULT_PDF_ROOT, help="Directory pdf_path column is relative to")
    parser.add_argument("--state-file", type=Path, default=DEFAULT_STATE_FILE, help="Resumability state JSON")
    parser.add_argument("--api-base", default=DEFAULT_API_BASE, help="Base URL of the running app")
    parser.add_argument("--concurrency", type=int, default=3, help="Max documents ingesting at once (default: 3, conservative for a single shared GPU)")
    parser.add_argument("--max-reingest-attempts", type=int, default=DEFAULT_MAX_REINGEST_ATTEMPTS, help="Reingest attempts before giving up on a document")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N rows (for a test run)")
    parser.add_argument("--dry-run", action="store_true", help="Preview without uploading")
    args = parser.parse_args()

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
