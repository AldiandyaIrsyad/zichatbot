#!/usr/bin/env python3
"""Re-ingest KB documents affected by the scan-only page misclassification fix.

Context: classify_page() (app/thesis/chunking/page_classifier.py) previously
missed scan-only pages whose only image element had empty OCR text but where
Unstructured's own noisy OCR pass also emitted real-word-shaped-but-garbled
text as separate NarrativeText/UncategorizedText elements (too long/dense for
the existing garbage heuristic). Those pages fell to MIXED instead of VISUAL,
so the page's image got the generic figure-description VLM prompt instead of
the full-page-extraction prompt — producing layout descriptions ("Logo
Institusi... Pada bagian atas tengah halaman...") instead of the document's
actual text. The fix adds a native_text_len signal (PyMuPDF's page.get_text()
length) that isn't fooled by Unstructured's OCR noise.

This script finds which already-ingested documents actually have at least
one such scan-only page (a free, local PyMuPDF check — no API cost), deletes
their existing chunks (Postgres parent/child rows + Qdrant vectors, but NOT
the PDFDocument row or source file), and re-triggers ingestion through the
running app so they get reprocessed under the fixed classifier.

Usage:
    # Preflight: see how many documents would actually be affected, no changes
    .venv/bin/python scripts/reingest_affected_docs.py --dry-run

    # Try a small batch first
    .venv/bin/python scripts/reingest_affected_docs.py --limit 20

    # The real run (resumable — safe to re-run after Ctrl-C or a crash)
    .venv/bin/python scripts/reingest_affected_docs.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import signal
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import asyncpg
import fitz  # PyMuPDF
import httpx
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.rag.chunking.page_classifier import NATIVE_TEXT_LEN_THRESHOLD  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_STATE_FILE = REPO_ROOT / "scripts" / "reingest_state.json"
DEFAULT_API_BASE = "http://localhost:8000"

PG_HOST = "127.0.0.1"
PG_PORT = 5432
PG_USER = "postgres"
PG_PASSWORD = "UdlpHsAngnTleKokBMvdGGpod"
PG_DB = "postgres"
QDRANT_HOST = "127.0.0.1"
QDRANT_PORT = 6333
QDRANT_COLLECTION = "knowledge_base"

POLL_INTERVAL_SEC = 5.0
MAX_INGEST_WAIT_SEC = 25 * 60
DEFAULT_MAX_REINGEST_ATTEMPTS = 2
DEFAULT_CONCURRENCY = 3


@dataclass
class DocState:
    doc_id: str
    title: str
    pdf_path: str
    scan_only_pages: int
    status: str = "pending"  # pending -> cleaned -> completed | failed
    attempts: int = 0
    last_error: Optional[str] = None
    updated_at: float = field(default_factory=time.time)


class StateStore:
    """JSON-backed state, keyed by doc_id. Written after every document."""

    def __init__(self, path: Path):
        self.path = path
        self._data: dict[str, dict] = {}
        if path.exists():
            self._data = json.loads(path.read_text())

    def get(self, doc_id: str) -> Optional[DocState]:
        raw = self._data.get(doc_id)
        return DocState(**raw) if raw else None

    def save(self, state: DocState) -> None:
        state.updated_at = time.time()
        self._data[state.doc_id] = asdict(state)
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


async def find_candidates() -> list[DocState]:
    """Scan every KB document's PDF locally to find ones with at least one
    scan-only page under the fixed classifier's threshold. Free — no API
    calls, just reads files already on disk."""
    conn = await asyncpg.connect(host=PG_HOST, port=PG_PORT, user=PG_USER, password=PG_PASSWORD, database=PG_DB)
    rows = await conn.fetch("SELECT id, title, pdf_path FROM pdf_documents")
    await conn.close()

    candidates: list[DocState] = []
    missing = 0
    for r in rows:
        path = r["pdf_path"]
        if not Path(path).exists():
            missing += 1
            continue
        try:
            doc = fitz.open(path)
        except Exception:
            continue
        scan_only_pages = 0
        for page in doc:
            text_len = len(page.get_text())
            has_image = len(page.get_images()) > 0
            if text_len <= NATIVE_TEXT_LEN_THRESHOLD and has_image:
                scan_only_pages += 1
        doc.close()
        if scan_only_pages > 0:
            candidates.append(DocState(
                doc_id=str(r["id"]),
                title=r["title"],
                pdf_path=path,
                scan_only_pages=scan_only_pages,
            ))
    if missing:
        print(f"  WARNING: {missing} document(s) had a missing pdf_path on disk, skipped", file=sys.stderr)
    return candidates


async def clean_old_chunks(pg: asyncpg.Connection, qdrant: AsyncQdrantClient, doc_id: str) -> None:
    """Delete a document's existing chunks (Postgres + Qdrant) without
    touching its PDFDocument row or source file, so ingest_document() can
    regenerate them fresh under the same doc_id. Chunk IDs are uuid4 (not
    deterministic), so re-ingesting without this step would create
    duplicates rather than replacing the old (broken) chunks."""
    await pg.execute("DELETE FROM child_chunks WHERE doc_id = $1", doc_id)
    await pg.execute("DELETE FROM parent_chunks WHERE doc_id = $1", doc_id)
    await qdrant.delete(
        collection_name=QDRANT_COLLECTION,
        points_selector=Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]),
    )


async def poll_until_done(client: httpx.AsyncClient, doc_id: str) -> tuple[str, Optional[str]]:
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
    pg_pool: asyncpg.Pool,
    qdrant: AsyncQdrantClient,
    store: StateStore,
    state: DocState,
    max_reingest_attempts: int,
    index: int,
    total: int,
) -> None:
    prefix = f"[{index}/{total}]"
    try:
        if state.status == "pending":
            print(f"{prefix} cleaning old chunks: {state.title[:60]} ({state.scan_only_pages} scan-only pages)")
            async with pg_pool.acquire() as pg:
                await clean_old_chunks(pg, qdrant, state.doc_id)
            state.status = "cleaned"
            store.save(state)

        while True:
            print(f"{prefix} reingesting: {state.title[:60]} (doc_id={state.doc_id})")
            await reingest(client, state.doc_id)
            status, error = await poll_until_done(client, state.doc_id)

            if status == "completed":
                state.status = "completed"
                state.last_error = None
                store.save(state)
                print(f"{prefix} DONE: {state.title[:60]}")
                return

            state.attempts += 1
            state.last_error = error
            if state.attempts > max_reingest_attempts:
                state.status = "failed"
                store.save(state)
                print(f"{prefix} FAILED (giving up after {state.attempts} attempts): {state.title[:60]}: {error}")
                return

            print(f"{prefix} status={status} error={error!r} — retrying (attempt {state.attempts}/{max_reingest_attempts})")
            store.save(state)
            await asyncio.sleep(min(30, 2 ** state.attempts))

    except Exception as exc:  # noqa: BLE001
        state.status = "failed"
        state.last_error = str(exc)
        store.save(state)
        print(f"{prefix} FAILED (exception): {state.title[:60]}: {exc}")


async def run(args: argparse.Namespace) -> None:
    print("Scanning KB documents locally for scan-only pages (no API cost)...")
    candidates = await find_candidates()
    print(f"Found {len(candidates)} document(s) with >=1 truly scan-only page under the fixed classifier.")

    store = StateStore(args.state_file)
    rows = candidates
    for i, row in enumerate(rows):
        prior = store.get(row.doc_id)
        if prior:
            rows[i] = prior

    if args.limit:
        rows = rows[: args.limit]

    todo = [r for r in rows if r.status != "completed"]
    already_done = len(rows) - len(todo)
    print(f"{already_done} already completed, {len(todo)} to process")

    if args.dry_run:
        print("--dry-run: not changing anything. Sample of what would run:")
        for r in todo[:15]:
            print(f"  [{r.status}] {r.scan_only_pages} scan-only page(s) — {r.title[:70]}")
        if len(todo) > 15:
            print(f"  ... and {len(todo) - 15} more")
        return

    if not todo:
        print("Nothing to do.")
        return

    pg_pool = await asyncpg.create_pool(host=PG_HOST, port=PG_PORT, user=PG_USER, password=PG_PASSWORD, database=PG_DB, min_size=1, max_size=args.concurrency)
    qdrant = AsyncQdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    async with httpx.AsyncClient(base_url=args.api_base, timeout=httpx.Timeout(120.0, connect=10.0)) as client:
        try:
            resp = await client.get("/api/admin/pdfs")
            resp.raise_for_status()
        except Exception as exc:
            print(f"ERROR: can't reach the API at {args.api_base}: {exc}", file=sys.stderr)
            sys.exit(1)

        sem = asyncio.Semaphore(args.concurrency)

        async def bound(i: int, r: DocState) -> None:
            async with sem:
                await process_one(client, pg_pool, qdrant, store, r, args.max_reingest_attempts, i, len(todo))

        tasks = [asyncio.create_task(bound(i + 1, r)) for i, r in enumerate(todo)]

        def on_sigint():
            print("\nInterrupt received — finishing in-flight documents, then stopping. Re-run to resume.")
            for t in tasks:
                t.cancel()

        loop = asyncio.get_running_loop()
        loop.add_signal_handler(signal.SIGINT, on_sigint)

        await asyncio.gather(*tasks, return_exceptions=True)

    await pg_pool.close()
    await qdrant.close()

    summary = store.summary()
    print("\n=== Summary ===")
    for status, count in sorted(summary.items()):
        print(f"  {status}: {count}")
    failed = [DocState(**v) for v in json.loads(args.state_file.read_text()).values() if v["status"] == "failed"]
    if failed:
        print(f"\n{len(failed)} document(s) permanently failed:")
        for f in failed:
            print(f"  - {f.title[:70]}: {f.last_error}")
        print("\nRe-run this script to retry them (they're not marked 'completed').")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--state-file", type=Path, default=DEFAULT_STATE_FILE, help="Resumability state JSON")
    parser.add_argument("--api-base", default=DEFAULT_API_BASE, help="Base URL of the running app")
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY, help="Max documents reingesting at once")
    parser.add_argument("--max-reingest-attempts", type=int, default=DEFAULT_MAX_REINGEST_ATTEMPTS, help="Reingest attempts before giving up on a document")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N candidates (for a test run)")
    parser.add_argument("--dry-run", action="store_true", help="Preview without changing anything")
    args = parser.parse_args()

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
