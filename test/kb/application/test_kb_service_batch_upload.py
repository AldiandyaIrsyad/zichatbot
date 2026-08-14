"""Tests for KBApplicationService.upload_pdfs_batch() per-file isolation.

Regression coverage: previously all files in a batch shared one uncommitted
DB transaction with no per-file try/except, so a failure anywhere in the
loop (disk error, DB error) rolled back every already-processed file in
that request — silently discarding successful uploads. Each file must now
be committed independently so a later failure can't undo earlier successes.
"""

from __future__ import annotations

import io
from typing import List
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import BackgroundTasks

from app.kb.application.kb_service import KBApplicationService, _safe_upload_filename
from app.kb.domain.models import PDFDocument


def _make_upload_file(filename: str, content: bytes = b"%PDF-1.4 fake") -> MagicMock:
    f = MagicMock()
    f.filename = filename
    f.file = io.BytesIO(content)
    return f


def _make_pdf(pdf_id: str, title: str) -> PDFDocument:
    return PDFDocument(id=pdf_id, title=title, description="", pdf_path=f"/tmp/{pdf_id}.pdf")


@pytest.mark.asyncio
async def test_all_files_succeed_and_are_committed_individually(tmp_path) -> None:
    repo = MagicMock()
    repo.create_pdf = AsyncMock(side_effect=[
        _make_pdf("d1", "Doc 1"),
        _make_pdf("d2", "Doc 2"),
    ])
    repo.commit = AsyncMock()
    repo.rollback = AsyncMock()

    worker = MagicMock()
    worker.ingest_document = AsyncMock()

    svc = KBApplicationService(
        kb_repo=repo,
        vector_store=MagicMock(),
        ingest_worker=worker,
        upload_dir=str(tmp_path),
    )

    files = [_make_upload_file("a.pdf"), _make_upload_file("b.pdf")]
    results, failures = await svc.upload_pdfs_batch(
        files=files,
        titles=["Doc 1", "Doc 2"],
        descriptions=["", ""],
        bg_tasks=BackgroundTasks(),
    )

    assert len(results) == 2
    assert failures == []
    assert repo.commit.await_count == 2
    assert repo.rollback.await_count == 0


@pytest.mark.asyncio
async def test_one_failure_does_not_lose_already_succeeded_files(tmp_path) -> None:
    """The core regression: file 2 fails, but file 1 (already committed)
    and file 3 (processed after) must both still succeed."""
    repo = MagicMock()
    repo.create_pdf = AsyncMock(side_effect=[
        _make_pdf("d1", "Doc 1"),
        RuntimeError("disk full"),
        _make_pdf("d3", "Doc 3"),
    ])
    repo.commit = AsyncMock()
    repo.rollback = AsyncMock()

    worker = MagicMock()
    worker.ingest_document = AsyncMock()

    svc = KBApplicationService(
        kb_repo=repo,
        vector_store=MagicMock(),
        ingest_worker=worker,
        upload_dir=str(tmp_path),
    )

    files = [_make_upload_file("a.pdf"), _make_upload_file("b.pdf"), _make_upload_file("c.pdf")]
    results, failures = await svc.upload_pdfs_batch(
        files=files,
        titles=["Doc 1", "Doc 2", "Doc 3"],
        descriptions=["", "", ""],
        bg_tasks=BackgroundTasks(),
    )

    assert [r.id for r in results] == ["d1", "d3"]
    assert len(failures) == 1
    assert failures[0]["filename"] == "b.pdf"
    assert "disk full" in failures[0]["error"]
    # commit for the 2 successes, rollback for the 1 failure
    assert repo.commit.await_count == 2
    assert repo.rollback.await_count == 1


class TestSafeUploadFilename:
    """Regression coverage for _safe_upload_filename().

    Two independent things break on long on-disk filenames, both reproduced
    live: the local filesystem raises ENAMETOOLONG around 255 bytes, and
    Unstructured Cloud's job API (which receives this same filename)
    silently completes with output_node_files: None — no error at all —
    once the filename gets long (reproduced at 240 chars). Real titles in
    this app's corpus (long Indonesian bureaucratic decree titles) routinely
    exceed that.
    """

    def test_long_title_and_filename_stays_well_under_filesystem_limit(self) -> None:
        long_name = ("A" * 300) + ".pdf"
        result = _safe_upload_filename(long_name)
        assert len(result) < 100

    def test_preserves_extension(self) -> None:
        assert _safe_upload_filename("report.pdf").endswith(".pdf")
        assert _safe_upload_filename("A" * 300 + ".pdf").endswith(".pdf")

    def test_missing_extension_defaults_to_pdf(self) -> None:
        assert _safe_upload_filename("no_extension_file").endswith(".pdf")

    def test_none_filename_handled(self) -> None:
        result = _safe_upload_filename(None)
        assert result.endswith(".pdf")
        assert len(result) < 100

    def test_two_calls_are_unique(self) -> None:
        """UUID-prefixed so concurrent uploads of same-named files never collide."""
        a = _safe_upload_filename("document.pdf")
        b = _safe_upload_filename("document.pdf")
        assert a != b

    def test_path_separators_in_filename_are_sanitized(self) -> None:
        """A filename containing "/" must not be interpreted as a path
        separator when joined into the upload directory — no "/" may survive
        even though "." is otherwise allowed (so a bare ".." with no
        separator around it is inert, not a traversal vector)."""
        result = _safe_upload_filename("../../etc/passwd.pdf")
        assert "/" not in result
