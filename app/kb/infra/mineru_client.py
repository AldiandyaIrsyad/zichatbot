"""MinerU document parser adapter.

Fulfills ``app/kb/domain/interfaces.py::IDocumentParser``; wired in
``app/kb/dependency.py::get_document_parser`` and selected with
``PARSER_BACKEND=mineru``.

Why this exists: Unstructured returns table *structure* but destroys table
*content* on this corpus. Measured on the UKT tariff schedule
(``003 Tahun 2022``), Unstructured produced headers like ``Kelompok 1}``, dropped
``Rp.`` prefixes and emitted ``[merged]`` cells, while MinerU recovered 96
programme codes and 714 tariff values — rows matching an independent VLM
transcription exactly. Those pages carry a bad OCR text layer, so they classify
as TEXT/MIXED and never reach the VLM path; the parser is the only lever.

Hosted limits (https://mineru.net/doc/docs/limit_en/, checked 2026-08-17):
200 MB per file, 200 files per batch request, 300 submit requests/min, 1000
result-poll requests/min, 10,000 files/day. A daily quota of 2,000 pages runs at
highest priority; pages beyond that are still parsed, at lower priority.

On page count the documentation and the API disagree: the docs advertise 600
pages per file, while this endpoint rejects the task after upload with
``number of pages exceeds limit (200 pages), please split the file and try
again`` (observed 2026-08-17 on three 211–333 page documents). ``MAX_PAGES``
therefore follows the API, not the docs — the failure arrives *after* a
successful upload, so trusting the higher number costs the whole transfer before
anything says no.

Documents larger than either per-file cap are handled rather than refused: see
``_parse_split``, which slices the PDF into page windows, parses each as its own
request, and re-joins the elements with their original page numbers. Any page
count is therefore parseable; the caps only decide how many requests it costs.

Flow (MinerU v4): request an upload URL, PUT the file, poll the batch until the
task reports ``done``, then download the result zip and map
``content_list.json`` onto :class:`ParsedElement`.
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import time
import zipfile
from typing import Any, Dict, List, Optional

import httpx
import structlog

from app.rag.chunking.models import ParsedElement

logger = structlog.get_logger(__name__)

# Hard per-file caps the hosted API enforces. A request over either one is
# rejected outright, so ``parse_pdf`` splits before sending rather than after
# being refused.
MAX_PAGES = 200
MAX_UPLOAD_BYTES = 200 * 1024 * 1024

# Pages per request when splitting. Deliberately below MAX_PAGES: the cap is
# what the API rejects, and a margin costs one extra request on a long document
# while protecting against an off-by-one in anyone's page arithmetic — including
# whether MinerU counts a 200-page file as at or over its limit.
DEFAULT_PAGES_PER_REQUEST = 180

# Floor for the recursive size-driven subdivision in ``_page_windows``. A window
# this small that still exceeds the byte cap is a single page over 200 MB, which
# no amount of further splitting can fix — it fails with a clear error instead of
# recursing forever.
MIN_SPLIT_PAGES = 1

# MinerU content_list types -> the element types the chunking router keys on
# (``app/rag/chunking/router.py``: TABLE_ELEMENT_TYPES / FIGURE_ELEMENT_TYPES).
_TYPE_MAP = {
    "text": "NarrativeText",
    "title": "Title",
    "header": "Title",
    "table": "Table",
    "image": "Image",
    "equation": "NarrativeText",
}


class MinerUClient:
    """Parse PDFs with the hosted MinerU API."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://mineru.net",
        batch_path: str = "/api/v4/file-urls/batch",
        language: str = "id",
        image_dir: str = "./uploads/knowledge_base/images",
        poll_interval_sec: float = 10.0,
        max_wait_sec: float = 1800.0,
        connect_timeout_sec: float = 30.0,
        read_timeout_sec: float = 60.0,
        write_timeout_sec: float = 300.0,
        upload_deadline_sec: float = 480.0,
        pages_per_request: int = DEFAULT_PAGES_PER_REQUEST,
        max_upload_bytes: int = MAX_UPLOAD_BYTES,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._batch_path = batch_path
        self._language = language
        self._image_dir = image_dir
        self._poll_interval = poll_interval_sec
        self._max_wait = max_wait_sec
        # Per-phase budgets rather than one flat `timeout` for read/write/pool.
        # A flat 300s meant every phase waited five minutes before failing, and
        # `_upload`'s retry repeated the hang — 14 minutes against a ~3.5 min
        # honest baseline for a 7-page scan.
        #
        # `write` stays deliberately large. Measured on this uplink, the
        # multi-megabyte PUT to object storage progresses in bursts separated by
        # stalls of tens of seconds (1.89 MB moved at ~20 KB/s effective), so a
        # tight per-write timeout fails uploads that would have succeeded. What
        # bounds the damage is `upload_deadline_sec` — a wall-clock budget
        # across *all* upload attempts — rather than a short per-write timeout.
        # `connect`/`read`/`pool` stay tight, since none of them carry a body.
        self._upload_deadline = upload_deadline_sec
        self._pages_per_request = max(1, min(pages_per_request, MAX_PAGES))
        self._max_upload_bytes = max_upload_bytes
        self._connect_timeout = connect_timeout_sec
        self._read_timeout = read_timeout_sec
        self._write_timeout = write_timeout_sec
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=connect_timeout_sec,
                read=read_timeout_sec,
                write=write_timeout_sec,
                pool=30.0,
            )
        )
        logger.info("MinerUClient initialized", base_url=self._base_url, language=language)

    @property
    def _headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}"}

    async def parse_pdf(self, file_path: str) -> List[ParsedElement]:
        """Parse ``file_path`` into ordered :class:`ParsedElement`s.

        Documents within the hosted per-file caps take one request. Larger ones
        are split into page windows (:meth:`_parse_split`) and re-joined, so page
        count is never a reason a document cannot be ingested.
        """
        page_count = _page_count(file_path)
        size_bytes = os.path.getsize(file_path)

        if page_count <= self._pages_per_request and size_bytes <= self._max_upload_bytes:
            elements = await self._parse_window(file_path, page_offset=0)
        else:
            elements = await self._parse_split(file_path, page_count)

        logger.info(
            "mineru.parse.success",
            filename=os.path.basename(file_path),
            elements=len(elements),
            tables=sum(1 for e in elements if e.element_type == "Table"),
            pages=page_count,
        )
        return elements

    async def _parse_window(
        self,
        file_path: str,
        page_offset: int = 0,
        image_label: Optional[str] = None,
    ) -> List[ParsedElement]:
        """Run one full request cycle for one file (a whole PDF or one slice).

        ``page_offset`` is the 0-based index of this slice's first page in the
        *original* document, added back to MinerU's ``page_idx`` so a split
        document keeps the page numbers the citations and the page classifier
        rely on. ``image_label`` keeps each slice's extracted images in their own
        directory — the default name is derived from the file stem, and slices of
        one document share a stem.
        """
        batch_id, upload_url = await self._request_upload(file_path)
        await self._upload(upload_url, file_path)
        zip_url = await self._await_result(batch_id, file_path)
        content_list, image_map = await self._download_result(
            zip_url, file_path, image_label=image_label
        )
        return _to_elements(content_list, image_map, page_offset=page_offset)

    async def _parse_split(
        self, file_path: str, page_count: int
    ) -> List[ParsedElement]:
        """Parse a document that exceeds a per-file cap, one page window at a time.

        Windows are sliced out with PyMuPDF (a lossless page copy, not a
        re-render) into a temporary directory, parsed **sequentially**, and their
        elements concatenated in document order with page numbers restored. Each
        window is an independent request, so a failure names the window it
        happened in rather than the whole document.

        Sequential rather than concurrent on purpose: a 1000-page document is
        already half of MinerU's daily high-priority page quota, and firing its
        windows in parallel only moves the same pages through a narrower
        priority window while making 429s more likely.
        """
        import tempfile

        name = os.path.basename(file_path)
        windows = _page_windows(
            file_path,
            pages_per_window=self._pages_per_request,
            max_bytes=self._max_upload_bytes,
        )
        logger.info(
            "mineru.split.start",
            filename=name,
            pages=page_count,
            windows=len(windows),
            pages_per_request=self._pages_per_request,
        )

        elements: List[ParsedElement] = []
        stem = os.path.splitext(name)[0]
        with tempfile.TemporaryDirectory(prefix="mineru_split_") as tmp_dir:
            for index, (first, last) in enumerate(windows, start=1):
                part_path = os.path.join(
                    tmp_dir, f"{stem[:60]}_p{first + 1}-{last + 1}.pdf"
                )
                _write_page_range(file_path, part_path, first, last)
                logger.info(
                    "mineru.split.window",
                    filename=name,
                    window=f"{index}/{len(windows)}",
                    pages=f"{first + 1}-{last + 1}",
                    size_mb=round(os.path.getsize(part_path) / 1e6, 1),
                )
                part_elements = await self._parse_window(
                    part_path,
                    page_offset=first,
                    image_label=f"{stem[:40]}_p{first + 1}",
                )
                elements.extend(part_elements)

        logger.info(
            "mineru.split.done", filename=name, windows=len(windows), elements=len(elements)
        )
        return elements

    async def _request_upload(self, file_path: str, attempts: int = 3) -> tuple[str, str]:
        """Ask MinerU for a batch id and pre-signed upload URL.

        Retried on the same terms as ``_upload``: MinerU rate-limits repeated
        calls in a short window with ``429``, and without a retry that fails the
        whole document at the very first step.
        """
        last: Optional[Exception] = None
        for attempt in range(1, attempts + 1):
            try:
                resp = await self._client.post(
                    self._base_url + self._batch_path,
                    headers={**self._headers, "Content-Type": "application/json"},
                    json={
                        "enable_formula": False,
                        "enable_table": True,
                        "language": self._language,
                        # is_ocr forces OCR rather than trusting the embedded text
                        # layer, which on this corpus is exactly the layer that is
                        # corrupt.
                        "files": [{"name": os.path.basename(file_path), "is_ocr": True}],
                    },
                )
                resp.raise_for_status()
                break
            except (httpx.TransportError, httpx.HTTPStatusError) as exc:
                last = exc
                logger.warning(
                    "mineru.batch_request_retry",
                    attempt=attempt,
                    attempts=attempts,
                    error=str(exc),
                    rate_limited=_is_rate_limited(exc),
                    filename=os.path.basename(file_path),
                )
                if attempt < attempts:
                    await asyncio.sleep(_retry_delay(exc, attempt))
        else:
            raise RuntimeError(
                f"MinerU batch request failed after {attempts} attempts for "
                f"{os.path.basename(file_path)}: {last}"
            ) from last

        data = resp.json()
        if data.get("code") not in (0, "0"):
            raise RuntimeError(f"MinerU batch request failed: {data}")
        payload = data["data"]
        return payload["batch_id"], payload["file_urls"][0]

    async def _upload(self, upload_url: str, file_path: str, attempts: int = 3) -> None:
        """PUT the file to the pre-signed URL, retrying transport errors.

        The upload is a multi-megabyte body to object storage and drops
        occasionally (``httpx.ReadError``). Without a retry a single dropped
        connection fails the whole document, which on a corpus-wide run means
        re-parsing it from scratch. The pre-signed URL must be sent without the
        auth header.

        Attempts are bounded by ``upload_deadline_sec`` of wall clock rather
        than by a tight per-write timeout, so a slow-but-progressing upload is
        allowed to finish while a genuinely stuck one still fails promptly.
        """
        with open(file_path, "rb") as fh:
            content = fh.read()
        started = time.monotonic()
        last: Optional[Exception] = None
        for attempt in range(1, attempts + 1):
            remaining = self._upload_deadline - (time.monotonic() - started)
            if remaining <= 0:
                break
            try:
                resp = await self._client.put(
                    upload_url,
                    content=content,
                    timeout=httpx.Timeout(
                        connect=self._connect_timeout,
                        read=self._read_timeout,
                        write=min(self._write_timeout, remaining),
                        pool=30.0,
                    ),
                )
                resp.raise_for_status()
                return
            except (httpx.TransportError, httpx.HTTPStatusError) as exc:
                last = exc
                logger.warning(
                    "mineru.upload_retry",
                    attempt=attempt,
                    attempts=attempts,
                    # WriteTimeout and friends stringify to "", which previously
                    # produced an error message ending in a bare colon.
                    error=str(exc) or type(exc).__name__,
                    rate_limited=_is_rate_limited(exc),
                    elapsed_sec=round(time.monotonic() - started, 1),
                    filename=os.path.basename(file_path),
                )
                if attempt < attempts:
                    await asyncio.sleep(_retry_delay(exc, attempt))
        raise RuntimeError(
            f"MinerU upload failed after {attempts} attempts "
            f"({time.monotonic() - started:.0f}s) for "
            f"{os.path.basename(file_path)}: {str(last) or type(last).__name__}"
        ) from last

    async def _await_result(
        self, batch_id: str, file_path: str, max_consecutive_failures: int = 5
    ) -> str:
        waited = 0.0
        consecutive_failures = 0
        url = f"{self._base_url}/api/v4/extract-results/batch/{batch_id}"
        while waited < self._max_wait:
            await asyncio.sleep(self._poll_interval)
            waited += self._poll_interval
            try:
                resp = await self._client.get(url, headers=self._headers)
                resp.raise_for_status()
            except httpx.HTTPError as exc:
                # A 429 on the *polling* endpoint says nothing about this task —
                # the result limit (1000/min) is shared across a corpus run, so
                # counting it toward the give-up budget would abandon documents
                # that are parsing perfectly well. Back off and keep waiting.
                if _is_rate_limited(exc):
                    logger.warning(
                        "mineru.poll_rate_limited",
                        elapsed_sec=waited,
                        filename=os.path.basename(file_path),
                    )
                    delay = _retry_delay(exc, 1)
                    await asyncio.sleep(delay)
                    waited += delay
                    continue
                # A transient poll failure is worth riding out, but a persistent
                # one used to burn the whole max_wait budget and then report a
                # bare TimeoutError that hid the real transport error. Surface
                # the underlying cause once it is clearly not transient.
                consecutive_failures += 1
                logger.warning(
                    "mineru.poll_failed",
                    error=str(exc),
                    elapsed_sec=waited,
                    consecutive_failures=consecutive_failures,
                )
                if consecutive_failures >= max_consecutive_failures:
                    raise RuntimeError(
                        f"MinerU polling failed {consecutive_failures} times in a "
                        f"row for {os.path.basename(file_path)}: {exc}"
                    ) from exc
                continue
            consecutive_failures = 0

            results = (resp.json().get("data") or {}).get("extract_result") or []
            for item in results:
                state = item.get("state")
                if state == "done":
                    return item["full_zip_url"]
                if state == "failed":
                    raise RuntimeError(
                        f"MinerU extraction failed for {os.path.basename(file_path)}: "
                        f"{item.get('err_msg')}"
                    )
            logger.info("mineru.poll", elapsed_sec=waited, states=[i.get("state") for i in results])

        raise TimeoutError(
            f"MinerU did not finish {os.path.basename(file_path)} within {self._max_wait}s"
        )

    async def _download_result(
        self, zip_url: str, file_path: str, image_label: Optional[str] = None
    ) -> tuple[List[Dict[str, Any]], Dict[str, str]]:
        """Fetch the result zip: content list plus extracted images on disk.

        MinerU's ``img_path`` refers to a member *inside the zip*
        ("images/<sha>.jpg"), so handing it downstream unchanged makes the VLM
        raise ``FileNotFoundError``. The images are written next to the ones the
        page renderer produces, and the returned map rewrites those paths.
        """
        resp = await self._client.get(zip_url, follow_redirects=True)
        resp.raise_for_status()
        archive = zipfile.ZipFile(io.BytesIO(resp.content))

        names = [n for n in archive.namelist() if n.endswith("content_list.json")]
        if not names:
            raise RuntimeError(f"MinerU result zip has no content_list.json: {archive.namelist()[:6]}")
        # Prefer the plain content_list over any _v2 variant for a stable schema.
        content_list = json.loads(archive.read(sorted(names, key=len)[0]))

        stem = image_label or os.path.splitext(os.path.basename(file_path))[0][:40]
        out_dir = os.path.join(self._image_dir, f"mineru_{stem}")
        os.makedirs(out_dir, exist_ok=True)
        image_map: Dict[str, str] = {}
        for member in archive.namelist():
            if not member.startswith("images/") or member.endswith("/"):
                continue
            target = os.path.join(out_dir, os.path.basename(member))
            with open(target, "wb") as fh:
                fh.write(archive.read(member))
            image_map[member] = target

        logger.info("mineru.images_extracted", count=len(image_map), dir=out_dir)
        return content_list, image_map

    async def close(self) -> None:
        await self._client.aclose()


def _is_rate_limited(exc: Exception) -> bool:
    resp = getattr(exc, "response", None)
    return resp is not None and resp.status_code == 429


def _retry_delay(exc: Exception, attempt: int) -> float:
    """Seconds to wait before retrying, honoring MinerU's rate-limit signal.

    Plain transport errors get exponential backoff. A ``429`` gets at least a
    half-minute: the submit endpoints allow 300 requests/min, so being refused
    means the window is genuinely full and retrying in 2 s just spends another
    attempt. ``Retry-After``, when sent, wins over both — capped so a bad header
    cannot park a corpus run for an hour.
    """
    base = float(2 ** attempt)
    if not _is_rate_limited(exc):
        return base
    retry_after = getattr(exc, "response").headers.get("Retry-After")
    if retry_after:
        try:
            return max(base, min(float(retry_after), 120.0))
        except ValueError:
            pass
    return max(base, 30.0)


def _page_count(file_path: str) -> int:
    import pymupdf

    with pymupdf.open(file_path) as doc:
        return doc.page_count


def _write_page_range(src_path: str, dst_path: str, first: int, last: int) -> None:
    """Copy pages ``first..last`` (0-based, inclusive) of ``src_path`` to ``dst_path``.

    ``insert_pdf`` copies page objects rather than re-rendering them, so the
    slice carries the same image data and text layer the original had — which
    matters here because the whole point of MinerU on this corpus is fidelity.
    """
    import pymupdf

    with pymupdf.open(src_path) as src, pymupdf.open() as out:
        out.insert_pdf(src, from_page=first, to_page=last)
        out.save(dst_path, garbage=3, deflate=True)


def _page_windows(
    file_path: str,
    pages_per_window: int,
    max_bytes: int,
) -> List[tuple[int, int]]:
    """Plan the page windows a split parse should send, as 0-based inclusive pairs.

    Two caps are in play and only one of them is knowable from the page count:
    a window of ``pages_per_window`` pages can still exceed ``max_bytes`` if the
    pages are dense scans. So each window is measured by actually slicing it, and
    a too-large one is halved and re-measured. Measuring costs a temporary write
    per candidate window, which is cheap next to the upload it prevents from
    being rejected.
    """
    import tempfile

    total = _page_count(file_path)
    pending = [
        (start, min(start + pages_per_window, total) - 1)
        for start in range(0, total, pages_per_window)
    ]

    windows: List[tuple[int, int]] = []
    with tempfile.TemporaryDirectory(prefix="mineru_probe_") as probe_dir:
        while pending:
            first, last = pending.pop(0)
            probe = os.path.join(probe_dir, f"probe_{first}_{last}.pdf")
            _write_page_range(file_path, probe, first, last)
            size = os.path.getsize(probe)
            os.remove(probe)

            span = last - first + 1
            if size <= max_bytes or span <= MIN_SPLIT_PAGES:
                if size > max_bytes:
                    raise ValueError(
                        f"{os.path.basename(file_path)} page {first + 1} is "
                        f"{size / 1e6:.0f} MB on its own, over MinerU's "
                        f"{max_bytes / 1e6:.0f} MB per-file limit; it cannot be "
                        "split any further."
                    )
                windows.append((first, last))
                continue

            middle = first + span // 2
            # Re-queue the halves at the front so windows stay in page order.
            pending.insert(0, (middle, last))
            pending.insert(0, (first, middle - 1))

    return windows


def _to_elements(
    content_list: List[Dict[str, Any]],
    image_map: Optional[Dict[str, str]] = None,
    page_offset: int = 0,
) -> List[ParsedElement]:
    """Map MinerU's content_list onto ParsedElement, preserving order.

    ``page_number`` is 1-based to match Unstructured (MinerU's ``page_idx`` is
    0-based); the page classifier and image extractor both assume 1-based.
    ``page_number`` entries are dropped — they are the printed folio, not content.

    ``page_offset`` shifts every page number by the slice's position in the
    original document. MinerU numbers each request from 0, so without the offset
    a split document would report page 1 several times and every citation past
    the first window would point at the wrong page.
    """
    elements: List[ParsedElement] = []
    for item in content_list:
        kind = (item.get("type") or "").lower()
        if kind == "page_number":
            continue

        element_type = _TYPE_MAP.get(kind, "NarrativeText")

        # MinerU marks headings with `text_level` while still typing them as
        # "text" — "Pasal 1", "MEMUTUSKAN:", "Menimbang" all arrive as
        # type=text, text_level=2. Mapping on `type` alone therefore produced a
        # document with no Title elements at all, so create_parent_chunks built
        # a flat structure: every parent at depth 0 with parent_id=None, no
        # breadcrumbs, and sibling hydration permanently disabled (it requires
        # parent_id). Unstructured emitted real Title elements, which is why the
        # hierarchy existed before the parser swap.
        #
        # `category_depth` is deliberately NOT set from text_level: MinerU's
        # levels are visual, while infer_heading_depth's BAB/Pasal heuristics
        # encode this corpus's actual legal hierarchy (BAB -> 0, Pasal -> 1).
        if kind in ("text", "title", "header") and item.get("text_level") is not None:
            element_type = "Title"
        metadata: Dict[str, Any] = {}
        page_idx = item.get("page_idx")
        if page_idx is not None:
            metadata["page_number"] = int(page_idx) + 1 + page_offset

        if kind == "table":
            # table_body is HTML, matching what Unstructured puts in `text` for
            # tables, so downstream table handling is unchanged.
            text = item.get("table_body") or ""
            caption = _join(item.get("table_caption"))
            if caption:
                metadata["table_caption"] = caption
            metadata["text_as_html"] = text
        elif kind == "image":
            text = _join(item.get("img_caption")) or ""
            raw_path = item.get("img_path")
            if raw_path:
                # Rewrite the in-zip path to where it was extracted; skip the
                # element if extraction missed it rather than handing the VLM a
                # path that cannot exist.
                local = (image_map or {}).get(raw_path)
                if local:
                    metadata["image_path"] = local
        else:
            text = item.get("text") or ""

        if not text and element_type != "Image":
            continue
        elements.append(ParsedElement(element_type=element_type, text=text, metadata=metadata))
    return elements


def _join(value: Any) -> str:
    if isinstance(value, list):
        return " ".join(str(v) for v in value if v)
    return str(value) if value else ""
