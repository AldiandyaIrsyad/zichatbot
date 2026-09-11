"""Page-window splitting for documents over MinerU's hosted per-file caps.

The hosted API rejects a file over 600 pages or 200 MB. These tests cover the
handler that makes page count a non-issue: plan windows, slice losslessly, and
re-join the parsed elements with the page numbers the original document had —
because a wrong page number is a wrong citation, which is worse than a refused
document.
"""

from __future__ import annotations

import os

import pymupdf
import pytest

from app.kb.infra.mineru_client import (
    MAX_PAGES,
    MinerUClient,
    _page_windows,
    _retry_delay,
    _to_elements,
    _write_page_range,
)


def _make_pdf(path: str, pages: int) -> str:
    doc = pymupdf.open()
    for i in range(pages):
        page = doc.new_page()
        page.insert_text((72, 72), f"page {i + 1}")
    doc.save(path)
    doc.close()
    return path


class TestPageWindows:
    def test_single_window_when_document_fits(self, tmp_path):
        pdf = _make_pdf(str(tmp_path / "small.pdf"), 10)
        assert _page_windows(pdf, pages_per_window=500, max_bytes=10**9) == [(0, 9)]

    def test_splits_into_inclusive_zero_based_windows(self, tmp_path):
        pdf = _make_pdf(str(tmp_path / "big.pdf"), 25)
        windows = _page_windows(pdf, pages_per_window=10, max_bytes=10**9)
        assert windows == [(0, 9), (10, 19), (20, 24)]

    def test_windows_cover_every_page_exactly_once(self, tmp_path):
        pdf = _make_pdf(str(tmp_path / "cover.pdf"), 37)
        windows = _page_windows(pdf, pages_per_window=8, max_bytes=10**9)
        covered = [p for first, last in windows for p in range(first, last + 1)]
        assert covered == list(range(37))

    def test_oversized_window_is_halved_until_it_fits(self, tmp_path):
        """A page-count-legal window can still bust the byte cap on dense scans."""
        pdf = _make_pdf(str(tmp_path / "heavy.pdf"), 16)
        whole = os.path.getsize(pdf)
        # A cap just under a quarter of the file forces two levels of halving.
        windows = _page_windows(pdf, pages_per_window=16, max_bytes=whole // 4)
        assert len(windows) > 2
        covered = [p for first, last in windows for p in range(first, last + 1)]
        assert covered == list(range(16))

    def test_single_page_over_the_cap_raises_rather_than_recursing(self, tmp_path):
        pdf = _make_pdf(str(tmp_path / "tiny_cap.pdf"), 3)
        with pytest.raises(ValueError, match="cannot be split any further"):
            _page_windows(pdf, pages_per_window=3, max_bytes=1)


class TestWritePageRange:
    def test_slice_has_the_requested_pages(self, tmp_path):
        src = _make_pdf(str(tmp_path / "src.pdf"), 20)
        dst = str(tmp_path / "slice.pdf")
        _write_page_range(src, dst, 5, 9)
        with pymupdf.open(dst) as doc:
            assert doc.page_count == 5
            assert "page 6" in doc[0].get_text()
            assert "page 10" in doc[4].get_text()


class TestPageOffset:
    def test_offset_restores_original_page_numbers(self):
        content_list = [
            {"type": "text", "text": "first", "page_idx": 0},
            {"type": "text", "text": "second", "page_idx": 3},
        ]
        elements = _to_elements(content_list, page_offset=500)
        # MinerU numbers each request from 0; page_idx 0 of the second window is
        # absolute page 501, not page 1.
        assert [e.metadata["page_number"] for e in elements] == [501, 504]

    def test_no_offset_is_the_unsplit_behaviour(self):
        elements = _to_elements([{"type": "text", "text": "x", "page_idx": 0}])
        assert elements[0].metadata["page_number"] == 1


class TestParsePdfRouting:
    """``parse_pdf`` must split on page count, and only on page count."""

    @pytest.mark.asyncio
    async def test_document_within_caps_takes_one_request(self, tmp_path, monkeypatch):
        pdf = _make_pdf(str(tmp_path / "one.pdf"), 12)
        client = MinerUClient(api_key="k", pages_per_request=500)
        calls: list[int] = []

        async def fake_window(path, page_offset=0, image_label=None):
            calls.append(page_offset)
            return []

        monkeypatch.setattr(client, "_parse_window", fake_window)
        await client.parse_pdf(pdf)
        assert calls == [0]
        await client.close()

    @pytest.mark.asyncio
    async def test_oversized_document_is_split_and_offsets_advance(self, tmp_path, monkeypatch):
        pdf = _make_pdf(str(tmp_path / "many.pdf"), 25)
        client = MinerUClient(api_key="k", pages_per_request=10)
        seen: list[tuple[int, int]] = []

        async def fake_window(path, page_offset=0, image_label=None):
            with pymupdf.open(path) as doc:
                seen.append((page_offset, doc.page_count))
            return []

        monkeypatch.setattr(client, "_parse_window", fake_window)
        await client.parse_pdf(pdf)
        assert seen == [(0, 10), (10, 10), (20, 5)]
        await client.close()

    @pytest.mark.asyncio
    async def test_elements_are_concatenated_in_document_order(self, tmp_path, monkeypatch):
        pdf = _make_pdf(str(tmp_path / "order.pdf"), 20)
        client = MinerUClient(api_key="k", pages_per_request=10)

        async def fake_window(path, page_offset=0, image_label=None):
            return _to_elements(
                [{"type": "text", "text": f"w{page_offset}", "page_idx": 0}],
                page_offset=page_offset,
            )

        monkeypatch.setattr(client, "_parse_window", fake_window)
        elements = await client.parse_pdf(pdf)
        assert [e.text for e in elements] == ["w0", "w10"]
        assert [e.metadata["page_number"] for e in elements] == [1, 11]
        await client.close()

    def test_pages_per_request_cannot_exceed_the_api_cap(self):
        client = MinerUClient(api_key="k", pages_per_request=5000)
        assert client._pages_per_request == MAX_PAGES


class TestRetryDelay:
    def _exc(self, status: int, headers: dict | None = None):
        import httpx

        request = httpx.Request("POST", "https://mineru.net/x")
        response = httpx.Response(status, headers=headers or {}, request=request)
        return httpx.HTTPStatusError("boom", request=request, response=response)

    def test_transport_error_uses_exponential_backoff(self):
        import httpx

        assert _retry_delay(httpx.ConnectError("x"), attempt=3) == 8.0

    def test_rate_limit_waits_at_least_half_a_minute(self):
        # The submit endpoints allow 300/min; retrying in 2s just burns an attempt.
        assert _retry_delay(self._exc(429), attempt=1) == 30.0

    def test_retry_after_header_wins(self):
        assert _retry_delay(self._exc(429, {"Retry-After": "45"}), attempt=1) == 45.0

    def test_absurd_retry_after_is_capped(self):
        assert _retry_delay(self._exc(429, {"Retry-After": "9999"}), attempt=1) == 120.0

    def test_non_429_status_error_is_plain_backoff(self):
        assert _retry_delay(self._exc(500), attempt=2) == 4.0
