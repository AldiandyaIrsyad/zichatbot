"""Tests for the shared external-API retry policy.

Regression coverage: at 922-PDF bulk-ingestion scale, a transient 429/5xx
from Unstructured Cloud or OpenRouter should be retried with backoff rather
than permanently failing that document — but a genuine 4xx client error
(e.g. a malformed-PDF 422) should NOT be retried, since it'll fail
identically every time.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import httpx
import pytest
from tenacity import stop_after_attempt, wait_none

from app.shared.retry import make_external_api_retry

# Same retry/stop logic as production, but with no sleep between attempts —
# these tests exercise the retry *decision* logic, not real backoff timing.
external_api_retry = make_external_api_retry(stop=stop_after_attempt(3), wait=wait_none())


def _make_status_error(status_code: int) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "http://example.test/x")
    response = httpx.Response(status_code, request=request)
    return httpx.HTTPStatusError("error", request=request, response=response)


@pytest.mark.asyncio
async def test_retries_on_transient_500_then_succeeds() -> None:
    call_count = 0

    @external_api_retry
    async def flaky():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise _make_status_error(500)
        return "ok"

    result = await flaky()
    assert result == "ok"
    assert call_count == 3


@pytest.mark.asyncio
async def test_retries_on_429() -> None:
    call_count = 0

    @external_api_retry
    async def flaky():
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise _make_status_error(429)
        return "ok"

    result = await flaky()
    assert result == "ok"
    assert call_count == 2


@pytest.mark.asyncio
async def test_does_not_retry_on_422_client_error() -> None:
    """A malformed-PDF 422 will fail identically every time — retrying it
    just wastes time, so the policy must not retry non-429 4xx responses."""
    call_count = 0

    @external_api_retry
    async def always_422():
        nonlocal call_count
        call_count += 1
        raise _make_status_error(422)

    with pytest.raises(httpx.HTTPStatusError):
        await always_422()
    assert call_count == 1


@pytest.mark.asyncio
async def test_retries_on_connection_error() -> None:
    call_count = 0

    @external_api_retry
    async def flaky():
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise httpx.ConnectError("connection refused")
        return "ok"

    result = await flaky()
    assert result == "ok"
    assert call_count == 2


@pytest.mark.asyncio
async def test_exhausts_retries_and_reraises() -> None:
    call_count = 0

    @external_api_retry
    async def always_500():
        nonlocal call_count
        call_count += 1
        raise _make_status_error(500)

    with pytest.raises(httpx.HTTPStatusError):
        await always_500()
    assert call_count == 3  # stop_after_attempt(3)
