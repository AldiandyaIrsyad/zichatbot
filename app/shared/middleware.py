"""HTTP middleware shared across all bounded contexts.

Currently provides :class:`CorrelationIdMiddleware`, which stamps every
request with a correlation ID so that all log lines emitted during that
request (via ``structlog``) can be traced end-to-end in Loki/Grafana.
"""

import uuid
from typing import Callable, Awaitable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
import structlog


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """Stamps each request with a correlation ID.

    Uses the ``X-Request-ID`` request header when present (so upstream
    proxies can propagate their trace ID), else a fresh UUID4. The ID is bound
    to ``structlog.contextvars`` (injected into every log line by the
    ``merge_contextvars`` processor), stored on ``request.state.request_id``,
    and echoed in the ``X-Request-ID`` response header.
    """

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        # Reuse an upstream-provided ID if present, else mint a new one.
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))

        # Reset per-request context so IDs from a previous request on a
        # reused worker don't leak in.
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)

        request.state.request_id = request_id

        response = await call_next(request)

        # Echo the ID back so clients can correlate response ↔ logs.
        response.headers["X-Request-ID"] = request_id
        return response
