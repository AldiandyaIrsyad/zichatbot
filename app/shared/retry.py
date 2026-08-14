"""Shared retry policy for outbound calls to rate-limited third-party APIs.

For clients calling external services with real rate limits (Unstructured
Cloud, OpenRouter), a transient 429/5xx/timeout would otherwise permanently
fail the triggering operation — significant at bulk-ingestion scale where
hundreds of documents each make several such calls.

Consumed by ``app/kb/infra/unstructured_client.py`` and
``app/thesis/vlm/client.py::OpenRouterVLMClient`` (the one ``thesis/`` module
allowed to import ``shared/``, since it already depends on ``httpx``).
"""

import httpx


class RetryableResponseError(RuntimeError):
    """A successful HTTP response whose model content is unusable."""
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception


def _is_retryable(exc: BaseException) -> bool:
    """Retry connection/timeout errors and 429/5xx responses.

    Explicitly does NOT retry other 4xx responses (e.g. a malformed-PDF 422,
    or a 401 bad API key) — those will fail identically on every attempt, so
    retrying just wastes time reproducing the same failure.
    """
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        return status == 429 or status >= 500
    return isinstance(exc, (httpx.TransportError, RetryableResponseError))


def make_external_api_retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=4, max=60),
):
    """Build the retry decorator. Factored out so tests can inject a
    near-zero wait strategy instead of sleeping for real seconds between
    attempts."""
    return retry(
        stop=stop,
        wait=wait,
        retry=retry_if_exception(_is_retryable),
        reraise=True,
    )


external_api_retry = make_external_api_retry()
