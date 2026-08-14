"""Tests for the InfinityReranker adapter.

Verifies:
    - Rerank reorders documents by descending relevance score.
    - Fail-closed behavior: on HTTP error, returns original order with zero scores.
    - Empty document list returns empty result.
    - top_k parameter is passed through.
"""

from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from app.kb.domain.interfaces import RerankResult
from app.kb.infra.infinity_reranker import InfinityReranker


@pytest.fixture
def reranker() -> InfinityReranker:
    """Create an InfinityReranker with a mocked HTTP client."""
    r = InfinityReranker(base_url="http://localhost:7997", model="bge-reranker-v2-m3")
    return r


def _mock_response(data: Dict[str, Any]) -> MagicMock:
    """Create a mock httpx.Response."""
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = data
    return resp


@pytest.mark.asyncio
async def test_rerank_reorders_by_score(reranker: InfinityReranker) -> None:
    """Rerank should return results sorted by descending score."""
    mock_data = {
        "results": [
            {"index": 2, "relevance_score": 0.95},
            {"index": 0, "relevance_score": 0.30},
            {"index": 1, "relevance_score": 0.80},
        ]
    }
    reranker._client = MagicMock(spec=httpx.AsyncClient)
    reranker._client.post = AsyncMock(return_value=_mock_response(mock_data))

    docs = ["doc zero", "doc one", "doc two"]
    results = await reranker.rerank("query", docs)

    assert len(results) == 3
    # Should be sorted by descending score
    assert results[0].index == 2
    assert results[0].score == pytest.approx(0.95)
    assert results[1].index == 1
    assert results[1].score == pytest.approx(0.80)
    assert results[2].index == 0
    assert results[2].score == pytest.approx(0.30)


@pytest.mark.asyncio
async def test_rerank_fail_closed_on_error(reranker: InfinityReranker) -> None:
    """On HTTP error, reranker should return original order with zero scores."""
    reranker._client = MagicMock(spec=httpx.AsyncClient)
    reranker._client.post = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))

    docs = ["doc a", "doc b", "doc c"]
    results = await reranker.rerank("query", docs)

    assert len(results) == 3
    for i, r in enumerate(results):
        assert r.index == i
        assert r.score == 0.0


@pytest.mark.asyncio
async def test_rerank_empty_documents(reranker: InfinityReranker) -> None:
    """Empty document list should return empty result without HTTP call."""
    reranker._client = MagicMock(spec=httpx.AsyncClient)
    reranker._client.post = AsyncMock()

    results = await reranker.rerank("query", [])
    assert results == []
    reranker._client.post.assert_not_called()


@pytest.mark.asyncio
async def test_rerank_passes_top_k(reranker: InfinityReranker) -> None:
    """top_k parameter should be included in the request payload."""
    mock_data = {"results": [{"index": 0, "relevance_score": 0.9}]}
    reranker._client = MagicMock(spec=httpx.AsyncClient)
    reranker._client.post = AsyncMock(return_value=_mock_response(mock_data))

    await reranker.rerank("query", ["doc"], top_k=5)

    call_args = reranker._client.post.call_args
    payload = call_args.kwargs["json"]
    assert payload["top_k"] == 5


@pytest.mark.asyncio
async def test_rerank_enforces_top_k_when_server_ignores_it(reranker: InfinityReranker) -> None:
    """If the server returns more results than top_k asked for, the client must
    still truncate — the deployed Infinity server has been observed to ignore
    the top_k request parameter entirely and return all documents reranked."""
    mock_data = {
        "results": [
            {"index": i, "relevance_score": 1.0 - i * 0.01} for i in range(5)
        ]
    }
    reranker._client = MagicMock(spec=httpx.AsyncClient)
    reranker._client.post = AsyncMock(return_value=_mock_response(mock_data))

    results = await reranker.rerank("query", [f"doc {i}" for i in range(5)], top_k=2)

    assert len(results) == 2
    assert results[0].index == 0
    assert results[1].index == 1


@pytest.mark.asyncio
async def test_rerank_handles_missing_score_field(reranker: InfinityReranker) -> None:
    """Reranker should handle missing 'relevance_score' by falling back to 'score'."""
    mock_data = {
        "results": [
            {"index": 0, "score": 0.7},
        ]
    }
    reranker._client = MagicMock(spec=httpx.AsyncClient)
    reranker._client.post = AsyncMock(return_value=_mock_response(mock_data))

    results = await reranker.rerank("query", ["doc"])
    assert len(results) == 1
    assert results[0].score == pytest.approx(0.7)


@pytest.mark.asyncio
async def test_rerank_returns_rerank_result_type(reranker: InfinityReranker) -> None:
    """Results should be RerankResult instances."""
    mock_data = {"results": [{"index": 0, "relevance_score": 0.9}]}
    reranker._client = MagicMock(spec=httpx.AsyncClient)
    reranker._client.post = AsyncMock(return_value=_mock_response(mock_data))

    results = await reranker.rerank("query", ["doc"])
    assert all(isinstance(r, RerankResult) for r in results)
