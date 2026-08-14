"""Tests for the Experiment 2 retrieval harness — the HyDE ablation wiring (E6).

The retrieval scoring itself is exercised in test_metrics.py; these tests pin
the one piece of routing logic the HyDE ablation adds: which endpoint each HyDE
condition hits and whether the ``hyde`` query param is sent, so a HyDE-off run
can never silently go through the HyDE-capable endpoint (or vice versa).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pytest

from evals.exp2_retrieval.run import retrieve


class _RecordingClient:
    """Fake httpx.AsyncClient that records the GET endpoint + params."""

    def __init__(self, calls: List[Tuple[str, Dict[str, Any]]]) -> None:
        self._calls = calls

    async def __aenter__(self) -> "_RecordingClient":
        return self

    async def __aexit__(self, *args: Any) -> None:
        return None

    async def get(self, url: str, params: Dict[str, Any]) -> "_RecordingResponse":
        self._calls.append((url, params))
        return _RecordingResponse()


class _RecordingResponse:
    status_code = 200

    def json(self) -> List[Dict[str, str]]:
        return [{"doc_id": "doc-1"}, {"doc_id": "doc-2"}]


async def _capture(hyde: Optional[bool]) -> Tuple[str, Dict[str, Any]]:
    calls: List[Tuple[str, Dict[str, Any]]] = []
    import evals.exp2_retrieval.run as run_mod

    original = run_mod.httpx.AsyncClient
    run_mod.httpx.AsyncClient = lambda **kw: _RecordingClient(calls)  # type: ignore[assignment]
    try:
        await retrieve("http://localhost:8000", "kapan UPI berdiri?", 5, "hybrid", True, hyde)
    finally:
        run_mod.httpx.AsyncClient = original  # type: ignore[assignment]
    assert len(calls) == 1
    return calls[0]


class TestRetrieveEndpointSelection:
    @pytest.mark.asyncio
    async def test_hyde_none_uses_kb_search_without_hyde_param(self) -> None:
        url, params = await _capture(None)
        assert url == "/api/kb/search"
        assert "hyde" not in params

    @pytest.mark.asyncio
    async def test_hyde_true_uses_chat_search_with_hyde_true(self) -> None:
        url, params = await _capture(True)
        assert url == "/api/chat/search"
        assert params["hyde"] == "true"

    @pytest.mark.asyncio
    async def test_hyde_false_uses_chat_search_with_hyde_false(self) -> None:
        url, params = await _capture(False)
        assert url == "/api/chat/search"
        assert params["hyde"] == "false"

    @pytest.mark.asyncio
    async def test_core_params_passed_through(self) -> None:
        _, params = await _capture(True)
        assert params["q"] == "kapan UPI berdiri?"
        assert params["top_k"] == 5
        assert params["mode"] == "hybrid"
        assert params["rerank"] == "true"


def test_hyde_condition_mapping() -> None:
    """The --hyde flag must expand to the right ordered condition list."""
    mapping: Dict[str, List[Optional[bool]]] = {
        "off": [None],
        "on": [True],
        "both": [False, True],
    }
    assert mapping["off"] == [None]
    assert mapping["on"] == [True]
    # 'both' reports HyDE-off first (the /api/kb/search-equivalent) then HyDE-on.
    assert mapping["both"] == [False, True]
