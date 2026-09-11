"""HuggingFace Text Embeddings Inference (TEI) reranker adapter.

Fulfills ``app/kb/domain/interfaces.py::IReranker``; wired in
``app/kb/dependency.py::get_reranker``.

Serves the same model as ``infinity_reranker.InfinityReranker``
(``BAAI/bge-reranker-v2-m3``) from a different server. TEI is maintained by
HuggingFace, does real continuous batching, and is not pinned to an old
transformers the way ``michaelf34/infinity:0.0.77`` is — that pin (transformers
4.49, Qwen3 needs >= 4.51) is what made Infinity a dead end for newer models.

Wire-level differences from Infinity's ``/rerank``, which is why this is a
separate adapter rather than a base-URL change:

* the passage field is ``texts``, not ``documents``
* no ``model`` field — a TEI process serves exactly one model
* the response is a bare JSON array, not ``{"results": [...]}``
* the score field is ``score``, not ``relevance_score``
* ``top_k`` is not a server-side parameter; the cap is applied here

Batching is TEI's job: one ``/rerank`` call carrying N passages is scheduled as
a batch server-side, so this adapter deliberately sends the whole candidate set
in a single request rather than chunking it. The server must therefore allow a
client batch at least as large as ``INITIAL_SEARCH_TOP_K`` — see
``--max-client-batch-size`` in docker-compose.yaml, whose default of 32 would
reject the 50 candidates SearchService sends.
"""

from typing import List, Optional

import httpx
import structlog

from app.kb.domain.interfaces import IReranker, RerankResult

logger = structlog.get_logger(__name__)


class TEIReranker(IReranker):
    """HTTP adapter for a TEI server running a cross-encoder reranker."""

    def __init__(self, base_url: str, model: str = "") -> None:
        """Open an HTTP client for the TEI ``/rerank`` endpoint.

        ``model`` is accepted for parity with ``InfinityReranker`` and logging
        only — a TEI process serves one model, chosen by its ``--model-id`` flag,
        and rejects a ``model`` field in the request body.
        """
        self.model = model
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=httpx.Timeout(120.0, connect=10.0),
        )
        logger.info("TEIReranker initialized", model=model, base_url=base_url)

    async def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None,
    ) -> List[RerankResult]:
        """Rerank ``documents`` against ``query``, descending by score."""
        if not documents:
            return []

        try:
            response = await self._client.post(
                "/rerank",
                json={"query": query, "texts": documents, "raw_scores": False},
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            # Fail-closed to the original retrieval order, matching
            # InfinityReranker: a reranker outage degrades ranking rather than
            # failing the search.
            logger.warning("rerank.failed", error=str(exc), doc_count=len(documents))
            return [RerankResult(index=i, score=0.0) for i in range(len(documents))]

        # TEI returns a bare array; tolerate a {"results": [...]} envelope too so
        # a future server change does not silently yield zero results.
        items = payload if isinstance(payload, list) else payload.get("results", [])

        results = [
            RerankResult(
                index=int(item.get("index", 0)),
                score=float(item.get("score", item.get("relevance_score", 0.0))),
            )
            for item in items
        ]

        # TEI sorts already; sort defensively to guarantee the port's contract.
        results.sort(key=lambda r: r.score, reverse=True)
        if top_k is not None:
            results = results[:top_k]

        logger.info(
            "kb.rerank.completed",
            query_length=len(query),
            doc_count=len(documents),
            returned_count=len(results),
        )
        return results

    async def close(self) -> None:
        await self._client.aclose()
