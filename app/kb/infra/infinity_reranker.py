"""Infinity reranker adapter for re-ranking retrieved documents.

Fulfills: ``app/kb/domain/interfaces.py::IReranker``.
Wired in: ``app/kb/dependency.py::get_reranker``.
"""

import httpx
import structlog
from typing import List, Optional

from app.kb.domain.interfaces import IReranker, RerankResult

logger = structlog.get_logger(__name__)


class InfinityReranker(IReranker):
    """HTTP adapter for the Infinity reranking server. Calls the ``/rerank``
    endpoint with the BGE-reranker-v2-m3 model.
    """

    def __init__(self, base_url: str, model: str) -> None:
        """Open an HTTP client for the Infinity reranking endpoint."""
        self.model = model
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=httpx.Timeout(120.0, connect=10.0),
        )
        logger.info("InfinityReranker initialized", model=model, base_url=base_url)

    async def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None,
    ) -> List[RerankResult]:
        """Rerank ``documents`` by relevance to ``query``, returning
        ``RerankResult``s in descending score order (capped to ``top_k`` when
        given).
        """
        if not documents:
            return []

        try:
            payload: dict = {
                "model": self.model,
                "query": query,
                "documents": documents,
                "return_text": False,
            }
            if top_k is not None:
                payload["top_k"] = top_k

            response = await self._client.post("/rerank", json=payload)
            response.raise_for_status()
            response_data = response.json()
        except Exception as exc:
            # Fail-closed: return original order with zero scores so the
            # pipeline degrades gracefully to the pre-rerank ordering.
            logger.warning("rerank.failed", error=str(exc), doc_count=len(documents))
            return [RerankResult(index=i, score=0.0) for i in range(len(documents))]

        results_raw = response_data.get("results", [])
        rerank_results: List[RerankResult] = []
        for item in results_raw:
            idx = int(item.get("index", 0))
            score = float(item.get("relevance_score", item.get("score", 0.0)))
            rerank_results.append(RerankResult(index=idx, score=score))

        # Infinity returns results sorted by score; sort defensively to
        # guarantee the contract.
        rerank_results.sort(key=lambda r: r.score, reverse=True)

        # The server should honor top_k, but enforce the cap client-side too.
        if top_k is not None:
            rerank_results = rerank_results[:top_k]

        logger.info(
            "kb.rerank.completed",
            query_length=len(query),
            doc_count=len(documents),
            returned_count=len(rerank_results),
        )
        return rerank_results

    async def close(self) -> None:
        await self._client.aclose()
