"""In-process BGE-M3 dense+sparse embedding adapter.

Runs BGE-M3 in-process rather than over HTTP via Infinity because Infinity
serves ``BAAI/bge-m3`` as dense-only (a server limitation). This uses BAAI's
reference implementation (``FlagEmbedding.BGEM3FlagModel``), the only path that
computes BGE-M3's lexical (sparse) weights alongside the dense vector. Fulfills
``app/kb/domain/interfaces.py::ITextEmbedder``; wired in
``app/kb/dependency.py::get_text_embedder``.
"""

import asyncio
import os
from functools import lru_cache
from typing import List

import structlog
import torch
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from app.kb.domain.interfaces import ITextEmbedder, EmbeddingResult

logger = structlog.get_logger(__name__)

# Retries transient CUDA OOM (e.g. a VRAM spike from another process). Not a
# network error, so it doesn't reuse app.shared.retry (which targets httpx).
# Clears the CUDA cache before each attempt, since retrying without freeing
# memory would likely OOM again.
def _clear_cuda_cache_before_retry(retry_state) -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


_cuda_oom_retry = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=20),
    retry=retry_if_exception_type(torch.cuda.OutOfMemoryError),
    before_sleep=_clear_cuda_cache_before_retry,
    reraise=True,
)


@lru_cache(maxsize=1)
def _resolve_model_source(model_name: str) -> str:
    """Resolve a Hub repo id to its local snapshot directory when already cached.

    Passing the bare repo id makes the Hub client revalidate the repo on every
    process start — two API round-trips before any weights load, which is
    latency on every worker boot and an outright failure when the machine is
    offline. The weights themselves are already on disk; only the revision
    lookup goes out. Handing the loader the snapshot path skips it.

    Falls back to ``model_name`` when the model isn't cached yet (first run) or
    when the path is already local, so the normal download still happens.
    """
    if os.path.isdir(model_name):
        return model_name
    try:
        from huggingface_hub import snapshot_download

        path = snapshot_download(model_name, local_files_only=True)
        logger.debug("bge_m3.resolved_local_snapshot", model=model_name, path=path)
        return path
    except Exception as exc:
        logger.info(
            "bge_m3.snapshot_not_cached",
            model=model_name,
            error=str(exc),
            action="falling back to Hub download",
        )
        return model_name


@lru_cache(maxsize=1)
def _load_model(model_name: str, use_fp16: bool, device: str):
    """Load BGEM3FlagModel once as a process-lifetime singleton. Loading takes
    seconds and several GB, so every instance sharing the same (model_name,
    use_fp16, device) reuses this one model rather than reloading per request.
    """
    from FlagEmbedding import BGEM3FlagModel

    source = _resolve_model_source(model_name)
    logger.info("bge_m3.loading", model=model_name, device=device, use_fp16=use_fp16)
    model = BGEM3FlagModel(source, use_fp16=use_fp16, device=device)
    logger.info("bge_m3.loaded", model=model_name, device=device)
    return model


class BGEM3Embeddings(ITextEmbedder):
    """Dense + sparse text embedding backed by an in-process BGE-M3."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: str = "cuda",
        use_fp16: bool = True,
        batch_size: int = 12,
    ) -> None:
        """Store model config; the model loads lazily (once per process) via
        :func:`_load_model` on the first ``embed_texts`` call.
        """
        self.model_name = model_name
        self.device = device
        self.use_fp16 = use_fp16
        self.batch_size = batch_size
        # FlagEmbedding's shared model is not thread-safe: concurrent encode()
        # calls can corrupt returned vectors and yield malformed Qdrant queries.
        # The embedder is a process singleton, so one lock protects all request
        # calls in this API process while still allowing model-internal batching.
        self._encode_lock = asyncio.Lock()
        logger.info(
            "BGEM3Embeddings initialized",
            model=model_name,
            device=device,
            use_fp16=use_fp16,
        )

    async def embed_texts(
        self, texts: List[str], is_query: bool = False
    ) -> List[EmbeddingResult]:
        """Encode texts via the shared BGE-M3 model, returning dense vectors and
        lexical (sparse) weights for each. Transient CUDA OOM is retried via
        :data:`_cuda_oom_retry` on the inner :meth:`_encode` call.

        ``is_query`` is accepted for interface parity and ignored: BGE-M3 is a
        symmetric encoder, so queries and documents take the same code path.
        """
        if not texts:
            return []

        try:
            async with self._encode_lock:
                output = await self._encode(texts)
        except Exception as exc:
            logger.error("bge_m3.embed_failed", batch_size=len(texts), error=str(exc))
            raise

        results: List[EmbeddingResult] = []
        for dense_vec, lexical_weights in zip(output["dense_vecs"], output["lexical_weights"]):
            sparse_indices = [int(k) for k in lexical_weights.keys()]
            sparse_values = [float(v) for v in lexical_weights.values()]
            results.append(
                EmbeddingResult(
                    dense=dense_vec.tolist(),
                    sparse_indices=sparse_indices,
                    sparse_values=sparse_values,
                )
            )
        return results

    @_cuda_oom_retry
    async def _encode(self, texts: List[str]) -> dict:
        model = _load_model(self.model_name, self.use_fp16, self.device)
        return await asyncio.to_thread(
            model.encode,
            texts,
            batch_size=self.batch_size,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )

    async def close(self) -> None:
        # The model is a process-lifetime singleton shared across instances —
        # nothing to release per-instance.
        pass
