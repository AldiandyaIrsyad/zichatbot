"""In-process Qwen3-Embedding adapter (dense) + pluggable sparse encoder.

Fulfills ``app/kb/domain/interfaces.py::ITextEmbedder``; wired in
``app/kb/dependency.py::get_text_embedder``.

Runs in-process rather than over HTTP via Infinity for a concrete reason:
``michaelf34/infinity:0.0.77`` ships ``transformers 4.49.0.dev0``, and the
Qwen3 architecture needs >= 4.51.0. Infinity rejects the checkpoint with
"Transformers does not recognize this architecture" (michaelfeil/infinity#611).
The app venv runs transformers 5.15.0, so the model loads here. This mirrors why
``bge_m3_embeddings.py`` is in-process, for a different underlying reason.

Qwen3-Embedding is **dense-only**, unlike BGE-M3 which emitted dense and lexical
weights together. The sparse half of hybrid search therefore comes from an
injected :class:`ISparseEncoder` — see ``sparse_encoders.py``.

The model is asymmetric: queries take an instruction prefix, documents do not.
Omitting it costs ~1-5% retrieval performance per the model card, so
``embed_texts(..., is_query=True)`` applies it and the corpus side never does.
"""

from __future__ import annotations

import asyncio
import os
from functools import lru_cache
from typing import List, Optional

import structlog
import torch
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from app.kb.domain.interfaces import ITextEmbedder, EmbeddingResult
from app.kb.infra.sparse_encoders import ISparseEncoder

logger = structlog.get_logger(__name__)

# Default retrieval instruction. Qwen3 was trained with one-sentence task
# descriptions in this shape; the corpus is Indonesian institutional/legal
# documents, so the task is stated in those terms.
DEFAULT_QUERY_INSTRUCTION = (
    "Given a question about Indonesian institutional and legal documents, "
    "retrieve the passages that answer it"
)

# Qwen3-Embedding-0.6B is natively 1024-dim, which happens to equal BGE-M3's
# width. That collision is a hazard, not a convenience: Qdrant will accept
# Qwen3 vectors into a BGE-M3 collection without error, silently mixing two
# incompatible spaces. Always index into a fresh collection.
QWEN3_DENSE_DIM: int = 1024


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
    """Resolve a Hub repo id to its cached snapshot dir, avoiding a revision
    round-trip on every process start. Same rationale as
    ``bge_m3_embeddings._resolve_model_source``.
    """
    if os.path.isdir(model_name):
        return model_name
    try:
        from huggingface_hub import snapshot_download

        path = snapshot_download(model_name, local_files_only=True)
        logger.debug("qwen3.resolved_local_snapshot", model=model_name, path=path)
        return path
    except Exception as exc:
        logger.info(
            "qwen3.snapshot_not_cached",
            model=model_name,
            error=str(exc),
            action="falling back to Hub download",
        )
        return model_name


@lru_cache(maxsize=1)
def _load_model(model_name: str, device: str, use_fp16: bool):
    """Load the SentenceTransformer once as a process-lifetime singleton."""
    from sentence_transformers import SentenceTransformer

    source = _resolve_model_source(model_name)
    kwargs = {"device": device}
    if use_fp16 and device.startswith("cuda"):
        kwargs["model_kwargs"] = {"torch_dtype": torch.float16}

    # Qwen3 needs left padding: it pools the last non-pad token, and right
    # padding would make that a pad token for every sequence but the longest.
    model = SentenceTransformer(source, **kwargs)
    try:
        model.tokenizer.padding_side = "left"
    except AttributeError:
        logger.warning("qwen3.padding_side_unset", model=model_name)
    return model


class Qwen3Embeddings(ITextEmbedder):
    """Dense Qwen3-Embedding + injected sparse encoder."""

    def __init__(
        self,
        sparse_encoder: ISparseEncoder,
        model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        device: str = "cuda",
        use_fp16: bool = True,
        batch_size: int = 8,
        query_instruction: str = DEFAULT_QUERY_INSTRUCTION,
        max_seq_length: Optional[int] = None,
    ) -> None:
        """Store config; the model loads lazily on first use via
        :func:`_load_model`.
        """
        self.sparse_encoder = sparse_encoder
        self.model_name = model_name
        self.device = device
        self.use_fp16 = use_fp16
        self.batch_size = batch_size
        self.query_instruction = query_instruction
        self.max_seq_length = max_seq_length
        # SentenceTransformer.encode is not safe under concurrent calls on one
        # shared model. The embedder is a process singleton, so a single lock
        # serialises API requests while leaving model-internal batching intact.
        # Same reasoning as BGEM3Embeddings._encode_lock.
        self._encode_lock = asyncio.Lock()
        logger.info(
            "Qwen3Embeddings initialized",
            model=model_name,
            device=device,
            use_fp16=use_fp16,
            sparse=type(sparse_encoder).__name__,
        )

    async def embed_texts(
        self, texts: List[str], is_query: bool = False
    ) -> List[EmbeddingResult]:
        """Embed ``texts``, returning dense Qwen3 vectors paired with sparse
        vectors from the injected encoder.

        Queries get the instruction prefix; documents are embedded bare.
        """
        if not texts:
            return []

        prepared = [self._format_query(t) for t in texts] if is_query else list(texts)

        try:
            async with self._encode_lock:
                dense_vectors = await self._encode(prepared)
        except Exception as exc:
            logger.error("qwen3.embed_failed", batch_size=len(texts), error=str(exc))
            raise

        # Sparse runs on the raw text: the instruction prefix is an artefact of
        # the dense model's training, and feeding it to BM25 would inject a
        # constant block of English tokens into every query's term vector.
        sparse_vectors = await self.sparse_encoder.encode(texts, is_query=is_query)

        results: List[EmbeddingResult] = []
        for dense_vec, (indices, values) in zip(dense_vectors, sparse_vectors):
            results.append(
                EmbeddingResult(
                    dense=dense_vec,
                    sparse_indices=indices,
                    sparse_values=values,
                )
            )
        return results

    def _format_query(self, query: str) -> str:
        """Apply Qwen3's documented query template."""
        return f"Instruct: {self.query_instruction}\nQuery: {query}"

    @_cuda_oom_retry
    async def _encode(self, texts: List[str]) -> List[List[float]]:
        model = _load_model(self.model_name, self.device, self.use_fp16)
        if self.max_seq_length is not None:
            model.max_seq_length = self.max_seq_length

        def _run() -> List[List[float]]:
            vectors = model.encode(
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            return [v.tolist() for v in vectors]

        return await asyncio.to_thread(_run)

    async def close(self) -> None:
        await self.sparse_encoder.close()
