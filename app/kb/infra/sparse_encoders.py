"""Sparse (lexical) encoders for the ``bm25`` vector in Qdrant.

BGE-M3 produced dense and lexical weights from one model, so the sparse channel
had no separate implementation. Qwen3-Embedding is dense-only, so swapping the
dense model leaves the sparse half without a producer — and the exp2 baseline
shows sparse is the *strongest* channel on this corpus (MRR@5 0.7786 sparse vs
0.6970 dense), so it cannot simply be dropped.

Two implementations are provided because the right choice is an empirical
question, not an obvious one:

* :class:`BM25SparseEncoder` — FastEmbed's ``Qdrant/bm25``. CPU-only, no VRAM,
  no transformer. Runs **unstemmed**: FastEmbed's stemmer supports 18 languages
  and Indonesian is not among them (Snowball has no Indonesian stemmer), so
  ``disable_stemmer=True`` is the only correct setting here — passing a foreign
  stemmer would mangle Indonesian morphology worse than leaving tokens intact.
  Exact-token matching suits the identifier-heavy queries ("Keputusan Rektor
  Nomor 998/UN40.R5/KM.06.01/2026") that sparse retrieval wins on.

* :class:`BGEM3LexicalSparseEncoder` — keeps BGE-M3's learned lexical weights,
  the exact channel the baseline measured, by running BGE-M3 on CPU purely for
  its sparse output. Costs ~160 ms/call and no VRAM, preserving GPU headroom for
  Qwen3. Use this to isolate the dense-model change from the sparse change.

Both fulfil :class:`ISparseEncoder` and are injected into
``qwen3_embeddings.Qwen3Embeddings``.
"""

from __future__ import annotations

import asyncio
from functools import lru_cache
from typing import List, Protocol, Tuple

import structlog

logger = structlog.get_logger(__name__)

# (indices, values) for one text, aligned positionally.
SparseVec = Tuple[List[int], List[float]]


class ISparseEncoder(Protocol):
    """Port for producing Qdrant sparse vectors from text."""

    async def encode(self, texts: List[str], is_query: bool = False) -> List[SparseVec]:
        """Return one ``(indices, values)`` pair per input text, in order."""
        ...

    async def close(self) -> None:
        """Release the underlying model."""
        ...


@lru_cache(maxsize=1)
def _load_bm25(disable_stemmer: bool):
    """Load FastEmbed's BM25 once per process (it reads a stopword/vocab bundle
    from disk on construction).
    """
    from fastembed.sparse.bm25 import Bm25

    return Bm25("Qdrant/bm25", disable_stemmer=disable_stemmer)


class BM25SparseEncoder(ISparseEncoder):
    """Classical BM25 sparse vectors via FastEmbed (CPU, no VRAM).

    The Qdrant collection applies ``Modifier.IDF``, so document vectors carry
    raw term frequencies and the IDF weighting happens server-side. Queries use
    ``query_embed``, which skips the document-side weighting — mixing the two up
    silently degrades scoring, which is why ``is_query`` is threaded through.
    """

    def __init__(self, disable_stemmer: bool = True) -> None:
        self.disable_stemmer = disable_stemmer
        logger.info("BM25SparseEncoder initialized", disable_stemmer=disable_stemmer)

    async def encode(self, texts: List[str], is_query: bool = False) -> List[SparseVec]:
        if not texts:
            return []
        return await asyncio.to_thread(self._encode_sync, texts, is_query)

    def _encode_sync(self, texts: List[str], is_query: bool) -> List[SparseVec]:
        model = _load_bm25(self.disable_stemmer)
        embeddings = model.query_embed(texts) if is_query else model.embed(texts)
        return [
            ([int(i) for i in e.indices], [float(v) for v in e.values])
            for e in embeddings
        ]

    async def close(self) -> None:
        # Process-lifetime singleton shared across instances — nothing per-instance.
        pass


class BGEM3LexicalSparseEncoder(ISparseEncoder):
    """BGE-M3's learned lexical weights, dense output discarded.

    Reuses ``bge_m3_embeddings.BGEM3Embeddings`` so the sparse vectors are
    bit-for-bit what the current collection holds. Defaults to CPU: the point of
    this encoder is to keep the measured sparse channel while the GPU goes to
    Qwen3.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: str = "cpu",
        use_fp16: bool = False,
        batch_size: int = 12,
    ) -> None:
        from app.kb.infra.bge_m3_embeddings import BGEM3Embeddings

        # fp16 on CPU is slower than fp32 on most x86 (no native half math),
        # so the default flips relative to the GPU embedder.
        self._inner = BGEM3Embeddings(
            model_name=model_name,
            device=device,
            use_fp16=use_fp16,
            batch_size=batch_size,
        )
        logger.info(
            "BGEM3LexicalSparseEncoder initialized", model=model_name, device=device
        )

    async def encode(self, texts: List[str], is_query: bool = False) -> List[SparseVec]:
        if not texts:
            return []
        results = await self._inner.embed_texts(texts)
        return [(r.sparse_indices, r.sparse_values) for r in results]

    async def close(self) -> None:
        await self._inner.close()


def build_sparse_encoder(kind: str, **kwargs) -> ISparseEncoder:
    """Construct the sparse encoder named by ``kind`` ("bm25" or "bge-m3")."""
    normalized = (kind or "").strip().lower()
    if normalized in {"bm25", "fastembed", "fastembed-bm25"}:
        return BM25SparseEncoder(**kwargs)
    if normalized in {"bge-m3", "bge_m3", "bgem3", "lexical"}:
        return BGEM3LexicalSparseEncoder(**kwargs)
    raise ValueError(
        f"Unknown sparse encoder {kind!r}. Expected 'bm25' or 'bge-m3'."
    )
