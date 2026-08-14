"""Strategy dispatcher for the chunking pipeline.

Single entry point the ingestion worker calls, routing on
``ChunkingConfig.strategy`` to either :mod:`app.rag.chunking.logic`
(hierarchical, heading-aware) or :mod:`app.rag.chunking.fixed`
(fixed-size token windows, domain-agnostic). Keeps the strategy choice out of
``app/kb/application/ingest_worker.py``, which just holds the config.

Signatures mirror the underlying functions one-for-one, so a caller can swap
``create_parent_chunks``/``split_into_children`` for these without restructuring.
"""

from typing import List, Optional

from .config import ChunkingConfig, ITokenizer
from .fixed import create_parent_chunks_fixed, split_into_children_fixed
from .logic import create_parent_chunks, split_into_children
from .models import ChildChunkData, ParentChunkData, ParsedElement


def _require_tokenizer(tokenizer: Optional[ITokenizer]) -> ITokenizer:
    if tokenizer is None:
        raise ValueError(
            "chunking strategy 'fixed' requires a tokenizer; pass the one from "
            "app.kb.dependency.get_chunk_tokenizer()"
        )
    return tokenizer


def chunk_parents(
    elements: List[ParsedElement],
    doc_id: str,
    config: ChunkingConfig,
    tokenizer: Optional[ITokenizer] = None,
) -> List[ParentChunkData]:
    """Build parent chunks using the strategy named in ``config``."""
    if config.strategy == "fixed":
        return create_parent_chunks_fixed(
            elements,
            doc_id,
            _require_tokenizer(tokenizer),
            max_tokens=config.fixed_parent_max_tokens,
        )
    return create_parent_chunks(elements, doc_id, max_chars=config.parent_max_chars)


def chunk_children(
    parent: ParentChunkData,
    config: ChunkingConfig,
    tokenizer: Optional[ITokenizer] = None,
) -> List[ChildChunkData]:
    """Split one parent chunk using the strategy named in ``config``."""
    if config.strategy == "fixed":
        return split_into_children_fixed(
            parent,
            _require_tokenizer(tokenizer),
            max_tokens=config.fixed_child_max_tokens,
            overlap_tokens=config.fixed_child_overlap_tokens,
        )
    return split_into_children(
        parent,
        max_chars=config.child_max_chars,
        overlap_chars=config.child_overlap_chars,
    )
