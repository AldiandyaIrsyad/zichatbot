"""Chunking strategy configuration for the ingestion pipeline.

Two strategies are available:

- ``hierarchical`` (default): :mod:`app.rag.chunking.logic` — parent chunks
  follow the document's heading structure, children are character splits at
  sentence boundaries. Encodes Indonesian legal structure (BAB/Pasal) in
  ``infer_heading_depth``, so it is domain-aware.
- ``fixed``: :mod:`app.rag.chunking.fixed` — parent and child chunks are
  fixed-size BGE-M3 token windows, ignoring headings entirely. Fully
  domain-agnostic; the ablation baseline used by ``exp2a_chunking``.

Both produce the same parent+child shape, so nothing downstream (DB schema,
Small-to-Big parent expansion, reranking) changes between them.

Pure dataclass/Protocol definitions — no pydantic-settings or infra imports,
per the ``thesis/`` purity rule. ``app/kb/config.py::ChunkingSettings`` reads
the ``CHUNKING_*`` env vars and maps them onto :class:`ChunkingConfig`.
"""

from dataclasses import dataclass
from typing import List, Literal, Protocol, runtime_checkable

from .logic import (
    DEFAULT_CHILD_MAX_CHARS,
    DEFAULT_CHILD_OVERLAP_CHARS,
    DEFAULT_PARENT_MAX_CHARS,
)

# Defaults for the fixed strategy, in BGE-M3 tokens. Child sizes match the
# ablation baseline in app/thesis/_eval/exp2a_chunking/run.py::fixed_chunks.
DEFAULT_FIXED_PARENT_MAX_TOKENS = 2048
DEFAULT_FIXED_CHILD_MAX_TOKENS = 512
DEFAULT_FIXED_CHILD_OVERLAP_TOKENS = 64

ChunkingStrategy = Literal["hierarchical", "fixed"]


@runtime_checkable
class ITokenizer(Protocol):
    """Minimal tokenizer port used by the fixed strategy.

    Structural on purpose: a HuggingFace ``AutoTokenizer`` satisfies it as-is,
    so the infra layer can inject one without an adapter class and this module
    stays free of a ``transformers`` import.
    """

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """Encode text into token ids."""
        ...

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token ids back into text."""
        ...


@dataclass(frozen=True)
class ChunkingConfig:
    """Which chunking strategy to run, and the sizes it uses.

    Character sizes apply to ``hierarchical``; token sizes to ``fixed``. The
    unused set is simply ignored, so one config object covers both strategies.
    Defaults reproduce today's production behaviour exactly.
    """

    strategy: ChunkingStrategy = "hierarchical"

    # hierarchical (character-based)
    parent_max_chars: int = DEFAULT_PARENT_MAX_CHARS
    child_max_chars: int = DEFAULT_CHILD_MAX_CHARS
    child_overlap_chars: int = DEFAULT_CHILD_OVERLAP_CHARS
    # How a table parent is represented among its children:
    #   "rows"    — embed row groups (legacy)
    #   "summary" — embed only a description; the parent supplies the table
    #   "both"    — default
    table_child_mode: str = "both"

    # fixed (token-based)
    fixed_parent_max_tokens: int = DEFAULT_FIXED_PARENT_MAX_TOKENS
    fixed_child_max_tokens: int = DEFAULT_FIXED_CHILD_MAX_TOKENS
    fixed_child_overlap_tokens: int = DEFAULT_FIXED_CHILD_OVERLAP_TOKENS

    def __post_init__(self) -> None:
        if self.fixed_child_overlap_tokens >= self.fixed_child_max_tokens:
            raise ValueError(
                "fixed_child_overlap_tokens must be smaller than "
                "fixed_child_max_tokens (the sliding window would not advance)"
            )
