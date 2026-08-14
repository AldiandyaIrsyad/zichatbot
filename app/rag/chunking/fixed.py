"""Fixed-size token-window chunking — the domain-agnostic strategy.

The counterpart to :mod:`app.rag.chunking.logic`: instead of following the
document's heading structure (which encodes Indonesian legal vocabulary), the
whole document is flattened to a token stream and cut into fixed windows.
Parents are non-overlapping windows (overlap there would duplicate text in the
LLM's context); children are overlapping windows of each parent, matching the
ablation baseline in ``app/thesis/_eval/exp2a_chunking/run.py::fixed_chunks``.

Parent/child shape and the ltree fields are identical to the hierarchical
strategy, so the rest of the pipeline is unaffected. Chunks carry no
breadcrumbs (there is no heading structure to record), and page numbers are
tracked per token so citations still resolve to the right PDF page.

The tokenizer is injected as an :class:`ITokenizer` rather than imported, so
this module keeps the ``thesis/`` purity rule (no model/HTTP/DB imports).

Note: windows are cut on token ids and then decoded to text, and for a subword
tokenizer decode→re-encode is not exactly lossless — a chunk can re-encode a
few tokens over its nominal size (~1% with BGE-M3). Same behaviour as the
``exp2a_chunking`` baseline; harmless well below the model's 8192-token limit.
"""

import uuid
from typing import List, Optional

import structlog

from .config import (
    DEFAULT_FIXED_CHILD_MAX_TOKENS,
    DEFAULT_FIXED_CHILD_OVERLAP_TOKENS,
    DEFAULT_FIXED_PARENT_MAX_TOKENS,
    ITokenizer,
)
from .logic import MIN_CHILD_TEXT_LENGTH
from .models import ChildChunkData, ContentType, ParentChunkData, ParsedElement
from .router import IGNORE_ELEMENT_TYPES

logger = structlog.get_logger(__name__)


def _window_starts(total: int, size: int, step: int) -> List[int]:
    """Start offsets of the windows covering ``total`` tokens.

    Stops as soon as a window reaches the end, so the final (short) window is
    emitted once rather than repeated by the overlap step.
    """
    starts: List[int] = []
    for start in range(0, total, step):
        starts.append(start)
        if start + size >= total:
            break
    return starts


def create_parent_chunks_fixed(
    elements: List[ParsedElement],
    doc_id: str,
    tokenizer: ITokenizer,
    max_tokens: int = DEFAULT_FIXED_PARENT_MAX_TOKENS,
) -> List[ParentChunkData]:
    """Group parsed elements into fixed-size token-window parent chunks.

    Element texts are concatenated in reading order (tables and figures already
    carry Markdown / VLM-description text by the time the ingest worker calls
    this) and cut into non-overlapping windows of ``max_tokens``. Noise elements
    (``IGNORE_ELEMENT_TYPES``) are dropped first, mirroring the hierarchical
    chunker.

    Each parent gets ``breadcrumbs=[]``, ``depth=0``, ``path=doc_id`` and no
    ``parent_id`` — there is no hierarchy to record. ``page`` is the page of the
    element that the window starts in.

    Args:
        elements (List[ParsedElement]): Structured elements.
        doc_id (str): UUID of the source document.
        tokenizer (ITokenizer): Tokenizer defining the token unit (BGE-M3 in
            production, so window sizes match what the embedder actually sees).
        max_tokens (int): Maximum token length per parent chunk.

    Returns:
        List[ParentChunkData]: Ordered list of ParentChunkData.
    """
    if not elements:
        return []

    # Flatten to one token stream, remembering which page each token came from
    # so a window can be attributed to a page for citations.
    token_ids: List[int] = []
    page_per_token: List[Optional[int]] = []
    for element in elements:
        if element.element_type in IGNORE_ELEMENT_TYPES:
            continue
        text = element.text.strip()
        if not text:
            continue
        element_tokens = tokenizer.encode(text, add_special_tokens=False)
        if not element_tokens:
            continue
        page = element.metadata.get("page_number")
        token_ids.extend(element_tokens)
        page_per_token.extend([page] * len(element_tokens))

    if not token_ids:
        return []

    parent_chunks: List[ParentChunkData] = []
    for start in _window_starts(len(token_ids), max_tokens, max_tokens):
        window = token_ids[start:start + max_tokens]
        text = tokenizer.decode(window, skip_special_tokens=True).strip()
        if not text:
            continue

        chunk_index = len(parent_chunks)
        parent_chunks.append(
            ParentChunkData(
                id=str(uuid.uuid4()),
                doc_id=doc_id,
                text=text,
                chunk_index=chunk_index,
                page=page_per_token[start],
                breadcrumbs=[],
                content_type=ContentType.TEXT,
                parent_id=None,
                ordinal=chunk_index,
                path=doc_id,
                depth=0,
            )
        )

    logger.info(
        "thesis.chunking.parents_created",
        strategy="fixed",
        parent_count=len(parent_chunks),
        element_count=len(elements),
        token_count=len(token_ids),
        doc_id=doc_id,
    )
    return parent_chunks


def split_into_children_fixed(
    parent: ParentChunkData,
    tokenizer: ITokenizer,
    max_tokens: int = DEFAULT_FIXED_CHILD_MAX_TOKENS,
    overlap_tokens: int = DEFAULT_FIXED_CHILD_OVERLAP_TOKENS,
) -> List[ChildChunkData]:
    """Split a parent chunk into overlapping fixed-size token-window children.

    Content-type agnostic by design — unlike the hierarchical splitter there is
    no table/figure routing, since the point of this strategy is to ignore
    document structure. Children under :data:`MIN_CHILD_TEXT_LENGTH` characters
    are dropped (same gibberish filter as the hierarchical path), then ordinals
    and ltree paths are assigned to the survivors.
    """
    if overlap_tokens >= max_tokens:
        raise ValueError("overlap_tokens must be smaller than max_tokens")

    token_ids = tokenizer.encode(parent.text, add_special_tokens=False)
    step = max_tokens - overlap_tokens

    children: List[ChildChunkData] = []
    for start in _window_starts(len(token_ids), max_tokens, step):
        window = token_ids[start:start + max_tokens]
        text = tokenizer.decode(window, skip_special_tokens=True).strip()
        if len(text) < MIN_CHILD_TEXT_LENGTH:
            continue

        children.append(
            ChildChunkData(
                id=str(uuid.uuid4()),
                parent_chunk_id=parent.id,
                doc_id=parent.doc_id,
                text=text,
                page=parent.page,
                breadcrumbs=parent.breadcrumbs,
                content_type=parent.content_type,
            )
        )

    for i, child in enumerate(children):
        child.ordinal = i
        if parent.path:
            child.path = f"{parent.path}.c{i}"

    logger.debug(
        "thesis.chunking.children_created",
        strategy="fixed",
        parent_id=parent.id,
        child_count=len(children),
    )
    return children
