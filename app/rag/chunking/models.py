"""Data models for the hierarchical chunking pipeline.

Defines the structures flowing through the pipeline:
- ``ParsedElement``: output of the document parser.
- ``ParentChunkData``: logical sections (for LLM context).
- ``ChildChunkData``: sentence-level splits of each parent (for retrieval).

Each carries a ``content_type`` so downstream components (embedding, vector
store, retrieval, citation) can distinguish narrative text from tables and
VLM-enriched figure descriptions. Pure ``pydantic`` models, no infra imports —
part of the ``thesis/`` purity rule.
"""

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class ContentType(str, Enum):
    """The structural type of a chunk's content.

    TEXT: narrative prose. TABLE: HTML/Markdown table (must not be
    character-split). FIGURE: VLM-generated description of a visual element.
    HYBRID: a parent chunk mixing text with table/figure elements.
    """

    TEXT = "text"
    TABLE = "table"
    FIGURE = "figure"
    HYBRID = "hybrid"


class ParsedElement(BaseModel):
    """An element parsed from a document.

    ``element_type`` is the parser's type (e.g. "Title", "NarrativeText",
    "Table", "Image"); ``text`` is the content (HTML for tables, possibly empty
    for images until VLM enrichment); ``metadata`` holds parser metadata
    (page_number, category_depth, text_as_html, image_path, ...). ``content_type``
    defaults to TEXT and is set by the content-type router before chunking.
    """

    element_type: str
    text: str
    metadata: Dict[str, Any] = {}
    content_type: ContentType = ContentType.TEXT


class ParentChunkData(BaseModel):
    """A parent chunk — a logical section of the document.

    Parent chunks are the unit of context returned to the LLM at query time
    (Small-to-Big retrieval), preserving a section's full text including tables
    and figure descriptions. ``text`` is pure body text with no inline
    breadcrumb tag (the structured path lives in ``breadcrumbs``). ltree
    hierarchy fields: ``parent_id`` (None for root sections), ``ordinal``,
    ``path`` (e.g. "doc_id.bab_i.pasal_5"), and ``depth`` (0 = root).
    """

    id: str
    doc_id: str
    text: str
    chunk_index: int
    page: Optional[int] = None
    breadcrumbs: List[str] = []
    content_type: ContentType = ContentType.TEXT
    element_metadata: Dict[str, Any] = {}

    # ltree hierarchy fields
    parent_id: Optional[str] = None
    ordinal: int = 0
    path: str = ""
    depth: int = 0


class ChildChunkData(BaseModel):
    """A child chunk — a sentence-level split of a parent chunk.

    Child chunks are the unit of vector search: embedded and stored in Qdrant.
    When a child matches a query, its parent's full text is retrieved for LLM
    context. ``text`` is prefixed with a breadcrumb tag (e.g. "BAB II > Pasal
    5\n\n") when the parent has breadcrumbs — embedding-only, never shown to the
    user or LLM. Page, breadcrumbs, and content_type are inherited from the
    parent; ``path`` is the parent path + ".c" + ordinal.
    """

    id: str
    parent_chunk_id: str
    doc_id: str
    text: str
    page: Optional[int] = None
    breadcrumbs: List[str] = []
    content_type: ContentType = ContentType.TEXT

    # ltree hierarchy fields
    ordinal: int = 0
    path: str = ""
