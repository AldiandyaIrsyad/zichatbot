"""Parent-child chunking strategy for Small-to-Big retrieval.

Organizes structured elements from the parser into a two-level hierarchy:
parent chunks (logical sections, for LLM context) and child chunks
(sentence-level splits, for retrieval precision).

Content-type aware:
- Text: RecursiveCharacterTextSplitter at sentence boundaries.
- Tables (HTML): stored whole — no character splitting, to preserve structure.
- Tables (Markdown): split by row groups, repeating the header in each child so
  every child is independently embeddable.
- Figures: VLM descriptions split at sentence boundaries if long.

Depends only on stdlib ``re``/``uuid`` and ``langchain_text_splitters`` (a pure
text-splitting utility) — no HTTP/DB imports, per the ``thesis/`` purity rule.
"""
import os
import re
import uuid
import structlog
from typing import Any, Dict, List, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter

from .models import ChildChunkData, ContentType, ParentChunkData, ParsedElement
from .router import (
    IGNORE_ELEMENT_TYPES,
    SECTION_BOUNDARY_TYPES,
    TABLE_ELEMENT_TYPES,
    FIGURE_ELEMENT_TYPES,
    classify_element,
)
from app.shared.table_converter import is_markdown_table, split_markdown_table_lines

logger = structlog.get_logger(__name__)

# Default chunking parameters
DEFAULT_PARENT_MAX_CHARS = 4096

# How a table parent is represented among its children. Overridable per-parent
# via element_metadata["table_child_mode"], or globally with CHUNKING_TABLE_CHILD_MODE.
TABLE_CHILD_MODE = os.getenv("CHUNKING_TABLE_CHILD_MODE", "both").strip().lower()
DEFAULT_CHILD_MAX_CHARS = 512
DEFAULT_CHILD_OVERLAP_CHARS = 50

# Minimum child text length — chunks shorter than this are treated as
# gibberish (no meaningful context) and dropped before embedding/upsert.
MIN_CHILD_TEXT_LENGTH = 8

# Matches Indonesian legal "ayat" markers like "(1)", "(2)". The hi_res layout
# model occasionally misclassifies these short numbered lines as "Title";
# without this guard they'd be treated as a section boundary, resetting the
# heading stack (infer_heading_depth has no pattern for a leading "(") and
# splitting ayat clauses of one Pasal into unrelated parent chunks.
_AYAT_MARKER_RE = re.compile(r"^\(\d+\)\s")


def infer_heading_depth(text: str, metadata: Dict[str, Any]) -> int:
    """Infer the hierarchical depth of a heading element (0 = root, higher =
    deeper). Uses ``metadata["category_depth"]`` when present, else heuristic
    pattern matching for Indonesian legal documents.
    """
    # Try parser-provided depth first
    category_depth = metadata.get("category_depth")
    if category_depth is not None:
        return int(category_depth)

    # Heuristic patterns for Indonesian legal documents
    text_stripped = text.strip()

    # BAB I, BAB II, BAB X (Roman numerals) → depth 0
    if re.match(r"^BAB\s+[IVXLC]+", text_stripped):
        return 0

    # Pasal 1, Pasal 5 → depth 1 (section-level)
    if re.match(r"^Pasal\s+\d+", text_stripped):
        return 1

    # A. Syarat, B. Ketentuan → depth 1
    if re.match(r"^[A-Z]\.\s", text_stripped):
        return 1

    # 1. Syarat, 2. Ketentuan → depth 2
    if re.match(r"^\d+\.\s", text_stripped):
        return 2

    # a) Dokumen, b) Persyaratan → depth 3
    if re.match(r"^[a-z]\)\s", text_stripped):
        return 3

    # 1) Dokumen, 2) Persyaratan → depth 4
    if re.match(r"^\d+\)\s", text_stripped):
        return 4

    # Default: treat as root
    return 0


def _slug(text: str) -> str:
    """Convert heading text to an ltree-safe slug: lowercase, spaces/punctuation
    to underscores, truncated to 50 chars, prefixed with ``h_`` if it starts
    with a digit (ltree labels can't). E.g. "bab_i", "pasal_5", "a_syarat".
    """
    slug = text.lower().strip()
    slug = re.sub(r"[^a-z0-9]+", "_", slug)
    slug = slug.strip("_")
    slug = slug[:50]
    if slug and slug[0].isdigit():
        slug = "h_" + slug
    return slug or "unnamed"


def create_parent_chunks(
    elements: List[ParsedElement],
    doc_id: str,
    max_chars: int = DEFAULT_PARENT_MAX_CHARS,
) -> List[ParentChunkData]:
    """Group parsed elements into logical parent chunks.

    Uses section boundary elements (Title) from the unstructured
    parser to create semantically meaningful parent chunks. Each parent
    chunk aggregates content under a section heading until the next heading
    or until the max character limit is reached.

    **Table-aware**: When a ``Table`` element is encountered, the current
    text buffer is flushed first (so the table is not mixed with prose),
    and the table becomes its own parent chunk with
    ``content_type=ContentType.TABLE``. This prevents the character
    splitter from fragmenting table HTML.

    Consecutive headers are grouped together, and page artifacts are ignored.
    Maintains a heading stack to track the hierarchical path (breadcrumbs).

    Args:
        elements (List[ParsedElement]): Structured elements.
        doc_id (str): UUID of the source document.
        max_chars (int): Maximum character length per parent chunk.

    Returns:
        List[ParentChunkData]: Ordered list of ParentChunkData.
    """
    if not elements:
        return []

    # Classify all elements by content type
    for el in elements:
        el.content_type = classify_element(el)

    parent_chunks: List[ParentChunkData] = []
    current_texts: List[str] = []
    current_length = 0
    chunk_index = 0
    current_page: Optional[int] = None
    current_fallback_page: Optional[int] = None
    has_body_text = False
    
    # Track the hierarchical path: [depth, title, ordinal, path, section_chunk_id]
    heading_stack: List[List[Any]] = []
    ordinal_counters: Dict[int, int] = {}
    current_breadcrumbs: List[str] = []

    def _flush_current() -> None:
        nonlocal current_texts, current_length, chunk_index, current_page, current_fallback_page, has_body_text
        if not current_texts:
            return
            
        combined_text = "\n\n".join(current_texts).strip()

        if combined_text:
            _parent_id = heading_stack[-2][4] if len(heading_stack) >= 2 else None
            _path = heading_stack[-1][3] if heading_stack else doc_id
            _depth = heading_stack[-1][0] if heading_stack else 0
            _chunk_id = str(uuid.uuid4())
            parent_chunks.append(
                ParentChunkData(
                    id=_chunk_id,
                    doc_id=doc_id,
                    text=combined_text,
                    chunk_index=chunk_index,
                    page=current_page if current_page is not None else current_fallback_page,
                    breadcrumbs=list(current_breadcrumbs),
                    content_type=ContentType.TEXT,
                    parent_id=_parent_id,
                    ordinal=chunk_index,
                    path=_path,
                    depth=_depth,
                )
            )
            if heading_stack and heading_stack[-1][4] is None:
                heading_stack[-1][4] = _chunk_id
            chunk_index += 1

        current_texts = []
        current_length = 0
        current_page = None
        current_fallback_page = None
        has_body_text = False

    def _flush_table(element: ParsedElement) -> None:
        """Create a standalone parent chunk for a table element."""
        nonlocal chunk_index
        text = element.text.strip()
        if not text:
            return

        _parent_id = heading_stack[-2][4] if len(heading_stack) >= 2 else None
        _path = heading_stack[-1][3] if heading_stack else doc_id
        _depth = heading_stack[-1][0] if heading_stack else 0
        _chunk_id = str(uuid.uuid4())
        parent_chunks.append(
            ParentChunkData(
                id=_chunk_id,
                doc_id=doc_id,
                text=text,
                chunk_index=chunk_index,
                page=element.metadata.get("page_number"),
                breadcrumbs=list(current_breadcrumbs),
                content_type=ContentType.TABLE,
                element_metadata=dict(element.metadata),
                parent_id=_parent_id,
                ordinal=chunk_index,
                path=_path,
                depth=_depth,
            )
        )
        if heading_stack and heading_stack[-1][4] is None:
            heading_stack[-1][4] = _chunk_id
        chunk_index += 1

    def _flush_figure(element: ParsedElement) -> None:
        """Create a standalone parent chunk for a figure/VLM description."""
        nonlocal chunk_index
        text = element.text.strip()
        if not text:
            return

        _parent_id = heading_stack[-2][4] if len(heading_stack) >= 2 else None
        _path = heading_stack[-1][3] if heading_stack else doc_id
        _depth = heading_stack[-1][0] if heading_stack else 0
        _chunk_id = str(uuid.uuid4())
        parent_chunks.append(
            ParentChunkData(
                id=_chunk_id,
                doc_id=doc_id,
                text=text,
                chunk_index=chunk_index,
                page=element.metadata.get("page_number"),
                breadcrumbs=list(current_breadcrumbs),
                content_type=ContentType.FIGURE,
                element_metadata=dict(element.metadata),
                parent_id=_parent_id,
                ordinal=chunk_index,
                path=_path,
                depth=_depth,
            )
        )
        if heading_stack and heading_stack[-1][4] is None:
            heading_stack[-1][4] = _chunk_id
        chunk_index += 1

    for element in elements:
        text = element.text.strip()
        if not text:
            continue

        # Ignore noisy elements like page headers/footers
        if element.element_type in IGNORE_ELEMENT_TYPES:
            continue

        # --- Table routing: flush current prose, emit table as own parent ---
        if element.content_type == ContentType.TABLE:
            _flush_current()
            _flush_table(element)
            continue

        # --- Figure routing: flush current prose, emit figure as own parent ---
        if element.content_type == ContentType.FIGURE:
            _flush_current()
            _flush_figure(element)
            continue

        # Treat as boundary only if it's a Title and has substantial text
        # (filters out 1-letter artifacts like bullets misclassified as Titles).
        # Ayat markers are excluded even if mistagged as Title — they are
        # never legitimate section boundaries (see _AYAT_MARKER_RE).
        is_boundary = (
            element.element_type in SECTION_BOUNDARY_TYPES
            and len(text) > 3
            and not _AYAT_MARKER_RE.match(text)
        )

        if is_boundary:
            # Start a new parent chunk at section boundaries, but only if we already
            # have body text in the current chunk. This prevents consecutive headers
            # from being split into separate tiny chunks.
            if has_body_text:
                _flush_current()

            # Update heading stack AFTER flushing the previous section
            depth = infer_heading_depth(text, element.metadata)

            # Pop elements from stack that are at the same or deeper level
            while heading_stack and heading_stack[-1][0] >= depth:
                heading_stack.pop()

            # Reset ordinal counters for deeper levels
            for d in list(ordinal_counters):
                if d > depth:
                    ordinal_counters[d] = 0

            # Compute ordinal and path
            ordinal_counters[depth] = ordinal_counters.get(depth, 0) + 1
            _ordinal = ordinal_counters[depth]
            _parent_path = heading_stack[-1][3] if heading_stack else doc_id
            _heading_path = f"{_parent_path}.{_slug(text)}"

            heading_stack.append([depth, text, _ordinal, _heading_path, None])
            current_breadcrumbs = [h[1] for h in heading_stack]

        # If adding this element would exceed the limit, flush first
        if current_length + len(text) > max_chars and current_texts:
            _flush_current()

        # Titles ARE kept in body text (helps readability)
        current_texts.append(text)
        current_length += len(text)

        if current_fallback_page is None:
            current_fallback_page = element.metadata.get("page_number")

        # Prefer the first *body* element's page over a heading's: headings
        # often sit at the bottom of one PDF page while their body starts on the
        # next, which would otherwise mis-attribute the chunk (and citation).
        if not is_boundary and current_page is None:
            current_page = element.metadata.get("page_number")

        if not is_boundary:
            has_body_text = True

    # Don't forget the last accumulated chunk
    _flush_current()

    logger.info(
        "thesis.chunking.parents_created",
        parent_count=len(parent_chunks),
        element_count=len(elements),
        doc_id=doc_id,
    )
    return parent_chunks


def split_into_children(
    parent: ParentChunkData,
    max_chars: int = DEFAULT_CHILD_MAX_CHARS,
    overlap_chars: int = DEFAULT_CHILD_OVERLAP_CHARS,
) -> List[ChildChunkData]:
    """Split a parent chunk into sentence-level child chunks.

    Content-type aware dispatcher routing on ``parent.content_type``:
    TEXT/HYBRID → :func:`_split_text_children` (RecursiveCharacterTextSplitter,
    sentence boundaries); TABLE → :func:`_split_table_children` (Markdown:
    row-group splitting with header repetition; HTML: single child, no split);
    FIGURE → :func:`_split_figure_children` (sentence split on the VLM
    description). Children carry pure body text — the hierarchical breadcrumbs
    are kept structurally (``breadcrumbs``) and appended post-retrieval by the
    search service, not embedded into the child vector.
    """
    if parent.content_type == ContentType.TABLE:
        children = _split_table_children(parent, max_chars)
    elif parent.content_type == ContentType.FIGURE:
        children = _split_figure_children(parent, max_chars, overlap_chars)
    else:
        # TEXT and HYBRID both use the standard text splitter
        children = _split_text_children(parent, max_chars, overlap_chars)

    # Drop gibberish children: chunks whose body text is below the minimum
    # threshold carry no meaningful context. Filters micro-fragments (lone
    # punctuation, single letters, OCR noise) before embedding/upsert.
    original_count = len(children)
    children = [
        child for child in children
        if len(child.text.strip()) >= MIN_CHILD_TEXT_LENGTH
    ]
    dropped = original_count - len(children)
    if dropped > 0:
        logger.debug(
            "thesis.chunking.gibberish_filtered",
            parent_id=parent.id,
            dropped=dropped,
            remaining=len(children),
        )

    # Assign ordinals and ltree paths to surviving children
    for i, child in enumerate(children):
        child.ordinal = i
        if parent.path:
            child.path = f"{parent.path}.c{i}"

    logger.debug(
        "thesis.chunking.children_created",
        parent_id=parent.id,
        child_count=len(children),
        content_type=parent.content_type.value,
    )
    return children


def _build_breadcrumb_tag(breadcrumbs: List[str]) -> str:
    """Build the breadcrumb path string used for post-retrieval context.

    Kept as a shared helper (and for ``_strip_breadcrumb_tag`` compatibility
    with the E1 dilution probe and tests). Children no longer embed this tag at
    ingestion time; the search service appends it to the retrieved parent text.
    Returns "" if there are no breadcrumbs.
    """
    if not breadcrumbs:
        return ""
    return f"{' > '.join(breadcrumbs)}\n\n"


def _strip_breadcrumb_tag(text: str, breadcrumbs: List[str]) -> str:
    """Return a child chunk's body text with its breadcrumb tag removed (if
    present).
    """
    tag = _build_breadcrumb_tag(breadcrumbs)
    if tag and text.startswith(tag):
        return text[len(tag):]
    return text


def _split_text_children(
    parent: ParentChunkData,
    max_chars: int = DEFAULT_CHILD_MAX_CHARS,
    overlap_chars: int = DEFAULT_CHILD_OVERLAP_CHARS,
) -> List[ChildChunkData]:
    """Split narrative text into child chunks using RecursiveCharacterTextSplitter.

    Respects sentence and word boundaries. Children contain only body text —
    no breadcrumb tag — so embeddings stay discriminative; the hierarchy is
    appended post-retrieval.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chars,
        chunk_overlap=overlap_chars,
        separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""],
        length_function=len,
    )

    child_texts = splitter.split_text(parent.text)

    children: List[ChildChunkData] = []
    for child_text in child_texts:
        child_text = child_text.strip()
        if not child_text:
            continue

        children.append(
            ChildChunkData(
                id=str(uuid.uuid4()),
                parent_chunk_id=parent.id,
                doc_id=parent.doc_id,
                text=child_text,
                page=parent.page,
                breadcrumbs=parent.breadcrumbs,
                content_type=parent.content_type,
            )
        )
    return children



# Indonesian legal drafting introduces every abbreviation explicitly, e.g.
# "Uang Kuliah Tunggal yang selanjutnya disingkat UKT adalah ...". That sentence
# is the document's own authority on which acronym its readers — and therefore
# its askers — will use, so acronyms are mined from it rather than guessed from
# capital letters (initialising the title's word runs yields KTU/TUK noise).
_ACRONYM_DEF_RE = re.compile(
    r"([A-Z][A-Za-z/\-]*(?:\s+[A-Za-z/\-]+){0,6}?)\s+yang\s+selanjutnya\s+"
    r"(?:disingkat|disebut)\s+(?:dengan\s+)?([A-Z][A-Za-z]{1,9})\b"
)

# Cells like "Rp. 3.390.000" / "3.390.000" — enough to say the table quantifies
# money without hard-coding this corpus's column names.
_MONEY_CELL_RE = re.compile(r"(?:Rp\.?\s*)?\d{1,3}(?:\.\d{3}){2,}")


def extract_acronym_definitions(text: str) -> Dict[str, str]:
    """Map long form -> acronym for every abbreviation a document defines.

    Used to make a table summary searchable by the name a question actually
    uses. The tariff table's title says "Uang Kuliah Tunggal" while questions
    say "UKT", and a reranker scores the un-expanded summary at 0.002 against
    such a question — effectively unreachable.
    """
    out: Dict[str, str] = {}
    for long_form, acronym in _ACRONYM_DEF_RE.findall(text or ""):
        long_form = long_form.strip()
        # A long form shorter than its acronym is a mis-capture, not a definition.
        if len(long_form) > len(acronym):
            out.setdefault(long_form, acronym)
    return out


# Words that never carry an initial in an Indonesian acronym, so they break a
# run rather than contributing a letter ("Uang Kuliah Tunggal" -> UKT, and the
# preceding "dan" keeps the run from swallowing the previous clause).
_ACRONYM_STOPWORDS = frozenset(
    {"dan", "atau", "bagi", "di", "ke", "dari", "untuk", "yang", "pada", "the"}
)


def infer_acronyms(doc_title: str, doc_text: str) -> Dict[str, str]:
    """Map long form -> acronym for a document, by definition then by usage.

    Two strategies, because the documents that most need this do not define
    their terms. The tariff schedule (``003 Tahun 2022``) writes "UKT" nine
    times and never once says "yang selanjutnya disingkat", so
    :func:`extract_acronym_definitions` alone leaves its table unsearchable by
    the acronym every question uses.

    The fallback initialises runs of capitalised words in the title and keeps
    only those whose initials actually occur in the document. That verification
    is what makes it safe: "Uang Kuliah Tunggal" yields UKT, which appears, so
    it is kept; the overlapping "Kelompok Tarif Uang" yields KTU, which does
    not appear, so the noise is discarded.
    """
    acronyms = extract_acronym_definitions(doc_text)

    words = re.findall(r"[\w/]+", doc_title or "")
    runs: List[List[str]] = []
    current: List[str] = []
    for word in words:
        if word[:1].isupper() and word.lower() not in _ACRONYM_STOPWORDS:
            current.append(word)
        else:
            if len(current) > 1:
                runs.append(current)
            current = []
    if len(current) > 1:
        runs.append(current)

    for run in runs:
        for size in range(2, min(len(run), 5) + 1):
            for start in range(len(run) - size + 1):
                window = run[start : start + size]
                candidate = "".join(w[0].upper() for w in window)
                long_form = " ".join(window)
                if long_form in acronyms or candidate in acronyms.values():
                    continue
                if re.search(rf"\b{re.escape(candidate)}\b", doc_text or ""):
                    acronyms[long_form] = candidate
    return acronyms


def derive_table_summary(
    table_text: str,
    breadcrumbs: Optional[List[str]] = None,
    max_labels: int = 40,
    doc_title: str = "",
    acronyms: Optional[Dict[str, str]] = None,
) -> str:
    """Describe what a table enumerates, for embedding in place of its cells.

    A table's cells are a poor search target: "berapa UKT untuk Pendidikan
    Sejarah" has to match a row that is mostly digits and column labels. This
    builds a sentence out of the parts that *do* carry meaning — the section
    heading, the column headers, and the row labels — so the vector describes
    the table's subject rather than its numbers. The full table still reaches
    the LLM, because retrieval hydrates the parent (see
    ``search_service`` step 6).

    Deterministic and free: no model call. Intended as the baseline a VLM-written
    summary has to beat.
    """
    lines = [ln.strip() for ln in table_text.splitlines() if ln.strip().startswith("|")]
    if not lines:
        return ""

    def cells(line: str) -> List[str]:
        return [c.strip() for c in line.strip().strip("|").split("|")]

    # Header = first row that is not the --- separator.
    header: List[str] = []
    for line in lines[:3]:
        if not all(c in "|:- " for c in line):
            header = [c for c in cells(line) if c and c != "[merged]"]
            break

    # Row labels = the descriptive cells of each body row: the key a question
    # actually names ("Pendidikan Sejarah"), not its values. Taking merely the
    # first non-numeric cell picks up the code column instead ("B025"), because
    # these tables run | Unit Kerja | Kode | Jenjang | Departemen/Program Studi |
    # — so prefer the longest text cell in the row and keep short codes out.
    labels: List[str] = []
    for line in lines[1:]:
        if all(c in "|:- " for c in line):
            continue
        candidates = [
            cell
            for cell in cells(line)
            if cell
            and cell != "[merged]"
            and not re.fullmatch(r"[\d.,\s]+", cell)
            and not re.fullmatch(r"[A-Z]?\d{2,4}[A-Z]?", cell)   # codes: B025, A015
            and not re.fullmatch(r"[SD]\d", cell)                # jenjang: S1, D3
            and cell not in header
            and len(cell) >= 6
        ]
        if candidates:
            labels.append(max(candidates, key=len))
        if len(labels) >= max_labels:
            break

    # Lead with a sentence phrased the way a question is, not a schema dump.
    # "Kolom: Unit Kerja, Kode, Jenjang..." shares almost no vocabulary with
    # "berapa biaya UKT?", so a general question could not reach the table even
    # when its document ranked first. The document title carries the subject
    # ("Kelompok Tarif Uang Kuliah Tunggal"), and the header supplies what the
    # table quantifies; both are what a reader would name.
    subject = doc_title.strip() or (" > ".join(breadcrumbs) if breadcrumbs else "")
    lead_bits = ["Tabel"]
    if subject:
        lead_bits.append(subject)
    else:
        # No title to name the subject, so fall back to the longest header cell,
        # which is the closest thing the table itself offers. With a title this
        # clause is skipped: it added "— daftar Unit Kerja" (merely the first
        # column) to a title that already said "Kelompok Tarif Uang Kuliah
        # Tunggal".
        longest = max(
            (h for h in header if not re.fullmatch(r"[\d.,\s]+", h)),
            key=len,
            default="",
        )
        if longest:
            lead_bits.append(f"daftar {longest}")
    lead = " ".join(lead_bits).strip()

    # Name the subject the way a question names it. The title spells the term
    # out ("...Tarif Uang Kuliah Tunggal") but questions use the acronym
    # ("berapa biaya UKT"), and without the expansion the two share almost no
    # vocabulary. Measured on bge-reranker-v2-m3 against "berapa biaya UKT":
    # 0.0019 without, 0.33 with this and the cost clause below.
    for long_form, acronym in (acronyms or {}).items():
        if long_form and long_form in lead and acronym not in lead:
            lead = lead.replace(long_form, f"{long_form} ({acronym})", 1)

    if labels:
        lead += f". Memuat {len(dict.fromkeys(labels))} baris"
    lead += "."

    parts: List[str] = [lead]

    # A table of money is what "berapa biaya/tarif ...?" is asking for, but the
    # cells are digits and the headers say "Kelompok 1". Say so in words, once,
    # only when the cells actually carry money.
    if _MONEY_CELL_RE.search(table_text):
        parts.append("Memuat besaran biaya atau tarif dalam Rupiah.")
    if breadcrumbs and doc_title:
        parts.append(" > ".join(breadcrumbs))
    if header:
        parts.append("Kolom: " + ", ".join(dict.fromkeys(header)))
    if labels:
        parts.append("Baris: " + ", ".join(dict.fromkeys(labels)))
    return " ".join(p for p in parts if p).strip()


def _split_table_children(
    parent: ParentChunkData,
    max_chars: int = DEFAULT_CHILD_MAX_CHARS,
) -> List[ChildChunkData]:
    """Create child chunk(s) for a table parent, dispatching on Markdown vs HTML.

    **Markdown tables** (``| col | col |``): stored as a single child if they
    fit ``max_chars``; otherwise split into row-group children, each repeating
    the header so every child is independently embeddable.

    **HTML tables** (legacy): stored as a single child without splitting —
    splitting HTML at row boundaries is fragile (unclosed tags, colspan/
    rowspan). Large HTML tables are better converted to Markdown upstream (see
    :mod:`shared.table_converter`).

    If a table summary is available in ``element_metadata``, it's appended as an
    extra child — the summary is vector-searched while the full table (parent)
    is retrieved for LLM context. Returns at least one child.
    """
    children: List[ChildChunkData] = []

    table_text = parent.text
    if not table_text.strip():
        return children

    # --- Detect whether this is a Markdown or HTML table ---
    is_markdown = _is_markdown_table(table_text)

    if is_markdown and len(table_text) > max_chars:
        # Large Markdown table → row-group splitting
        row_group_children = _split_markdown_table_rows(
            parent=parent,
            raw_table=table_text,
            max_chars=max_chars,
        )
        children.extend(row_group_children)
    else:
        # Small Markdown table OR HTML table → single child (no splitting)
        children.append(
            ChildChunkData(
                id=str(uuid.uuid4()),
                parent_chunk_id=parent.id,
                doc_id=parent.doc_id,
                text=table_text,
                page=parent.page,
                breadcrumbs=parent.breadcrumbs,
                content_type=ContentType.TABLE,
            )
        )

    # Summary child: embedded in place of (or alongside) the rows, while the
    # parent still supplies the full table at retrieval time. Falls back to a
    # derived summary when no producer set one, so the behaviour does not depend
    # on the parser having provided a caption.
    table_summary = parent.element_metadata.get("table_summary")
    if not (isinstance(table_summary, str) and table_summary.strip()):
        raw_acronyms = parent.element_metadata.get("acronyms")
        table_summary = derive_table_summary(
            table_text,
            parent.breadcrumbs,
            doc_title=str(parent.element_metadata.get("doc_title") or ""),
            acronyms=raw_acronyms if isinstance(raw_acronyms, dict) else None,
        )

    # TABLE_CHILD_MODE: "rows" (legacy), "summary" (embed only the description),
    # "both" (default). Summary-only makes the table's subject the search target
    # instead of its digits.
    mode = (parent.element_metadata.get("table_child_mode") or TABLE_CHILD_MODE).lower()
    if mode == "summary" and table_summary.strip():
        children = []
    if table_summary and table_summary.strip() and mode in ("summary", "both"):
        summary_text = table_summary.strip()
        children.append(
            ChildChunkData(
                id=str(uuid.uuid4()),
                parent_chunk_id=parent.id,
                doc_id=parent.doc_id,
                text=summary_text,
                page=parent.page,
                breadcrumbs=parent.breadcrumbs,
                content_type=ContentType.TABLE,
            )
        )

    return children


# Re-exported for backwards compatibility — the implementation now lives in
# table_converter, shared with app.guardrails.ram's citation-verification path.
_is_markdown_table = is_markdown_table


def _split_markdown_table_rows(
    parent: ParentChunkData,
    raw_table: str,
    max_chars: int,
) -> List[ChildChunkData]:
    """Split a large Markdown table into row-group child chunks.

    Extracts the header row (line 0) and separator row (line 1), then
    groups the remaining data rows into batches such that each batch
    (header + separator + rows) fits within ``max_chars``. Every child
    chunk repeats the header so it can be embedded independently.

    If the table cannot be parsed (fewer than 2 lines, no separator),
    falls back to a single child chunk with the full table text.

    Args:
        parent: The table parent chunk.
        raw_table: Raw Markdown table text.
        max_chars: Maximum characters per child chunk.

    Returns:
        List of child chunks, each containing header + data row subset.
    """
    # Validate Markdown structure: need at least header + separator + 1 data row
    parsed = split_markdown_table_lines(raw_table)
    if parsed is None:
        return [
            ChildChunkData(
                id=str(uuid.uuid4()),
                parent_chunk_id=parent.id,
                doc_id=parent.doc_id,
                text=raw_table,
                page=parent.page,
                breadcrumbs=parent.breadcrumbs,
                content_type=ContentType.TABLE,
            )
        ]

    header_line, separator_line, data_lines = parsed  # | Col A | Col B | / | --- | --- | / remaining rows

    # Base overhead: header + separator + two newlines
    base_overhead = len(header_line) + len(separator_line) + 2

    children: List[ChildChunkData] = []
    current_rows: List[str] = []
    current_len = base_overhead

    def _flush_group(rows: List[str]) -> None:
        if not rows:
            return
        group_text = "\n".join([header_line, separator_line] + rows)
        children.append(
            ChildChunkData(
                id=str(uuid.uuid4()),
                parent_chunk_id=parent.id,
                doc_id=parent.doc_id,
                text=group_text,
                page=parent.page,
                breadcrumbs=parent.breadcrumbs,
                content_type=ContentType.TABLE,
            )
        )

    for row in data_lines:
        row_len = len(row) + 1  # +1 for newline
        if current_rows and current_len + row_len > max_chars:
            _flush_group(current_rows)
            current_rows = [row]
            current_len = base_overhead + row_len
        else:
            current_rows.append(row)
            current_len += row_len

    # Flush remaining rows
    _flush_group(current_rows)

    # Edge case: no children were created (all data rows were empty)
    if not children:
        return [
            ChildChunkData(
                id=str(uuid.uuid4()),
                parent_chunk_id=parent.id,
                doc_id=parent.doc_id,
                text=raw_table,
                page=parent.page,
                breadcrumbs=parent.breadcrumbs,
                content_type=ContentType.TABLE,
            )
        ]

    return children


def _split_figure_children(
    parent: ParentChunkData,
    max_chars: int = DEFAULT_CHILD_MAX_CHARS,
    overlap_chars: int = DEFAULT_CHILD_OVERLAP_CHARS,
) -> List[ChildChunkData]:
    """Split a VLM figure description into child chunks.

    Figure descriptions are natural-language text, so they split at sentence
    boundaries like narrative text. But if the description fits in a single
    child (the common case), no splitting is applied — preserving the full
    description as one retrievable unit.
    """
    # If the description fits in a single child, don't split — preserve the
    # full VLM description as one retrievable unit.
    if len(parent.text) <= max_chars:
        return [
            ChildChunkData(
                id=str(uuid.uuid4()),
                parent_chunk_id=parent.id,
                doc_id=parent.doc_id,
                text=parent.text,
                page=parent.page,
                breadcrumbs=parent.breadcrumbs,
                content_type=ContentType.FIGURE,
            )
        ]

    # Long descriptions: fall back to text splitter
    return _split_text_children(parent, max_chars, overlap_chars)
