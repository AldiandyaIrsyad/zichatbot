import pytest
from app.rag.chunking.logic import (
    create_parent_chunks,
    split_into_children,
    infer_heading_depth,
    _build_breadcrumb_tag,
    _slug,
)
from app.rag.chunking.models import ParsedElement, ParentChunkData, ContentType


# ---------------------------------------------------------------------------
# infer_heading_depth tests
# ---------------------------------------------------------------------------

class TestInferHeadingDepth:
    """Tests for heuristic heading depth inference."""

    def test_category_depth_from_metadata(self):
        """If category_depth is in metadata, use it directly."""
        assert infer_heading_depth("Anything", {"category_depth": 3}) == 3
        assert infer_heading_depth("Anything", {"category_depth": 0}) == 0

    def test_bab_roman_numeral_depth_0(self):
        """BAB followed by Roman numeral → depth 0."""
        assert infer_heading_depth("BAB I", {}) == 0
        assert infer_heading_depth("BAB II", {}) == 0
        assert infer_heading_depth("BAB X", {}) == 0
        assert infer_heading_depth("BAB XIV", {}) == 0

    def test_pasal_depth_1(self):
        """Pasal N → depth 1."""
        assert infer_heading_depth("Pasal 1", {}) == 1
        assert infer_heading_depth("Pasal 5", {}) == 1
        assert infer_heading_depth("Pasal 100", {}) == 1

    def test_uppercase_letter_dot_depth_1(self):
        """A. Something → depth 1."""
        assert infer_heading_depth("A. Syarat", {}) == 1
        assert infer_heading_depth("B. Ketentuan", {}) == 1
        assert infer_heading_depth("Z. Last", {}) == 1

    def test_number_dot_depth_2(self):
        """1. Something → depth 2."""
        assert infer_heading_depth("1. Syarat", {}) == 2
        assert infer_heading_depth("2. Ketentuan", {}) == 2
        assert infer_heading_depth("10. Item", {}) == 2

    def test_lowercase_paren_depth_3(self):
        """a) Something → depth 3."""
        assert infer_heading_depth("a) Dokumen", {}) == 3
        assert infer_heading_depth("b) Persyaratan", {}) == 3

    def test_number_paren_depth_4(self):
        """1) Something → depth 4."""
        assert infer_heading_depth("1) Dokumen", {}) == 4
        assert infer_heading_depth("2) Persyaratan", {}) == 4

    def test_default_depth_0(self):
        """Unrecognized patterns → depth 0."""
        assert infer_heading_depth("Some Random Title", {}) == 0
        assert infer_heading_depth("Introduction", {}) == 0

    def test_category_depth_takes_priority_over_heuristic(self):
        """category_depth should win even if text matches a heuristic pattern."""
        # "BAB I" would be depth 0 by heuristic, but metadata says 2
        assert infer_heading_depth("BAB I", {"category_depth": 2}) == 2


# ---------------------------------------------------------------------------
# _slug tests
# ---------------------------------------------------------------------------

class TestSlug:
    """Tests for the _slug helper."""

    def test_basic_slug(self):
        assert _slug("BAB I") == "bab_i"

    def test_pasal_slug(self):
        assert _slug("Pasal 5") == "pasal_5"

    def test_punctuation_replaced(self):
        assert _slug("A. Syarat & Ketentuan") == "a_syarat_ketentuan"

    def test_digit_prefix_gets_h_prefix(self):
        """Slugs starting with a digit get 'h_' prefix."""
        assert _slug("1. Syarat") == "h_1_syarat"

    def test_empty_text(self):
        assert _slug("") == "unnamed"

    def test_truncation(self):
        long_text = "A" * 100
        slug = _slug(long_text)
        assert len(slug) <= 50


# ---------------------------------------------------------------------------
# Heading stack + breadcrumbs + ltree paths
# ---------------------------------------------------------------------------

def test_heading_stack_and_breadcrumbs():
    elements = [
        ParsedElement(element_type="Title", text="Chapter 1", metadata={"category_depth": 0}),
        ParsedElement(element_type="Text", text="Introduction to Chapter 1."),
        ParsedElement(element_type="Title", text="Section 1.1", metadata={"category_depth": 1}),
        ParsedElement(element_type="Text", text="This is the first section."),
        ParsedElement(element_type="Title", text="Article 1.1.1", metadata={"category_depth": 2}),
        ParsedElement(element_type="Text", text="Here is a deep article rule."),
        # Sibling section
        ParsedElement(element_type="Title", text="Section 1.2", metadata={"category_depth": 1}),
        ParsedElement(element_type="Text", text="This is the second section."),
    ]
    
    parents = create_parent_chunks(elements, doc_id="doc1", max_chars=1000)
    
    # Expecting chunks: 
    # 1. Chapter 1 (flushed when Section 1.1 title is found)
    # 2. Section 1.1 (flushed when Article 1.1.1 is found)
    # 3. Article 1.1.1 (flushed when Section 1.2 is found)
    # 4. Section 1.2 (flushed at end)
    assert len(parents) == 4
    
    # 1. Chapter 1
    assert parents[0].breadcrumbs == ["Chapter 1"]
    assert "[Context:" not in parents[0].text

    # 2. Section 1.1
    assert parents[1].breadcrumbs == ["Chapter 1", "Section 1.1"]
    assert "[Context:" not in parents[1].text

    # 3. Article 1.1.1
    assert parents[2].breadcrumbs == ["Chapter 1", "Section 1.1", "Article 1.1.1"]
    assert "[Context:" not in parents[2].text

    # 4. Section 1.2
    assert parents[3].breadcrumbs == ["Chapter 1", "Section 1.2"]
    assert "[Context:" not in parents[3].text


def test_build_breadcrumb_tag_has_no_bracket_boilerplate():
    """The child-chunk breadcrumb tag is a plain breadcrumb path — no
    "[Context: ...]" brackets or label, since that boilerplate is constant
    across nearly every chunk and would dilute embeddings without adding
    discriminative signal.
    """
    assert _build_breadcrumb_tag([]) == ""
    assert _build_breadcrumb_tag(["BAB II", "Pasal 5"]) == "BAB II > Pasal 5\n\n"


def test_parent_text_never_carries_inline_breadcrumb_tag():
    """Parent chunk text must stay pure body text even with deep
    breadcrumbs — the LLM prompt and frontend already build their own
    breadcrumb header from the structured ``breadcrumbs`` field, and RAM's
    sentence-splitter would otherwise chop an inline tag into its own
    pseudo-"sentence", polluting NLI premise windows.
    """
    elements = [
        ParsedElement(element_type="Title", text="BAB II", metadata={"category_depth": 0}),
        ParsedElement(element_type="Title", text="Pasal 5", metadata={"category_depth": 1}),
        ParsedElement(element_type="NarrativeText", text="Isi pasal lima."),
    ]

    parents = create_parent_chunks(elements, doc_id="doc1", max_chars=1000)

    assert len(parents) == 1
    assert parents[0].breadcrumbs == ["BAB II", "Pasal 5"]
    assert parents[0].text == "BAB II\n\nPasal 5\n\nIsi pasal lima."
    assert "[Context:" not in parents[0].text
    assert "Isi pasal lima" in parents[0].text


def test_ayat_marker_mistagged_as_title_does_not_split_pasal():
    """A parenthesized ayat marker like "(1)" must never act as a section
    boundary, even if the unstructured parser mistags it as a Title element
    (a known hi_res layout-detection false positive for short numbered
    lines). Otherwise infer_heading_depth falls through to the default
    depth 0, which pops the whole heading stack (including the current
    Pasal) and splits complementary ayat clauses of one article into
    unrelated parent chunks.
    """
    elements = [
        ParsedElement(element_type="Title", text="Pasal 5", metadata={"category_depth": 1}),
        ParsedElement(
            element_type="Title",
            text="(1) Pengelola tidak bertanggung jawab atas kehilangan kendaraan yang tidak berbayar.",
        ),
        ParsedElement(
            element_type="Title",
            text="(2) Pengelola hanya bertanggung jawab atas kehilangan kendaraan yang berbayar atau berlangganan.",
        ),
    ]

    parents = create_parent_chunks(elements, doc_id="doc1", max_chars=1000)

    assert len(parents) == 1
    assert parents[0].breadcrumbs == ["Pasal 5"]
    assert "(1) Pengelola tidak bertanggung jawab" in parents[0].text
    assert "(2) Pengelola hanya bertanggung jawab" in parents[0].text


def test_parent_chunks_have_ltree_paths():
    """Parent chunks should carry ltree-style paths and depth fields."""
    elements = [
        ParsedElement(element_type="Title", text="BAB I", metadata={}),
        ParsedElement(element_type="Text", text="Content of BAB I."),
        ParsedElement(element_type="Title", text="Pasal 1", metadata={}),
        ParsedElement(element_type="Text", text="Content of Pasal 1."),
        ParsedElement(element_type="Title", text="Pasal 2", metadata={}),
        ParsedElement(element_type="Text", text="Content of Pasal 2."),
    ]

    parents = create_parent_chunks(elements, doc_id="doc1", max_chars=1000)

    assert len(parents) == 3

    # Root: BAB I → depth 0, path starts with doc_id
    assert parents[0].depth == 0
    assert parents[0].path == "doc1.bab_i"
    assert parents[0].parent_id is None

    # Pasal 1 → depth 1, path = doc1.bab_i.pasal_1
    assert parents[1].depth == 1
    assert parents[1].path == "doc1.bab_i.pasal_1"
    assert parents[1].parent_id == parents[0].id

    # Pasal 2 → depth 1 (sibling of Pasal 1), path = doc1.bab_i.pasal_2
    assert parents[2].depth == 1
    assert parents[2].path == "doc1.bab_i.pasal_2"
    assert parents[2].parent_id == parents[0].id


def test_heuristic_depth_inference_no_metadata():
    """When category_depth is missing, heuristic patterns should infer depth."""
    elements = [
        ParsedElement(element_type="Title", text="BAB I", metadata={}),
        ParsedElement(element_type="Text", text="Intro to BAB I."),
        ParsedElement(element_type="Title", text="A. Syarat Umum", metadata={}),
        ParsedElement(element_type="Text", text="Some requirements."),
        ParsedElement(element_type="Title", text="1. Dokumen", metadata={}),
        ParsedElement(element_type="Text", text="Document details."),
        ParsedElement(element_type="Title", text="a) Paspor", metadata={}),
        ParsedElement(element_type="Text", text="Passport details."),
    ]

    parents = create_parent_chunks(elements, doc_id="doc1", max_chars=1000)

    # All headings have body text after them, so each flushes to its own parent
    assert len(parents) == 4

    # BAB I → depth 0
    assert parents[0].depth == 0
    assert parents[0].breadcrumbs == ["BAB I"]

    # A. Syarat Umum → depth 1
    assert parents[1].depth == 1
    assert parents[1].breadcrumbs == ["BAB I", "A. Syarat Umum"]

    # 1. Dokumen → depth 2
    assert parents[2].depth == 2
    assert parents[2].breadcrumbs == ["BAB I", "A. Syarat Umum", "1. Dokumen"]

    # a) Paspor → depth 3
    assert parents[3].depth == 3
    assert parents[3].breadcrumbs == ["BAB I", "A. Syarat Umum", "1. Dokumen", "a) Paspor"]


def test_sibling_sections_share_parent_id():
    """Sibling sections at the same depth should share the same parent_id."""
    elements = [
        ParsedElement(element_type="Title", text="BAB I", metadata={}),
        ParsedElement(element_type="Text", text="Content."),
        ParsedElement(element_type="Title", text="Pasal 1", metadata={}),
        ParsedElement(element_type="Text", text="Content."),
        ParsedElement(element_type="Title", text="Pasal 2", metadata={}),
        ParsedElement(element_type="Text", text="Content."),
    ]

    parents = create_parent_chunks(elements, doc_id="doc1", max_chars=1000)

    # Pasal 1 and Pasal 2 are siblings (both depth 1 under BAB I)
    assert parents[1].parent_id == parents[0].id
    assert parents[2].parent_id == parents[0].id
    # But they have different paths
    assert parents[1].path != parents[2].path


def test_table_and_figure_inherit_path_and_depth():
    """Table and figure parent chunks should carry path/depth from heading stack."""
    elements = [
        ParsedElement(element_type="Title", text="BAB I", metadata={}),
        ParsedElement(element_type="Text", text="Intro text."),
        ParsedElement(
            element_type="Table",
            text="<table><tr><td>Data</td></tr></table>",
            metadata={"page_number": 5},
        ),
        ParsedElement(
            element_type="Image",
            text="A flowchart diagram.",
            metadata={"image_path": "/tmp/img.png", "page_number": 6},
        ),
    ]

    parents = create_parent_chunks(elements, doc_id="doc1", max_chars=1000)

    table_parent = [p for p in parents if p.content_type == ContentType.TABLE][0]
    figure_parent = [p for p in parents if p.content_type == ContentType.FIGURE][0]

    # Both should have the BAB I path and depth 0
    assert table_parent.path == "doc1.bab_i"
    assert table_parent.depth == 0
    assert figure_parent.path == "doc1.bab_i"
    assert figure_parent.depth == 0


# ---------------------------------------------------------------------------
# Child chunk ordinal + path tests
# ---------------------------------------------------------------------------

def test_child_chunks_have_ordinals_and_paths():
    """Child chunks should have ordinal and path fields assigned."""
    parent = ParentChunkData(
        id="p1",
        doc_id="d1",
        text="Sentence one is long enough. Sentence two is also long enough. "
             "Sentence three is long enough too. Sentence four is long enough.",
        chunk_index=0,
        breadcrumbs=["Section A"],
        path="d1.section_a",
        depth=0,
    )

    children = split_into_children(parent, max_chars=80, overlap_chars=5)

    assert len(children) > 1
    for i, child in enumerate(children):
        assert child.ordinal == i
        assert child.path == f"d1.section_a.c{i}"


def test_child_path_empty_when_parent_path_empty():
    """When parent has no path, child path should also be empty."""
    parent = ParentChunkData(
        id="p1",
        doc_id="d1",
        text="Sentence one is long enough. Sentence two is also long enough.",
        chunk_index=0,
        breadcrumbs=[],
        path="",  # No path
    )

    children = split_into_children(parent, max_chars=80, overlap_chars=5)

    for child in children:
        assert child.path == ""


def test_child_ordinals_sequential_after_gibberish_filter():
    """Ordinals should be sequential (0, 1, 2...) after gibberish filtering."""
    parent = ParentChunkData(
        id="p1",
        doc_id="d1",
        text="This is a long enough sentence that will survive. "
             "Ab. "  # gibberish — filtered out
             "Another good sentence that is long enough to keep. "
             "Yet another long enough sentence here.",
        chunk_index=0,
        breadcrumbs=[],
        path="d1.sec",
    )

    children = split_into_children(parent, max_chars=60, overlap_chars=5)

    # Ordinals should be 0, 1, 2... (sequential, no gaps from filtering)
    for i, child in enumerate(children):
        assert child.ordinal == i

def test_split_into_children_retains_context():
    parent = ParentChunkData(
        id="p1",
        doc_id="d1",
        text="Sentence one. Sentence two. Sentence three. Sentence four. Sentence five.",
        chunk_index=0,
        breadcrumbs=["Chapter 1", "Section 1.1"]
    )

    # Set max chars small enough to force a split (no prefix budget is
    # reserved anymore — the tag is added post-split — so this must be
    # smaller than the body text itself, not the old prefix+body sum).
    children = split_into_children(parent, max_chars=40, overlap_chars=5)

    assert len(children) > 1
    for child in children:
        assert child.text.startswith("Chapter 1 > Section 1.1\n\n")
        assert "[Context:" not in child.text
        assert child.breadcrumbs == ["Chapter 1", "Section 1.1"]


# ---------------------------------------------------------------------------
# Table protection tests
# ---------------------------------------------------------------------------

def test_table_becomes_own_parent_chunk():
    """A Table element should become its own parent chunk, not mixed with prose."""
    elements = [
        ParsedElement(element_type="Title", text="Section A", metadata={"category_depth": 0}),
        ParsedElement(element_type="NarrativeText", text="Some intro text."),
        ParsedElement(
            element_type="Table",
            text="<table><tr><td>Col1</td><td>Col2</td></tr></table>",
            metadata={"text_as_html": "<table>...</table>", "page_number": 1},
        ),
        ParsedElement(element_type="NarrativeText", text="Text after the table."),
    ]

    parents = create_parent_chunks(elements, doc_id="doc1", max_chars=1000)

    # Expect: [Section A text] [Table] [text after table]
    # The table is flushed as its own parent, separating it from prose
    table_parents = [p for p in parents if p.content_type == ContentType.TABLE]
    text_parents = [p for p in parents if p.content_type == ContentType.TEXT]

    assert len(table_parents) == 1, "Table should be its own parent chunk"
    assert "<table>" in table_parents[0].text
    assert table_parents[0].element_metadata.get("text_as_html") == "<table>...</table>"


def test_table_not_split_into_multiple_children():
    """A table parent chunk should produce a single child (no character splitting)."""
    table_html = "<table>" + "<tr><td>A</td><td>B</td></tr>" * 20 + "</table>"

    parent = ParentChunkData(
        id="p1",
        doc_id="d1",
        text=table_html,
        chunk_index=0,
        breadcrumbs=["Section A"],
        content_type=ContentType.TABLE,
        element_metadata={"text_as_html": table_html},
    )

    # Even with a small max_chars, the table should NOT be split
    children = split_into_children(parent, max_chars=50, overlap_chars=5)

    assert len(children) == 1, "Table must not be character-split"
    assert "<table>" in children[0].text
    assert "</table>" in children[0].text
    assert children[0].content_type == ContentType.TABLE


def test_table_with_summary_produces_two_children():
    """A table with a summary in element_metadata should produce 2 children."""
    table_html = "<table><tr><td>A</td></tr></table>"
    summary = "This table contains column A with value A."

    parent = ParentChunkData(
        id="p1",
        doc_id="d1",
        text=table_html,
        chunk_index=0,
        breadcrumbs=[],
        content_type=ContentType.TABLE,
        element_metadata={"text_as_html": table_html, "table_summary": summary},
    )

    children = split_into_children(parent, max_chars=512, overlap_chars=50)

    assert len(children) == 2
    # First child: full table
    assert children[0].text == table_html
    # Second child: summary
    assert summary in children[1].text


def test_table_preserves_breadcrumbs():
    """A table parent chunk should carry the breadcrumbs from its section."""
    elements = [
        ParsedElement(element_type="Title", text="Chapter 1", metadata={"category_depth": 0}),
        ParsedElement(
            element_type="Table",
            text="<table><tr><td>Data</td></tr></table>",
            metadata={"page_number": 5},
        ),
    ]

    parents = create_parent_chunks(elements, doc_id="doc1", max_chars=1000)

    table_parents = [p for p in parents if p.content_type == ContentType.TABLE]
    assert len(table_parents) == 1
    assert table_parents[0].breadcrumbs == ["Chapter 1"]
    assert "[Context:" not in table_parents[0].text


def test_consecutive_tables_become_separate_parents():
    """Two consecutive tables should become two separate parent chunks."""
    elements = [
        ParsedElement(
            element_type="Table",
            text="<table><tr><td>Table 1</td></tr></table>",
            metadata={},
        ),
        ParsedElement(
            element_type="Table",
            text="<table><tr><td>Table 2</td></tr></table>",
            metadata={},
        ),
    ]

    parents = create_parent_chunks(elements, doc_id="doc1", max_chars=1000)

    table_parents = [p for p in parents if p.content_type == ContentType.TABLE]
    assert len(table_parents) == 2
    assert "Table 1" in table_parents[0].text
    assert "Table 2" in table_parents[1].text


# ---------------------------------------------------------------------------
# Figure / VLM description tests
# ---------------------------------------------------------------------------

def test_figure_becomes_own_parent_chunk():
    """A Figure/Image element should become its own parent chunk."""
    elements = [
        ParsedElement(element_type="Title", text="Section A", metadata={"category_depth": 0}),
        ParsedElement(element_type="NarrativeText", text="Intro text."),
        ParsedElement(
            element_type="Image",
            text="Flowchart showing step 1 to step 5 with a decision point.",
            metadata={"image_path": "/tmp/page_1.png", "page_number": 2},
        ),
    ]

    parents = create_parent_chunks(elements, doc_id="doc1", max_chars=1000)

    figure_parents = [p for p in parents if p.content_type == ContentType.FIGURE]
    assert len(figure_parents) == 1
    assert "Flowchart" in figure_parents[0].text
    assert figure_parents[0].element_metadata.get("image_path") == "/tmp/page_1.png"


def test_figure_short_description_single_child():
    """A short VLM description should produce a single child (no splitting)."""
    description = "This flowchart shows the approval process with 3 steps."

    parent = ParentChunkData(
        id="p1",
        doc_id="d1",
        text=description,
        chunk_index=0,
        breadcrumbs=[],
        content_type=ContentType.FIGURE,
    )

    children = split_into_children(parent, max_chars=512, overlap_chars=50)

    assert len(children) == 1
    assert children[0].text == description
    assert children[0].content_type == ContentType.FIGURE


def test_figure_long_description_splits_at_sentences():
    """A long VLM description should split at sentence boundaries."""
    # Build a description longer than max_chars
    description = "Step one is initiated. " * 50  # ~1000 chars

    parent = ParentChunkData(
        id="p1",
        doc_id="d1",
        text=description,
        chunk_index=0,
        breadcrumbs=[],
        content_type=ContentType.FIGURE,
    )

    children = split_into_children(parent, max_chars=100, overlap_chars=10)

    assert len(children) > 1
    for child in children:
        assert child.content_type == ContentType.FIGURE


# ---------------------------------------------------------------------------
# Content type inheritance tests
# ---------------------------------------------------------------------------

def test_text_children_inherit_text_content_type():
    """Text parent chunks should produce TEXT children."""
    parent = ParentChunkData(
        id="p1",
        doc_id="d1",
        text="Sentence one. Sentence two. " * 50,
        chunk_index=0,
        breadcrumbs=[],
        content_type=ContentType.TEXT,
    )

    children = split_into_children(parent, max_chars=100, overlap_chars=10)

    assert len(children) > 1
    for child in children:
        assert child.content_type == ContentType.TEXT


def test_empty_elements_returns_empty_list():
    """Empty elements list should return empty parent list."""
    assert create_parent_chunks([], doc_id="doc1") == []


def test_empty_text_elements_skipped():
    """Elements with empty text should be skipped."""
    elements = [
        ParsedElement(element_type="NarrativeText", text=""),
        ParsedElement(element_type="Table", text=""),
    ]

    parents = create_parent_chunks(elements, doc_id="doc1", max_chars=1000)
    assert parents == []


# ---------------------------------------------------------------------------
# Min-length gibberish filter tests
# ---------------------------------------------------------------------------

def test_short_child_chunks_filtered_out():
    """Child chunks shorter than MIN_CHILD_TEXT_LENGTH (8 chars) should be dropped."""
    from app.rag.chunking.logic import MIN_CHILD_TEXT_LENGTH

    # Mix of long and very short sentences — the short ones should be filtered
    parent = ParentChunkData(
        id="p1",
        doc_id="d1",
        text="This is a long enough sentence that will survive the filter. "
             "Ab. "  # 3 chars — too short, should be dropped
             "Another good sentence that is long enough to keep.",
        chunk_index=0,
        breadcrumbs=[],
        content_type=ContentType.TEXT,
    )

    children = split_into_children(parent, max_chars=60, overlap_chars=5)

    assert len(children) > 0
    for child in children:
        assert len(child.text.strip()) >= MIN_CHILD_TEXT_LENGTH, (
            f"Child text too short ({len(child.text.strip())} chars): {child.text!r}"
        )


def test_all_gibberish_children_produces_empty_list():
    """If all children are shorter than the threshold, return an empty list."""
    parent = ParentChunkData(
        id="p1",
        doc_id="d1",
        text="Ab. Cd.",  # 7 chars total — single child, below 8-char threshold
        chunk_index=0,
        breadcrumbs=[],
        content_type=ContentType.TEXT,
    )

    children = split_into_children(parent, max_chars=512, overlap_chars=5)

    assert children == [], "All gibberish children should be filtered out"


def test_min_child_text_length_constant():
    """The MIN_CHILD_TEXT_LENGTH constant should be 8."""
    from app.rag.chunking.logic import MIN_CHILD_TEXT_LENGTH

    assert MIN_CHILD_TEXT_LENGTH == 8


def test_gibberish_filter_applies_under_active_breadcrumbs():
    """Regression test: the breadcrumb tag (e.g.
    "BAB I KETENTUAN UMUM > Pasal 1\\n\\n") prepended to every child chunk
    could mask a short/garbage body from the MIN_CHILD_TEXT_LENGTH check if
    the check measured the whole child text — the tag alone is far longer
    than 8 chars. The filter must measure the body text only, regardless of
    whether breadcrumbs are present.
    """
    from app.rag.chunking.logic import MIN_CHILD_TEXT_LENGTH

    parent = ParentChunkData(
        id="p1",
        doc_id="d1",
        text="Ab. Cd.",  # 7-char body — below threshold on its own
        chunk_index=0,
        breadcrumbs=["BAB I KETENTUAN UMUM", "Pasal 1"],
        content_type=ContentType.TEXT,
    )

    children = split_into_children(parent, max_chars=512, overlap_chars=5)

    assert children == [], (
        "Gibberish body text should still be filtered even when a breadcrumb "
        "context prefix is prepended"
    )


def test_default_parent_max_chars_is_4096():
    """DEFAULT_PARENT_MAX_CHARS should be 4096 (updated from 2000)."""
    from app.rag.chunking.logic import DEFAULT_PARENT_MAX_CHARS

    assert DEFAULT_PARENT_MAX_CHARS == 4096


# ---------------------------------------------------------------------------
# Parent chunk page attribution (citation accuracy)
# ---------------------------------------------------------------------------


def test_page_uses_first_body_element_not_trailing_heading():
    """A heading at the bottom of one page followed by body text on the
    next page should attribute the chunk to the body's page, not the
    heading's — otherwise citations point at the wrong PDF page."""
    elements = [
        ParsedElement(
            element_type="Title",
            text="BAB I Ketentuan Umum",
            metadata={"category_depth": 0, "page_number": 3},
        ),
        ParsedElement(
            element_type="NarrativeText",
            text="Pasal ini menjelaskan definisi umum yang berlaku.",
            metadata={"page_number": 4},
        ),
    ]

    parents = create_parent_chunks(elements, doc_id="doc1", max_chars=1000)

    assert len(parents) == 1
    assert parents[0].page == 4


def test_page_falls_back_to_heading_page_when_chunk_has_no_body_text():
    """A chunk made up only of heading elements (no body text before the
    next flush) should still report a page rather than None."""
    elements = [
        ParsedElement(
            element_type="Title",
            text="BAB I Ketentuan Umum",
            metadata={"category_depth": 0, "page_number": 3},
        ),
        ParsedElement(element_type="Table", text="| A | B |\n|---|---|\n| 1 | 2 |"),
    ]

    parents = create_parent_chunks(elements, doc_id="doc1", max_chars=1000)

    # The Table flushes the heading-only buffer as its own parent chunk first.
    heading_only = next(p for p in parents if p.content_type == ContentType.TEXT)
    assert heading_only.page == 3
