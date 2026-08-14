"""Unit tests for the content-type router.

Tests the pure ``classify_element()`` function which routes parsed
elements to the correct splitting strategy based on their type.
"""

import pytest

from app.rag.chunking.models import ContentType, ParsedElement
from app.rag.chunking.router import (
    classify_element,
    classify_elements,
    TABLE_ELEMENT_TYPES,
    FIGURE_ELEMENT_TYPES,
    SECTION_BOUNDARY_TYPES,
    IGNORE_ELEMENT_TYPES,
)


class TestClassifyElement:
    """Tests for ``classify_element()``."""

    def test_table_element_classified_as_table(self) -> None:
        """A 'Table' element should be classified as TABLE."""
        element = ParsedElement(
            element_type="Table",
            text="<table><tr><td>A</td></tr></table>",
            metadata={"text_as_html": "<table>...</table>"},
        )
        assert classify_element(element) == ContentType.TABLE

    def test_image_element_classified_as_figure(self) -> None:
        """An 'Image' element should be classified as FIGURE."""
        element = ParsedElement(
            element_type="Image",
            text="",
            metadata={"image_path": "/tmp/page_1.png"},
        )
        assert classify_element(element) == ContentType.FIGURE

    def test_figure_element_classified_as_figure(self) -> None:
        """A 'Figure' element should be classified as FIGURE."""
        element = ParsedElement(
            element_type="Figure",
            text="",
            metadata={},
        )
        assert classify_element(element) == ContentType.FIGURE

    def test_narrative_text_classified_as_text(self) -> None:
        """NarrativeText should be classified as TEXT."""
        element = ParsedElement(
            element_type="NarrativeText",
            text="This is a paragraph of narrative text.",
            metadata={},
        )
        assert classify_element(element) == ContentType.TEXT

    def test_title_classified_as_text(self) -> None:
        """Title elements are section boundaries, not tables/figures → TEXT."""
        element = ParsedElement(
            element_type="Title",
            text="Chapter 1: Introduction",
            metadata={"category_depth": 0},
        )
        assert classify_element(element) == ContentType.TEXT

    def test_list_item_classified_as_text(self) -> None:
        """ListItem should be classified as TEXT."""
        element = ParsedElement(
            element_type="ListItem",
            text="- First item",
            metadata={},
        )
        assert classify_element(element) == ContentType.TEXT

    def test_unknown_element_type_defaults_to_text(self) -> None:
        """Unknown element types should default to TEXT (safe fallback)."""
        element = ParsedElement(
            element_type="SomeNewType",
            text="Unknown content",
            metadata={},
        )
        assert classify_element(element) == ContentType.TEXT

    def test_empty_text_table_still_classified_as_table(self) -> None:
        """A Table element with empty text is still TABLE (for routing)."""
        element = ParsedElement(
            element_type="Table",
            text="",
            metadata={"text_as_html": "<table></table>"},
        )
        assert classify_element(element) == ContentType.TABLE


class TestClassifyElements:
    """Tests for ``classify_elements()`` batch function."""

    def test_classify_elements_sets_content_type_in_place(self) -> None:
        """``classify_elements()`` should set content_type on all elements."""
        elements = [
            ParsedElement(element_type="NarrativeText", text="Hello"),
            ParsedElement(element_type="Table", text="<table>"),
            ParsedElement(element_type="Image", text=""),
        ]
        result = classify_elements(elements)

        assert result is elements  # Same list, modified in-place
        assert result[0].content_type == ContentType.TEXT
        assert result[1].content_type == ContentType.TABLE
        assert result[2].content_type == ContentType.FIGURE

    def test_classify_elements_empty_list(self) -> None:
        """Empty list should return empty list without error."""
        assert classify_elements([]) == []

    def test_classify_elements_all_text(self) -> None:
        """All-text elements should all be classified as TEXT."""
        elements = [
            ParsedElement(element_type="Title", text="Heading"),
            ParsedElement(element_type="NarrativeText", text="Body"),
            ParsedElement(element_type="ListItem", text="Item"),
        ]
        result = classify_elements(elements)
        assert all(e.content_type == ContentType.TEXT for e in result)


class TestElementTypeSets:
    """Tests for the element type constant sets."""

    def test_table_element_types_contains_table(self) -> None:
        assert "Table" in TABLE_ELEMENT_TYPES

    def test_figure_element_types_contains_image_and_figure(self) -> None:
        assert "Image" in FIGURE_ELEMENT_TYPES
        assert "Figure" in FIGURE_ELEMENT_TYPES

    def test_section_boundary_types_contains_title(self) -> None:
        assert "Title" in SECTION_BOUNDARY_TYPES

    def test_ignore_element_types_contains_header_footer_pagenumber(self) -> None:
        assert "Header" in IGNORE_ELEMENT_TYPES
        assert "Footer" in IGNORE_ELEMENT_TYPES
        assert "PageNumber" in IGNORE_ELEMENT_TYPES
