"""Content-type router for the chunking pipeline.

A pure function classifying each :class:`ParsedElement` into a
:class:`ContentType` so the chunker can route it to the right splitting
strategy: ``element_type == "Table"`` → TABLE, image/figure types → FIGURE,
else TEXT. Pure Python (no infra imports), per the ``thesis/`` purity rule.
"""

from __future__ import annotations

from typing import List

from app.rag.chunking.models import ContentType, ParsedElement


# Element types from unstructured.io that indicate a table
TABLE_ELEMENT_TYPES = {"Table"}

# Element types from unstructured.io that indicate a visual/figure element
# "Image" is the standard unstructured type; "Figure" is a common variant
FIGURE_ELEMENT_TYPES = {"Image", "Figure"}

# Element types that indicate section boundaries (used by the chunker)
SECTION_BOUNDARY_TYPES = {"Title"}

# Element types to ignore (noise, page numbers, repeating headers/footers)
IGNORE_ELEMENT_TYPES = {"Header", "Footer", "PageNumber"}


def classify_element(element: ParsedElement) -> ContentType:
    """Classify a parsed element into a content type (TABLE, FIGURE, or TEXT)
    for routing.
    """
    if element.element_type in TABLE_ELEMENT_TYPES:
        return ContentType.TABLE
    if element.element_type in FIGURE_ELEMENT_TYPES:
        return ContentType.FIGURE
    return ContentType.TEXT


def classify_elements(elements: List[ParsedElement]) -> List[ParsedElement]:
    """Classify all elements in-place (setting ``element.content_type`` via
    :func:`classify_element`) and return the same list for convenience.
    """
    for element in elements:
        element.content_type = classify_element(element)
    return elements
