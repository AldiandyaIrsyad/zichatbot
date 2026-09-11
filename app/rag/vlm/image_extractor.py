"""PyMuPDF-based image extraction and drawing analysis.

A fallback VLM enricher that uses PyMuPDF (fitz) to analyse visual elements
without an actual Vision-Language Model — detecting drawings (vector graphics),
text annotations, and image regions per page to produce a heuristic text
description. This "text-only fallback" calls no external API (suitable for
offline/privacy-sensitive deployments); descriptions are less rich than a true
VLM but preserve structural information.
"""

from __future__ import annotations

import os
import structlog
from typing import List, Optional, Tuple

import fitz  # PyMuPDF

logger = structlog.get_logger(__name__)

# Ceilings that keep a rendered page inside the VLM endpoint's request limit.
# OpenRouter returned 413 for ~5 MB PNGs; 2200 px on the long edge keeps dense
# A3 scans readable while landing comfortably under it.
MAX_LONG_EDGE_PX = 2200
MIN_DPI = 72
MAX_IMAGE_BYTES = 3_500_000
JPEG_QUALITY = 85

# Drawing-density threshold for flowchart detection: a page with many vector
# drawings (lines, rectangles, curves) likely holds a flowchart or diagram.
FLOWCHART_DRAWING_THRESHOLD = 10


class PyMuPDFImageExtractor:
    """Extracts images and analyses drawings from PDF pages using PyMuPDF. Used
    by :class:`FallbackVLMClient` for text-only heuristic descriptions when no
    VLM is available.
    """

    def __init__(self) -> None:
        logger.info("PyMuPDFImageExtractor initialized")

    def extract_page_image(
        self,
        pdf_path: str,
        page_number: int,
        output_dir: str,
        dpi: int = 150,
    ) -> Optional[str]:
        """Render a PDF page as a PNG image. Returns the saved PNG path, or
        None if extraction failed.
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            doc = fitz.open(pdf_path)
            # page_number is 1-indexed in metadata, 0-indexed in PyMuPDF
            page_idx = max(0, page_number - 1)
            if page_idx >= len(doc):
                logger.warning("image.page_out_of_range", page=page_number, total=len(doc))
                doc.close()
                return None

            page = doc[page_idx]
            # Scale DPI down for oversized pages so the render stays under the
            # VLM endpoint's request limit. A large scan at 150 DPI can exceed
            # 5 MB, which OpenRouter rejects with "413 Payload Too Large" — that
            # accounted for every VLM failure in the first full reingest (45 of
            # 1845 calls), silently leaving those pages untranscribed.
            rect = page.rect
            effective_dpi = dpi
            long_edge_px = max(rect.width, rect.height) * dpi / 72
            if long_edge_px > MAX_LONG_EDGE_PX:
                effective_dpi = max(MIN_DPI, dpi * MAX_LONG_EDGE_PX / long_edge_px)
                logger.info(
                    "image.dpi_reduced",
                    page=page_number,
                    requested_dpi=dpi,
                    effective_dpi=round(effective_dpi),
                )
            mat = fitz.Matrix(effective_dpi / 72, effective_dpi / 72)
            pix = page.get_pixmap(matrix=mat)

            output_path = os.path.join(
                output_dir,
                f"page_{page_number}.png",
            )
            pix.save(output_path)

            # PNG of a dense scan can still be large; fall back to JPEG, which
            # is typically 5-10x smaller for photographic scan content.
            if os.path.getsize(output_path) > MAX_IMAGE_BYTES:
                jpeg_path = os.path.join(output_dir, f"page_{page_number}.jpg")
                pix.save(jpeg_path, jpg_quality=JPEG_QUALITY)
                if os.path.getsize(jpeg_path) < os.path.getsize(output_path):
                    logger.info(
                        "image.recompressed_jpeg",
                        page=page_number,
                        png_bytes=os.path.getsize(output_path),
                        jpg_bytes=os.path.getsize(jpeg_path),
                    )
                    os.remove(output_path)
                    output_path = jpeg_path
                else:
                    os.remove(jpeg_path)
            doc.close()

            logger.debug("image.extracted", path=output_path, page=page_number)
            return output_path
        except Exception as exc:
            logger.error("image.extract_failed", page=page_number, error=str(exc))
            return None

    def analyse_page_drawings(
        self,
        pdf_path: str,
        page_number: int,
    ) -> dict:
        """Analyse vector drawings and text annotations on a page, to detect
        flowcharts and annotated screenshots by counting drawing primitives and
        text blocks.

        Returns a dict with ``drawing_count``, ``text_block_count``,
        ``image_count``, ``has_flowchart`` (drawing density suggests a
        flowchart), and ``has_annotations`` (short text blocks, likely diagram
        annotations).
        """
        try:
            doc = fitz.open(pdf_path)
            page_idx = max(0, page_number - 1)
            if page_idx >= len(doc):
                doc.close()
                return {
                    "drawing_count": 0,
                    "text_block_count": 0,
                    "image_count": 0,
                    "has_flowchart": False,
                    "has_annotations": False,
                }

            page = doc[page_idx]
            drawings = page.get_drawings()
            text_blocks = page.get_text("blocks")
            images = page.get_images()

            drawing_count = len(drawings)
            text_block_count = len(text_blocks)
            image_count = len(images)

            # Heuristic: many drawings = likely a flowchart/diagram
            has_flowchart = drawing_count >= FLOWCHART_DRAWING_THRESHOLD

            # Heuristic: short text blocks (annotations) on a page with drawings
            has_annotations = (
                drawing_count > 0
                and any(
                    len(block[4].strip()) < 50  # type: ignore[index]
                    for block in text_blocks
                    if isinstance(block, tuple) and len(block) > 4
                )
            )

            doc.close()
            return {
                "drawing_count": drawing_count,
                "text_block_count": text_block_count,
                "image_count": image_count,
                "has_flowchart": has_flowchart,
                "has_annotations": has_annotations,
            }
        except Exception as exc:
            logger.error("drawing_analysis.failed", page=page_number, error=str(exc))
            return {
                "drawing_count": 0,
                "text_block_count": 0,
                "image_count": 0,
                "has_flowchart": False,
                "has_annotations": False,
            }

    def generate_heuristic_description(
        self,
        pdf_path: str,
        page_number: int,
    ) -> str:
        """Generate a text-only heuristic description of a page's visual content.

        This is the fallback when no VLM is available. It analyses the
        page structure (drawings, text blocks, images) and produces a
        description like:

        "This page contains a flowchart or process diagram with 15 drawing
        elements and 3 text annotations. The diagram appears to show a
        multi-step process with decision points."

        Args:
            pdf_path: Path to the source PDF file.
            page_number: 1-indexed page number.

        Returns:
            Heuristic text description of the visual content.
        """
        analysis = self.analyse_page_drawings(pdf_path, page_number)

        if not analysis["has_flowchart"] and analysis["drawing_count"] == 0:
            if analysis["image_count"] > 0:
                return (
                    f"This page contains {analysis['image_count']} embedded image(s). "
                    "No vector drawings detected. The image may be a photograph "
                    "or a rasterised diagram."
                )
            return "This page does not contain significant visual elements."

        parts: List[str] = []

        if analysis["has_flowchart"]:
            parts.append(
                f"This page contains a flowchart or process diagram with "
                f"{analysis['drawing_count']} drawing elements"
            )
        elif analysis["drawing_count"] > 0:
            parts.append(
                f"This page contains {analysis['drawing_count']} drawing elements "
                f"(lines, shapes, or connectors)"
            )

        if analysis["has_annotations"]:
            parts.append(
                f"and {analysis['text_block_count']} text annotation(s)"
            )

        if analysis["image_count"] > 0:
            parts.append(f"and {analysis['image_count']} embedded image(s)")

        description = ". ".join(parts) + "."
        return description

    def close(self) -> None:
        """No persistent resources to release."""
        pass
