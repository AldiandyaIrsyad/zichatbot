"""Protocol interfaces for Vision-Language Model enrichment.

Defines the Protocol the pure ``thesis`` core needs but cannot import; concrete
adapters live in ``thesis/vlm/client.py``. The enricher converts visual
elements (flowcharts, annotated screenshots) into structured text during
ingestion, *before* the chunker runs — keeping the chunker text-only.
"""

from __future__ import annotations

from typing import Protocol


class IVLMEnricher(Protocol):
    """Protocol for Vision-Language Model enrichment of visual elements.

    Implementations wrap a VLM service. All in ``app/thesis/vlm/client.py``:
    ``OpenRouterVLMClient`` (cloud, default), ``OllamaVLMClient`` (local), and
    ``FallbackVLMClient`` (text-only PyMuPDF analysis, no external model).
    Wired in ``app/kb/dependency.py::get_vlm_enricher``.

    Called during ingestion for each ``ParsedElement`` whose ``content_type``
    is :attr:`ContentType.FIGURE`; the returned description replaces the
    element's (likely empty) text so the chunker can process it as narrative.
    """

    async def describe_image(
        self,
        image_path: str,
        prompt: str = (
            "Deskripsikan gambar ini secara detail. Jika berupa bagan alir "
            "(flowchart) atau diagram proses, jelaskan setiap langkah, pihak/aktor "
            "yang terlibat, titik keputusan, dan urutan kejadiannya. Jika berupa "
            "tangkapan layar (screenshot) yang diberi anotasi, jelaskan apa yang "
            "ditunjuk oleh setiap anotasi."
        ),
    ) -> str:
        """Generate a structured text description of a visual element.

        ``prompt`` instructs the VLM; the default is optimised for flowcharts
        and annotated screenshots. Returns a description preserving the visual
        logic.

        Raises:
            Exception: If the VLM call fails. Callers should handle this
                gracefully (fail-closed: skip enrichment, keep original text).
        """
        ...

    async def close(self) -> None:
        """Release resources (HTTP clients, etc.)."""
        ...
