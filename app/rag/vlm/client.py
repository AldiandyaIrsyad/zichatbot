"""Vision-Language Model adapters for figure enrichment.

Three adapters implement :class:`IVLMEnricher`:

1. :class:`OpenRouterVLMClient` — cloud VLM via OpenRouter (Gemini, GPT-4o).
2. :class:`OllamaVLMClient` — local VLM via Ollama (LLaVA, Qwen-VL).
3. :class:`FallbackVLMClient` — no VLM; PyMuPDF drawing analysis for text-only
   heuristic descriptions.

Wired in ``app/kb/dependency.py::get_vlm_enricher``, which selects one at
runtime (API key presence, Ollama base URL, else :class:`FallbackVLMClient`).
Unlike the rest of ``thesis/``, this module is the deliberate exception to the
stdlib-only purity rule: it calls ``httpx`` directly against OpenRouter/Ollama.
"""

from __future__ import annotations

import base64
import os
import structlog
from typing import Optional

import httpx

from app.shared.retry import external_api_retry
from app.rag.vlm.image_extractor import PyMuPDFImageExtractor
from app.rag.vlm.interfaces import IVLMEnricher

logger = structlog.get_logger(__name__)

# Default prompt for VLM enrichment — optimised for flowcharts and
# annotated screenshots commonly found in government/institutional PDFs.
DEFAULT_VLM_PROMPT = (
    "Deskripsikan gambar ini secara detail. Jika berupa bagan alir (flowchart) "
    "atau diagram proses, jelaskan setiap langkah, pihak/aktor yang terlibat, "
    "titik keputusan, dan urutan kejadiannya. Jika berupa tangkapan layar "
    "(screenshot) yang diberi anotasi, jelaskan apa yang ditunjuk oleh setiap "
    "anotasi. Jika berupa gambar tabel, jelaskan kolom-kolom dan data pentingnya."
)


class OpenRouterVLMClient(IVLMEnricher):
    """Cloud VLM adapter using the OpenRouter API (Gemini, GPT-4o). Sends a
    base64-encoded image to the chat completions endpoint with a vision-capable
    model; requires an API key. The default when an OpenRouter key is configured.
    """

    def __init__(self, api_key: str, model: str, base_url: str, timeout: float = 120.0) -> None:
        self._api_key = api_key
        self._model = model
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=httpx.Timeout(timeout, connect=30.0),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )
        logger.info("OpenRouterVLMClient initialized", model=model)

    async def describe_image(self, image_path: str, prompt: str = DEFAULT_VLM_PROMPT) -> str:
        """Generate a text description of an image using a cloud VLM.

        Raises:
            Exception: The API call fails or the image can't be read.
        """
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        with open(image_path, "rb") as fh:
            image_b64 = base64.b64encode(fh.read()).decode("utf-8")

        # Determine MIME type from extension
        ext = os.path.splitext(image_path)[1].lower()
        mime_map = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}
        mime_type = mime_map.get(ext, "image/png")

        payload = {
            "model": self._model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime_type};base64,{image_b64}"},
                        },
                    ],
                }
            ],
        }

        try:
            response = await self._post_chat_completions(payload)
            data = response.json()
            description = (data["choices"][0]["message"]["content"] or "").strip()
            logger.info("vlm.cloud.success", model=self._model, image=image_path, desc_len=len(description))
            return description
        except Exception as exc:
            logger.error("vlm.cloud.failed", image=image_path, error=str(exc))
            raise

    @external_api_retry
    async def _post_chat_completions(self, payload: dict):
        response = await self._client.post("/chat/completions", json=payload)
        response.raise_for_status()
        return response

    async def close(self) -> None:
        await self._client.aclose()


class OllamaVLMClient(IVLMEnricher):
    """Local VLM adapter using the Ollama API (LLaVA, Qwen-VL). Sends a
    base64-encoded image to the local Ollama instance; no API key required.
    Selected when no OpenRouter key is configured but an Ollama base URL is.
    """

    def __init__(self, base_url: str, model: str, timeout: float = 120.0) -> None:
        self._model = model
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=httpx.Timeout(timeout, connect=30.0),
        )
        logger.info("OllamaVLMClient initialized", model=model, base_url=base_url)

    async def describe_image(self, image_path: str, prompt: str = DEFAULT_VLM_PROMPT) -> str:
        """Generate a text description of an image using a local VLM.

        Raises:
            Exception: The Ollama API call fails.
        """
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        with open(image_path, "rb") as fh:
            image_b64 = base64.b64encode(fh.read()).decode("utf-8")

        payload = {
            "model": self._model,
            "prompt": prompt,
            "images": [image_b64],
            "stream": False,
        }

        try:
            response = await self._client.post("/api/generate", json=payload)
            response.raise_for_status()
            data = response.json()
            description = (data.get("response") or "").strip()
            logger.info("vlm.local.success", model=self._model, image=image_path, desc_len=len(description))
            return description
        except Exception as exc:
            logger.error("vlm.local.failed", image=image_path, error=str(exc))
            raise

    async def close(self) -> None:
        await self._client.aclose()


class FallbackVLMClient(IVLMEnricher):
    """Text-only fallback VLM adapter using PyMuPDF drawing analysis.

    When no VLM is available, analyses the PDF page structure (vector drawings,
    text blocks, images) to produce a heuristic description without any external
    API call. Less rich than a true VLM but preserves structural information.
    The fallback when neither an OpenRouter key nor an Ollama URL is configured.
    """

    def __init__(self, pdf_path: Optional[str] = None) -> None:
        self._extractor = PyMuPDFImageExtractor()
        self._pdf_path = pdf_path
        logger.info("FallbackVLMClient initialized (no VLM — heuristic mode)")

    def set_pdf_path(self, pdf_path: str) -> None:
        """Set the source PDF path for drawing analysis. Must be called before
        :meth:`describe_image` if not provided in the constructor.
        """
        self._pdf_path = pdf_path

    async def describe_image(self, image_path: str, prompt: str = DEFAULT_VLM_PROMPT) -> str:
        """Generate a heuristic text description using PyMuPDF drawing analysis.

        Extracts the page number from the image filename (expects ``page_N.png``)
        and analyses the corresponding PDF page. ``prompt`` is unused in
        fallback mode (kept for protocol compliance).
        """
        if not self._pdf_path:
            return (
                "Visual element detected but no PDF path available for "
                "heuristic analysis. VLM enrichment was not performed."
            )

        # Extract page number from filename (page_1.png → 1)
        filename = os.path.basename(image_path)
        page_number = self._extract_page_number(filename)

        if page_number is None:
            return f"Visual element detected on page (file: {filename}). Heuristic analysis unavailable."

        description = self._extractor.generate_heuristic_description(
            self._pdf_path, page_number
        )
        logger.info("vlm.fallback.success", image=image_path, page=page_number, desc_len=len(description))
        return description

    def _extract_page_number(self, filename: str) -> Optional[int]:
        """Extract the page number from a filename like ``page_3.png``, or None
        if not found.
        """
        import re

        match = re.search(r"page_(\d+)", filename)
        if match:
            return int(match.group(1))
        return None

    async def close(self) -> None:
        self._extractor.close()
