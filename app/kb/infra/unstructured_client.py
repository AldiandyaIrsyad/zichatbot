"""Unstructured API document parser adapter.

Fulfills: ``app/kb/domain/interfaces.py::IDocumentParser``.
Wired in: ``app/kb/dependency.py::get_document_parser``.
"""

import os
import time
import asyncio
import json
import httpx
import structlog
from typing import List

from app.kb.domain.interfaces import IDocumentParser
from app.shared.retry import external_api_retry
from app.rag.chunking.models import ParsedElement

logger = structlog.get_logger(__name__)

class UnstructuredClient(IDocumentParser):
    """HTTP adapter for the unstructured-api container (local Docker) or
    Unstructured Platform (cloud, job-based), selected by whether ``api_key``
    is set.

    With ``extract_images=True``, requests Image/Table block extraction so
    ``Image`` elements come back with image paths in metadata for later VLM
    enrichment.
    """

    def __init__(self, base_url: str, extract_images: bool = True, api_key: str = "") -> None:
        """Configure the HTTP client for local or cloud Unstructured. An empty
        ``api_key`` selects the local self-hosted parsing path.
        """
        headers: dict[str, str] = {"accept": "application/json"}
        if api_key:
            # Unstructured API uses a custom header, not Authorization: Bearer
            headers["unstructured-api-key"] = api_key
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=httpx.Timeout(900.0, connect=30.0),
            headers=headers,
        )
        self._extract_images = extract_images
        self._api_key = api_key
        logger.info(
            "UnstructuredClient initialized",
            base_url=base_url,
            extract_images=extract_images,
            auth="cloud" if api_key else "local",
        )

    async def parse_pdf(self, file_path: str) -> List[ParsedElement]:
        """Parse a PDF into typed elements via local or cloud Unstructured
        (dispatched on whether an API key is configured).

        Image elements are kept even with empty text (for later VLM
        enrichment); Table elements prefer ``text_as_html`` over plain text;
        any other empty-text element is dropped.
        """
        resolved = os.path.realpath(file_path)
        if not os.path.isfile(resolved):
            raise FileNotFoundError(f"PDF not found: {resolved!r}")

        filename = os.path.basename(resolved)
        log = logger.bind(filename=filename)
        start_time = time.perf_counter()

        strategy = "hi_res"

        # Build form data — add image extraction when enabled
        form_data: dict[str, str] = {"strategy": strategy}
        if self._extract_images:
            form_data["extract_image_block_types"] = '["Image", "Table"]'
            form_data["extract_image_block_to_payload"] = "false"

        try:
            if self._api_key:
                raw_output = await self._parse_pdf_cloud(resolved, filename, log)
            else:
                raw_output = await self._parse_pdf_local(resolved, filename, log)
        except Exception as exc:
            log.error("parse.failed", error=str(exc))
            raise

        elements: List[ParsedElement] = []
        image_count = 0
        table_count = 0

        for elem in raw_output:
            text = (elem.get("text") or "").strip()

            elem_type = elem.get("type", "UncategorizedText")
            metadata = elem.get("metadata") or {}

            # For tables, if text_as_html is available, prefer it over plain text
            if elem_type == "Table" and metadata.get("text_as_html"):
                text = metadata["text_as_html"]
                table_count += 1

            # For Image elements, the text may be empty but we still want to
            # keep the element so the VLM enricher can process it. The image
            # path is in metadata["image_path"].
            if elem_type == "Image":
                image_count += 1
                # Keep the element even if text is empty — the VLM will fill it
                elements.append(
                    ParsedElement(
                        element_type=elem_type,
                        text=text,  # May be empty — VLM will enrich
                        metadata=metadata,
                    )
                )
                continue

            if not text:
                continue

            elements.append(
                ParsedElement(
                    element_type=elem_type,
                    text=text,
                    metadata=metadata,
                )
            )

        log.info(
            "parse.success",
            elements_count=len(elements),
            image_count=image_count,
            table_count=table_count,
            execution_time_sec=round(time.perf_counter() - start_time, 2)
        )
        return elements

    async def _parse_pdf_local(
        self, resolved: str, filename: str, log: structlog.BoundLogger
    ) -> list:
        """Parse a PDF via the local Docker Unstructured API (synchronous).
        Returns the raw element dicts.
        """
        strategy = "hi_res"
        form_data: dict[str, str] = {"strategy": strategy}
        if self._extract_images:
            form_data["extract_image_block_types"] = '["Image", "Table"]'
            form_data["extract_image_block_to_payload"] = "false"

        with open(resolved, "rb") as fh:
            file_bytes = fh.read()
        response = await self._post_local_general(filename, file_bytes, form_data)
        return response.json()

    @external_api_retry
    async def _post_local_general(self, filename: str, file_bytes: bytes, form_data: dict[str, str]):
        # file_bytes (not a file handle) so a retry re-sends the same body —
        # a handle would already be at EOF after the first attempt.
        # raise_for_status() must happen inside the retried function so a
        # 429/5xx actually triggers a retry (httpx.HTTPStatusError is what
        # the retry policy watches for).
        response = await self._client.post(
            "/general/v0/general",
            files={"files": (filename, file_bytes, "application/pdf")},
            data=form_data,
        )
        response.raise_for_status()
        return response

    async def _parse_pdf_cloud(
        self, resolved: str, filename: str, log: structlog.BoundLogger
    ) -> list:
        """Parse a PDF via the Unstructured Platform API (job-based async):
        create a partitioning job (POST /jobs/), poll GET /jobs/{job_id} until
        COMPLETED, then download results. Returns the raw element dicts.
        """
        # Step 1: Create the job
        settings: dict = {"strategy": "hi_res"}
        if self._extract_images:
            settings["extract_image_block_types"] = ["Image", "Table"]
            settings["pdf_infer_table_structure"] = True

        request_data = json.dumps({
            "job_nodes": [
                {
                    "name": "Partitioner",
                    "type": "partition",
                    "subtype": "unstructured_api",
                    "settings": settings,
                }
            ]
        })

        with open(resolved, "rb") as fh:
            file_bytes = fh.read()
        response = await self._create_job(filename, file_bytes, request_data)
        job_data = response.json()
        job_id = job_data["id"]
        log.info("unstructured.job.created", job_id=job_id)

        # Step 2: Poll until completed
        poll_interval = 5.0
        max_wait = 900.0  # 15 minutes
        elapsed = 0.0
        while True:
            response = await self._poll_job(job_id)
            job_data = response.json()
            status = job_data.get("status", "")
            log.info(
                "unstructured.job.poll",
                job_id=job_id,
                status=status,
                elapsed_sec=round(elapsed, 1),
            )
            if status == "COMPLETED":
                break
            if status in ("FAILED", "STOPPED"):
                raise RuntimeError(
                    f"Unstructured job {job_id} ended with status: {status}"
                )
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval
            if elapsed >= max_wait:
                raise TimeoutError(
                    f"Unstructured job {job_id} timed out after {max_wait}s"
                )

        # Step 3: Download results
        output_files = job_data.get("output_node_files") or []
        if not output_files:
            raise RuntimeError(
                f"Unstructured job {job_id} completed but produced no output files"
            )

        all_elements: list = []
        for file_info in output_files:
            file_id = file_info["file_id"]
            response = await self._download_job_file(job_id, file_id)
            downloaded = response.json()
            if isinstance(downloaded, list):
                all_elements.extend(downloaded)
            else:
                all_elements.append(downloaded)

        log.info(
            "unstructured.job.downloaded",
            job_id=job_id,
            output_files=len(output_files),
            total_elements=len(all_elements),
        )
        return all_elements

    @external_api_retry
    async def _create_job(self, filename: str, file_bytes: bytes, request_data: str):
        # file_bytes (not a file handle) so a retry re-sends the same body —
        # a handle would already be at EOF after the first attempt.
        response = await self._client.post(
            "/jobs/",
            data={"request_data": request_data},
            files={"input_files": (filename, file_bytes, "application/pdf")},
        )
        response.raise_for_status()
        return response

    @external_api_retry
    async def _poll_job(self, job_id: str):
        response = await self._client.get(f"/jobs/{job_id}")
        response.raise_for_status()
        return response

    @external_api_retry
    async def _download_job_file(self, job_id: str, file_id: str):
        response = await self._client.get(
            f"/jobs/{job_id}/download",
            params={"file_id": file_id},
        )
        response.raise_for_status()
        return response

    async def close(self) -> None:
        """Release the underlying HTTP client."""
        await self._client.aclose()
