"""Local MinerU document parser adapter.

Fulfills ``app/kb/domain/interfaces.py::IDocumentParser``; wired in
``app/kb/dependency.py::get_document_parser`` and selected with
``PARSER_BACKEND=mineru_local``.

Why this exists alongside ``mineru_client.py``: the hosted API is the same
parser but reached over the network, and that network is the weak link. A
measured upload of a 1.89 MB scan moved in bursts separated by stalls of tens of
seconds (~20 KB/s effective) and eventually blocked a single write for the full
300 s budget, so a 7-page document could not be parsed at all. Running the same
pipeline locally removes the upload, the rate limit, and the 200-page cap that
``mineru_client.MAX_PAGES`` has to enforce.

Isolation: MinerU pins its own torch/transformers, which conflict with the ones
this app runs (bge-m3 embeddings, the NLI clients). It is therefore installed
into a *separate* virtualenv and invoked as a subprocess rather than imported.
The two never share a Python process, so neither can break the other.

The output contract is identical to the hosted API — MinerU writes the same
``content_list.json`` — so ``_to_elements`` is reused verbatim and both backends
produce byte-identical elements for the same PDF.

VRAM note: the pipeline backend peaks at **2,495 MiB** — measured 2026-08-19 on
the corpus's largest document (333 pages, ``-m ocr -t true``), 1,692 MiB on a
6-page scan. The cost is transient: this class spawns the CLI per document, so
the card is released between parses.

Nothing needs to be stopped to run it. Sampling the card every 0.4 s during that
333-page parse, *while* the app and the TEI reranker served a continuous query
loop, peaked at **7,103 MiB of 8,192 (1,089 MiB free)** and completed. Only
5,143 MiB of that peak is this stack; the rest is the WSL2 desktop compositor,
which a headless card would not carry. An earlier version of this note claimed a
local parse did not fit alongside serving and told the reader to stop the
reranker first — that was wrong, and it came from believing TEI cost 4.5-6.4 GB
rather than the 1,324 MiB it actually costs.

One caveat: MinerU logs ``GPU Memory: 8 GB, Batch Ratio: 4``, sizing its batch
from the card's *total* memory. 2.4 GB is this card's figure, not a fixed cost —
a larger card will choose a larger ratio and use more.
"""

from __future__ import annotations

import asyncio
import glob
import json
import os
import shutil
import tempfile
from typing import Any, Dict, List, Optional

import structlog

from app.kb.infra.mineru_client import _to_elements
from app.rag.chunking.models import ParsedElement

logger = structlog.get_logger(__name__)


class MinerULocalClient:
    """Parse PDFs with a locally installed MinerU."""

    # MinerU 3.x accepts only these OCR language packs. Indonesian is not one of
    # them, and passing an unsupported value is a hard CLI error — so `id` (the
    # value the hosted API takes happily) is dropped rather than forwarded.
    # Indonesian is Latin script, which the default pack already covers.
    SUPPORTED_LANGS = frozenset(
        {
            "ch", "ch_server", "korean", "ta", "te", "ka", "th", "el",
            "arabic", "east_slavic", "cyrillic", "devanagari",
        }
    )

    def __init__(
        self,
        binary: str = ".venv-mineru/bin/mineru",
        language: str = "id",
        image_dir: str = "./uploads/knowledge_base/images",
        backend: str = "pipeline",
        device: str = "cuda",
        method: str = "ocr",
        timeout_sec: float = 1800.0,
    ) -> None:
        self._binary = binary
        self._language = language if language in self.SUPPORTED_LANGS else ""
        if language and not self._language:
            logger.info(
                "mineru_local.lang_unsupported_using_default",
                requested=language,
                reason="Latin script is covered by MinerU's default pack",
            )
        self._image_dir = image_dir
        self._backend = backend
        self._device = device
        # "ocr" rather than "auto" for the same reason the hosted client sends
        # is_ocr=True: this corpus's embedded text layer is the corrupt part, so
        # trusting it reproduces exactly the garbage the parser switch was meant
        # to escape.
        self._method = method
        self._timeout = timeout_sec
        logger.info(
            "MinerULocalClient initialized",
            binary=binary,
            backend=backend,
            device=device,
            method=method,
            language=language,
        )

    async def parse_pdf(self, file_path: str) -> List[ParsedElement]:
        """Parse ``file_path`` into ordered :class:`ParsedElement`s.

        No page cap: unlike the hosted API, a local run has no 200-page limit,
        so the 3 documents that ``mineru_client.parse_pdf`` refuses are parseable
        here.
        """
        work_dir = tempfile.mkdtemp(prefix="mineru_local_")
        try:
            await self._run_cli(file_path, work_dir)
            content_list, image_map = self._collect_output(work_dir, file_path)
            elements = _to_elements(content_list, image_map)
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

        logger.info(
            "mineru_local.parse.success",
            filename=os.path.basename(file_path),
            elements=len(elements),
            tables=sum(1 for e in elements if e.element_type == "Table"),
        )
        return elements

    async def _run_cli(self, file_path: str, out_dir: str) -> None:
        cmd = [
            self._binary,
            "-p", file_path,
            "-o", out_dir,
            "-b", self._backend,
            "-m", self._method,
            # Mirror the hosted client's request body: tables on, formulas off.
            "-t", "true",
            "-f", "false",
        ]
        if self._language:
            cmd += ["-l", self._language]

        # There is no --device flag; the pipeline reads MINERU_DEVICE_MODE.
        env = {**os.environ, "MINERU_DEVICE_MODE": self._device}
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=env,
        )
        try:
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=self._timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            raise TimeoutError(
                f"local MinerU did not finish {os.path.basename(file_path)} "
                f"within {self._timeout}s"
            ) from None

        if proc.returncode != 0:
            # MinerU writes its diagnostics to stdout/stderr, not to a file, so
            # the tail is the only thing that explains an OOM or a missing model.
            tail = (stdout or b"").decode("utf-8", "replace")[-2000:]
            raise RuntimeError(
                f"local MinerU failed ({proc.returncode}) for "
                f"{os.path.basename(file_path)}: {tail}"
            )

    def _collect_output(
        self, out_dir: str, file_path: str
    ) -> tuple[List[Dict[str, Any]], Dict[str, str]]:
        """Read content_list.json and copy images out of the temp directory.

        Mirrors ``mineru_client._download_result``: ``img_path`` is relative to
        MinerU's output tree, so the images are copied next to the ones the page
        renderer produces and the returned map rewrites those paths. Without it
        the VLM enricher raises ``FileNotFoundError``.
        """
        matches = glob.glob(
            os.path.join(out_dir, "**", "*content_list.json"), recursive=True
        )
        if not matches:
            raise RuntimeError(
                f"local MinerU produced no content_list.json for "
                f"{os.path.basename(file_path)} under {out_dir}"
            )
        # Prefer the plain content_list over any _v2 variant for a stable schema.
        chosen = sorted(matches, key=len)[0]
        with open(chosen, "r", encoding="utf-8") as fh:
            content_list = json.load(fh)

        stem = os.path.splitext(os.path.basename(file_path))[0][:40]
        target_dir = os.path.join(self._image_dir, f"mineru_{stem}")
        os.makedirs(target_dir, exist_ok=True)

        image_map: Dict[str, str] = {}
        parse_root = os.path.dirname(chosen)
        for src in glob.glob(os.path.join(parse_root, "images", "*")):
            if not os.path.isfile(src):
                continue
            target = os.path.join(target_dir, os.path.basename(src))
            shutil.copyfile(src, target)
            # content_list refers to images as "images/<name>"; map both that and
            # the bare name so either spelling resolves.
            image_map[os.path.join("images", os.path.basename(src))] = target
            image_map[os.path.basename(src)] = target

        logger.info(
            "mineru_local.images_extracted",
            count=len(image_map) // 2 or len(image_map),
            dir=target_dir,
        )
        return content_list, image_map

    async def close(self) -> None:  # parity with MinerUClient; nothing to close
        return None
