"""Dependency injection for the KB domain — the composition root that builds
concrete adapters (``app/kb/infra/``) and wires them into the application
services behind the domain's Protocol ports (``app/kb/domain/interfaces.py``).
"""

import os
from functools import lru_cache

import structlog

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional

from app.shared.db import get_db_session
from app.kb.config import (
    get_qdrant_settings, get_infinity_settings, get_unstructured_settings,
    get_storage_settings, get_vlm_settings, get_bge_m3_settings,
    get_chunking_settings, get_retrieval_settings, get_qwen3_settings,
    get_reranker_settings, get_mineru_settings,
)
from app.kb.application.retrieval_strategies import BaselineStrategy, DatePriorityStrategy
from app.kb.infra import PostgresKBRepository, QdrantStore, UnstructuredClient, BGEM3Embeddings, InfinityReranker
from app.rag.vlm import FallbackVLMClient, IVLMEnricher, OllamaVLMClient, OpenRouterVLMClient
from app.kb.application.ingest_worker import IngestWorker
from app.kb.application.search_service import SearchService
from app.kb.application.kb_service import KBApplicationService
from app.kb.domain.interfaces import IDocumentParser, IQueryExpander, IReranker, ITextEmbedder

logger = structlog.get_logger(__name__)

async def get_kb_repo(db: AsyncSession = Depends(get_db_session)) -> PostgresKBRepository:
    """Provide ``IKBRepository`` (per-request, bound to the request's DB session)."""
    return PostgresKBRepository(db)

# These four are process-lifetime singletons (not per-request factories): each
# wraps an httpx.AsyncClient (or, for BGEM3Embeddings, an in-process model)
# that's expensive to open/load and must not be recreated per request. Closed
# on shutdown in app/main.py's lifespan.

@lru_cache
def get_vector_store() -> QdrantStore:
    """Provide the process-lifetime ``IVectorStore`` singleton (Qdrant)."""
    config = get_qdrant_settings()
    return QdrantStore(
        host=config.host,
        port=config.port,
        collection_name=config.collection_name,
    )

@lru_cache
def get_document_parser() -> IDocumentParser:
    """Provide the process-lifetime ``IDocumentParser`` singleton
    (Unstructured, local or cloud per ``UnstructuredSettings.api_key``)."""
    mineru = get_mineru_settings()
    if mineru.backend == "mineru_local":
        from app.kb.infra.mineru_local_client import MinerULocalClient

        if not os.path.exists(mineru.local_binary):
            raise RuntimeError(
                f"PARSER_BACKEND=mineru_local but {mineru.local_binary} does not "
                "exist. Install it with: python3 -m venv .venv-mineru && "
                ".venv-mineru/bin/pip install 'mineru[core]'"
            )
        return MinerULocalClient(
            binary=mineru.local_binary,
            language=mineru.language,
            image_dir=get_storage_settings().image_dir,
            backend=mineru.local_backend,
            device=mineru.local_device,
            method=mineru.local_method,
            timeout_sec=mineru.max_wait_sec,
        )

    if mineru.backend == "mineru":
        if not mineru.api_key:
            raise RuntimeError(
                "PARSER_BACKEND=mineru but MINERU_API_KEY is empty. Refusing to "
                "fall back to Unstructured silently — that is how a parser "
                "choice becomes invisible in the index."
            )
        from app.kb.infra.mineru_client import MinerUClient

        return MinerUClient(
            api_key=mineru.api_key,
            base_url=mineru.base_url,
            batch_path=mineru.batch_path,
            language=mineru.language,
            image_dir=get_storage_settings().image_dir,
            poll_interval_sec=mineru.poll_interval_sec,
            max_wait_sec=mineru.max_wait_sec,
            pages_per_request=mineru.pages_per_request,
        )

    config = get_unstructured_settings()
    return UnstructuredClient(
        base_url=config.base_url,
        extract_images=config.extract_images,
        api_key=config.api_key,
    )

@lru_cache
def get_text_embedder() -> ITextEmbedder:
    """Provide the process-lifetime ``ITextEmbedder`` singleton.

    Defaults to in-process BGE-M3 (see ``bge_m3_embeddings.py`` for why not
    Infinity). ``QWEN3_EMBEDDER_ENABLED=true`` switches to Qwen3-Embedding for
    dense plus a separate sparse encoder, since Qwen3 is dense-only.
    """
    qwen3 = get_qwen3_settings()
    if not qwen3.embedder_enabled:
        config = get_bge_m3_settings()
        return BGEM3Embeddings(
            model_name=config.model,
            device=config.device,
            use_fp16=config.use_fp16,
            batch_size=config.batch_size,
        )

    from app.kb.infra.qwen3_embeddings import Qwen3Embeddings
    from app.kb.infra.sparse_encoders import build_sparse_encoder

    # The BGE-M3 lexical encoder deliberately stays on CPU: its purpose is to
    # preserve the measured sparse channel while the GPU budget goes to Qwen3.
    sparse_kwargs = (
        {} if qwen3.sparse_encoder.lower().startswith("bm25")
        else {"model_name": get_bge_m3_settings().model, "device": "cpu"}
    )
    from app.kb.infra.qwen3_embeddings import DEFAULT_QUERY_INSTRUCTION

    return Qwen3Embeddings(
        sparse_encoder=build_sparse_encoder(qwen3.sparse_encoder, **sparse_kwargs),
        model_name=qwen3.embedder_model,
        device=qwen3.device,
        use_fp16=qwen3.use_fp16,
        batch_size=qwen3.embed_batch_size,
        query_instruction=qwen3.query_instruction or DEFAULT_QUERY_INSTRUCTION,
    )

@lru_cache
def get_reranker() -> Optional[IReranker]:
    """Provide the process-lifetime ``IReranker`` singleton, or None if
    reranking is disabled via ``KBInfinitySettings.reranker_enabled``.

    ``QWEN3_RERANKER_ENABLED=true`` swaps Infinity's bge-reranker-v2-m3 for the
    in-process Qwen3 reranker (Infinity's transformers is too old to load it).
    Reranking is query-time only, so this switch needs no reindex and is
    independent of which embedder is active.
    """
    config = get_infinity_settings()
    if not config.reranker_enabled:
        return None

    qwen3 = get_qwen3_settings()
    reranker = get_reranker_settings()

    # RERANKER_BACKEND is the single selector. QWEN3_RERANKER_ENABLED is the
    # older flag for the same choice; honour it so existing .env files and the
    # post_sidang_exp reproduction commands keep working, but treat the explicit
    # backend as authoritative when both are set.
    backend = reranker.backend
    if qwen3.reranker_enabled and backend == "tei":
        backend = "qwen3"

    if backend == "qwen3":
        from app.kb.infra.qwen3_reranker import (
            DEFAULT_RERANK_INSTRUCTION,
            Qwen3Reranker,
        )

        return Qwen3Reranker(
            model_name=qwen3.reranker_model,
            device=qwen3.device,
            use_fp16=qwen3.use_fp16,
            batch_size=qwen3.rerank_batch_size,
            max_length=qwen3.rerank_max_length,
            instruction=qwen3.rerank_instruction or DEFAULT_RERANK_INSTRUCTION,
        )

    if backend == "infinity":
        return InfinityReranker(base_url=config.base_url, model=config.reranker_model)

    from app.kb.infra.tei_reranker import TEIReranker

    return TEIReranker(base_url=reranker.base_url, model=reranker.model)

def get_query_expander() -> Optional[IQueryExpander]:
    """Returns None by default — HyDE query expansion is optional.

    The HyDEExpander requires an LLM connection (``chat/infra``). Wiring it
    here would violate the dependency rule that ``kb/`` must not import
    ``chat/infra``. When HyDE is desired, override this provider at the
    application composition layer (``chat/dependency.py``) to inject a
    ``HyDEExpander`` built from ``chat/infra``.
    """
    return None

def get_vlm_enricher() -> Optional[IVLMEnricher]:
    """Create a VLM enricher based on VLMSettings.

    This is the composition root for the VLM adapter: it reads
    ``VLMSettings`` and selects the concrete ``app.rag.vlm`` client
    based on ``settings.mode``. Returns None if VLM is disabled. The
    enricher is created per-request because the fallback mode needs the
    PDF path (set later by the IngestWorker). For cloud/local modes, a
    new client is created each time — this is acceptable since ingestion
    is not high-frequency.

    Returns:
        IVLMEnricher instance or None.
    """
    settings = get_vlm_settings()
    if not settings.enabled:
        return None

    mode = settings.mode.lower().strip()

    if mode == "cloud":
        if not settings.cloud_api_key:
            # Previously this returned FallbackVLMClient, and that is exactly how
            # the corpus ended up full of raw OCR: VLM_CLOUD_API_KEY interpolated
            # to an empty string in .env, every page silently took the heuristic
            # path, and the damage only surfaced months later as garbled tables
            # in retrieval. Asking for cloud VLM without a key is a
            # misconfiguration, not a degraded mode.
            if settings.strict:
                raise RuntimeError(
                    "VLM_MODE=cloud but VLM_CLOUD_API_KEY is empty. Ingestion "
                    "would silently fall back to heuristic text and bake "
                    "unusable OCR into the index. Set the key, or set "
                    "VLM_MODE=fallback / VLM_STRICT=false to accept that "
                    "deliberately."
                )
            logger.warning("vlm.cloud.no_api_key.degrading", strict=False)
            return FallbackVLMClient()
        return OpenRouterVLMClient(
            api_key=settings.cloud_api_key,
            model=settings.cloud_model,
            base_url=settings.cloud_base_url,
            timeout=settings.timeout,
        )

    if mode == "local":
        if not settings.local_base_url:
            if settings.strict:
                raise RuntimeError(
                    "VLM_MODE=local but VLM_LOCAL_BASE_URL is empty. See the "
                    "cloud branch above for why this is fatal rather than a "
                    "fallback."
                )
            logger.warning("vlm.local.no_base_url.degrading", strict=False)
            return FallbackVLMClient()
        return OllamaVLMClient(
            base_url=settings.local_base_url,
            model=settings.local_model,
            timeout=settings.timeout,
        )

    # mode == "fallback" is an explicit, deliberate choice — allowed. Any other
    # value is a typo that would otherwise degrade silently.
    if mode != "fallback" and settings.strict:
        raise RuntimeError(
            f"Unknown VLM_MODE={settings.mode!r}. Expected 'cloud', 'local', or "
            "'fallback'."
        )
    return FallbackVLMClient()

@lru_cache(maxsize=1)
def get_chunk_tokenizer():
    """Provide the tokenizer defining the token unit for ``fixed`` chunking.

    The same BGE-M3 tokenizer the embedder uses, so window sizes match what the
    model actually sees. Imported lazily and cached for the process lifetime —
    only ``CHUNKING_STRATEGY=fixed`` ever pays for loading it.
    """
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        get_bge_m3_settings().model, local_files_only=True
    )

async def get_ingest_worker(
    db: AsyncSession = Depends(get_db_session),
    repo: PostgresKBRepository = Depends(get_kb_repo),
    parser: UnstructuredClient = Depends(get_document_parser),
    embedder: ITextEmbedder = Depends(get_text_embedder),
    vstore: QdrantStore = Depends(get_vector_store),
    vlm_enricher: Optional[IVLMEnricher] = Depends(get_vlm_enricher),
) -> IngestWorker:
    """Assemble a per-request ``IngestWorker`` from the process-lifetime
    adapters plus VLM/storage settings."""
    storage_config = get_storage_settings()
    vlm_settings = get_vlm_settings()
    chunking_config = get_chunking_settings().to_chunking_config()
    return IngestWorker(
        db=db,
        document_parser=parser,
        text_embedder=embedder,
        vector_store=vstore,
        kb_repo=repo,
        vlm_enricher=vlm_enricher,
        image_dir=storage_config.image_dir,
        page_image_ratio_threshold=vlm_settings.page_image_ratio_threshold,
        page_garbage_ratio_threshold=vlm_settings.page_garbage_ratio_threshold,
        image_description_prompt=vlm_settings.image_description_prompt,
        page_extraction_prompt=vlm_settings.page_extraction_prompt,
        vlm_strict=vlm_settings.strict,
        chunking_config=chunking_config,
        tokenizer=get_chunk_tokenizer() if chunking_config.strategy == "fixed" else None,
    )

async def get_kb_service(
    repo: PostgresKBRepository = Depends(get_kb_repo),
    vstore: QdrantStore = Depends(get_vector_store),
    worker: IngestWorker = Depends(get_ingest_worker),
) -> KBApplicationService:
    """Assemble a per-request ``KBApplicationService`` (admin CRUD + upload)."""
    config = get_storage_settings()
    return KBApplicationService(
        kb_repo=repo,
        vector_store=vstore,
        ingest_worker=worker,
        upload_dir=config.upload_dir,
    )

def get_retrieval_strategy() -> BaselineStrategy | DatePriorityStrategy:
    """Build the retrieval strategy selected by ``RetrievalSettings``.

    Config-default selection (no per-request switch): baseline is the current
    score-only pipeline; date_priority applies the same pipeline plus an age
    penalty on document release date.
    """
    settings = get_retrieval_settings()
    if settings.strategy == "date_priority":
        return DatePriorityStrategy(lam=settings.date_priority_lambda)
    return BaselineStrategy()


async def get_search_service(
    repo: PostgresKBRepository = Depends(get_kb_repo),
    vstore: QdrantStore = Depends(get_vector_store),
    embedder: ITextEmbedder = Depends(get_text_embedder),
    reranker: Optional[InfinityReranker] = Depends(get_reranker),
    query_expander: Optional[IQueryExpander] = Depends(get_query_expander),
    strategy: BaselineStrategy | DatePriorityStrategy = Depends(get_retrieval_strategy),
) -> SearchService:
    """Assemble a per-request ``SearchService`` for the 6-step retrieval
    pipeline. ``query_expander`` resolves to None here unless
    ``chat/dependency.py::get_query_expander`` has overridden the provider
    to enable HyDE. ``strategy`` selects the final document-ranking rule
    (config default)."""
    return SearchService(
        text_embedder=embedder,
        vector_store=vstore,
        kb_repo=repo,
        reranker=reranker,
        query_expander=query_expander,
        retrieval_strategy=strategy,
        rerank_probe=get_retrieval_settings().rerank_probe,
    )
