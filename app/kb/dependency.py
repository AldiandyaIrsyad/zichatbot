"""Dependency injection for the KB domain — the composition root that builds
concrete adapters (``app/kb/infra/``) and wires them into the application
services behind the domain's Protocol ports (``app/kb/domain/interfaces.py``).
"""

from functools import lru_cache

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional

from app.shared.db import get_db_session
from app.kb.config import (
    get_qdrant_settings, get_infinity_settings, get_unstructured_settings,
    get_storage_settings, get_vlm_settings, get_bge_m3_settings,
    get_chunking_settings,
)
from app.kb.infra import PostgresKBRepository, QdrantStore, UnstructuredClient, BGEM3Embeddings, InfinityReranker
from app.rag.vlm import FallbackVLMClient, IVLMEnricher, OllamaVLMClient, OpenRouterVLMClient
from app.kb.application.ingest_worker import IngestWorker
from app.kb.application.search_service import SearchService
from app.kb.application.kb_service import KBApplicationService
from app.kb.domain.interfaces import IQueryExpander

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
def get_document_parser() -> UnstructuredClient:
    """Provide the process-lifetime ``IDocumentParser`` singleton
    (Unstructured, local or cloud per ``UnstructuredSettings.api_key``)."""
    config = get_unstructured_settings()
    return UnstructuredClient(
        base_url=config.base_url,
        extract_images=config.extract_images,
        api_key=config.api_key,
    )

@lru_cache
def get_text_embedder() -> BGEM3Embeddings:
    """Provide the process-lifetime ``ITextEmbedder`` singleton (in-process
    BGE-M3 — see ``bge_m3_embeddings.py`` for why not Infinity)."""
    config = get_bge_m3_settings()
    return BGEM3Embeddings(
        model_name=config.model,
        device=config.device,
        use_fp16=config.use_fp16,
        batch_size=config.batch_size,
    )

@lru_cache
def get_reranker() -> Optional[InfinityReranker]:
    """Provide the process-lifetime ``IReranker`` singleton (Infinity), or
    None if reranking is disabled via ``KBInfinitySettings.reranker_enabled``."""
    config = get_infinity_settings()
    if not config.reranker_enabled:
        return None
    return InfinityReranker(base_url=config.base_url, model=config.reranker_model)

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
            return FallbackVLMClient()
        return OpenRouterVLMClient(
            api_key=settings.cloud_api_key,
            model=settings.cloud_model,
            base_url=settings.cloud_base_url,
            timeout=settings.timeout,
        )

    if mode == "local":
        return OllamaVLMClient(
            base_url=settings.local_base_url,
            model=settings.local_model,
            timeout=settings.timeout,
        )

    # Default: fallback mode
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
    embedder: BGEM3Embeddings = Depends(get_text_embedder),
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

async def get_search_service(
    repo: PostgresKBRepository = Depends(get_kb_repo),
    vstore: QdrantStore = Depends(get_vector_store),
    embedder: BGEM3Embeddings = Depends(get_text_embedder),
    reranker: Optional[InfinityReranker] = Depends(get_reranker),
    query_expander: Optional[IQueryExpander] = Depends(get_query_expander),
) -> SearchService:
    """Assemble a per-request ``SearchService`` for the 6-step retrieval
    pipeline. ``query_expander`` resolves to None here unless
    ``chat/dependency.py::get_query_expander`` has overridden the provider
    to enable HyDE."""
    return SearchService(
        text_embedder=embedder,
        vector_store=vstore,
        kb_repo=repo,
        reranker=reranker,
        query_expander=query_expander,
    )
