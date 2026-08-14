"""Dependency injection for the Chat module.

Composition root where ``chat/infra`` adapters are injected into KB-domain
services (e.g. HyDEExpander → SearchService). May import from both ``chat/``
and ``kb/`` — the boundary where dependency inversion is resolved.
"""

from typing import Optional

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.shared.db import get_db_session
from app.chat.config import get_chat_config
from app.chat.infra import LLMConnection, PromptGuardClient, NLIClient, PostgresChatRepository, PdfTextExtractor
from app.chat.infra.hyde_expander import HyDEExpander
from app.chat.application.attachment_service import AttachmentService
from app.chat.application.chat_service import ChatService
from app.chat.application.query_condenser import QueryCondenser
from app.kb.dependency import (
    get_kb_repo,
    get_vector_store,
    get_text_embedder,
    get_reranker,
)
from app.kb.infra import PostgresKBRepository, QdrantStore, BGEM3Embeddings, InfinityReranker
from app.kb.application.search_service import SearchService
from app.kb.domain.interfaces import IQueryExpander

from app.guardrails.ivm.checkers import (
    LLMJudgeRelevanceChecker,
    NliEntailmentRelevanceChecker,
    SimilarityThresholdRelevanceChecker,
)
from app.chat.infra.qwen3guard_client import Qwen3GuardClient
from app.guardrails.ivm.interfaces import IRelevanceChecker, ISafetyModel
from app.guardrails.ivm.judge import LLMJudge
from app.guardrails.ivm.relevance_service import RelevanceService
from app.guardrails.ivm.service import IVMService
from app.guardrails.ram.service import RAMService

async def get_chat_repo(db: AsyncSession = Depends(get_db_session)) -> PostgresChatRepository:
    """Provides ``IChatRepository`` via Postgres, bound to the request-scoped DB session."""
    return PostgresChatRepository(db)

def get_llm_connection() -> LLMConnection:
    """Provides ``ILLMConnection`` — a fresh OpenAI-compatible client per call (not cached),
    pointed at ``ChatConfig``'s configured LLM backend."""
    config = get_chat_config()
    return LLMConnection(base_url=config.llm_base_url, api_key=config.llm_api_key)

def get_prompt_guard_client() -> PromptGuardClient:
    """Provide the local Prompt Guard adapter.

    Points at the dedicated ``prompt_guard_url`` rather than the shared
    inference server, so swapping the classifier for its fine-tune restarts one
    small container instead of reloading the reranker and NLI models too.
    """
    config = get_chat_config()
    return PromptGuardClient(
        base_url=config.prompt_guard_url,
        model=config.prompt_guard_model,
        security_threshold=config.security_threshold,
    )

def get_safety_model() -> ISafetyModel:
    """Provide ``ISafetyModel`` (IVM), selected by ``ChatConfig.safety_backend``.

    One env var swaps the adapter, so the off-the-shelf classifier, its
    fine-tune (same adapter, different ``prompt_guard_model``), and a hosted
    generative guard can be compared without a code change.
    """
    config = get_chat_config()

    if config.safety_backend == "qwen3guard":
        return Qwen3GuardClient(
            base_url=config.safety_api_base_url,
            api_key=config.safety_api_key,
            model=config.safety_api_model,
            controversial_is_unsafe=config.safety_controversial_is_unsafe,
        )

    return get_prompt_guard_client()

def get_nli_client() -> NLIClient:
    """Provides ``INLIModel`` (RAM) via the Infinity-hosted NLI model."""
    config = get_chat_config()
    return NLIClient(base_url=config.infinity_url, model=config.nli_model)

def get_ivm_service(
    safety_client: ISafetyModel = Depends(get_safety_model)
) -> IVMService:
    """Provides the IVM application service (safety gate) wrapping ``get_safety_model``."""
    return IVMService(safety_model=safety_client)

def get_pdf_text_extractor() -> PdfTextExtractor:
    """Provides ``IAttachmentExtractor`` via PyMuPDF, sized from ``ChatConfig``'s attachment limits."""
    config = get_chat_config()
    return PdfTextExtractor(
        max_pages=config.attachment_max_pages,
        max_chars=config.attachment_max_chars,
    )

def get_attachment_service(
    extractor: PdfTextExtractor = Depends(get_pdf_text_extractor),
    ivm_service: IVMService = Depends(get_ivm_service),
) -> AttachmentService:
    """Provides the application service handling chat PDF upload extraction + safety-checking."""
    return AttachmentService(extractor=extractor, ivm_service=ivm_service)

def get_relevance_checker(
    nli_client: NLIClient = Depends(get_nli_client),
) -> IRelevanceChecker:
    """Override ``app.kb.dependency.get_relevance_checker`` (via
    ``app.main``'s ``dependency_overrides``) since the LLM-as-judge check needs
    a cloud LLM (``chat/infra``) that ``kb/`` may not import. Branches on
    ``ChatConfig.ood_method`` to select the IRelevanceChecker backend.
    """
    config = get_chat_config()

    if config.ood_method == "similarity_threshold":
        return SimilarityThresholdRelevanceChecker(threshold=config.ood_similarity_threshold)

    if config.ood_method == "nli_entailment":
        return NliEntailmentRelevanceChecker(
            nli_model=nli_client,
            threshold=config.ood_nli_entailment_threshold,
        )

    judge_llm = LLMConnection(base_url=config.llm_base_url, api_key=config.llm_api_key)
    judge = LLMJudge(
        llm_connection=judge_llm,
        model=config.llm_model,
        system_prompt=config.relevance_judge_prompt,
        user_template=config.relevance_judge_user_template,
    )
    return LLMJudgeRelevanceChecker(judge=judge)

def get_relevance_service(
    checker: IRelevanceChecker = Depends(get_relevance_checker),
) -> RelevanceService:
    """Provides the IVM application service (topical/OOD relevance gate) wrapping ``get_relevance_checker``."""
    return RelevanceService(relevance_checker=checker)

def get_query_expander(
    llm_conn: LLMConnection = Depends(get_llm_connection),
    repo: PostgresKBRepository = Depends(get_kb_repo),
) -> Optional[IQueryExpander]:
    """Build a HyDEExpander if HyDE is enabled in ChatConfig, else None.

    Overrides the default ``None`` provider in ``kb/dependency.py`` — the
    composition layer where ``chat/infra`` adapters are injected into
    KB-domain services.
    """
    config = get_chat_config()
    if not config.hyde_enabled:
        return None
    return HyDEExpander(
        llm=llm_conn,
        model=config.llm_model,
        prompt_template=config.hyde_prompt_template,
        system_prompt=config.hyde_system_prompt,
        max_tokens=config.hyde_max_tokens,
        temperature=config.hyde_temperature,
        num_passages=config.hyde_num_passages,
        kb_repo=repo if config.hyde_context_enabled else None,
        context_max_docs=config.hyde_context_max_docs,
        context_refresh_seconds=config.hyde_context_refresh_seconds,
    )

def get_ram_service(
    nli_client: NLIClient = Depends(get_nli_client),
    reranker: Optional[InfinityReranker] = Depends(get_reranker)
) -> RAMService:
    """Provides the RAM application service (per-sentence citation/hallucination assessment)."""
    if reranker is None:
        raise RuntimeError("Reranker is required for RAMService.")
    return RAMService(
        nli_model=nli_client,
        reranker_model=reranker,
        enabled=True
    )

async def get_search_service(
    repo: PostgresKBRepository = Depends(get_kb_repo),
    vstore: QdrantStore = Depends(get_vector_store),
    embedder: BGEM3Embeddings = Depends(get_text_embedder),
    reranker: Optional[InfinityReranker] = Depends(get_reranker),
    query_expander: Optional[IQueryExpander] = Depends(get_query_expander),
) -> SearchService:
    """Override of kb.dependency.get_search_service injecting HyDE expander."""
    return SearchService(
        text_embedder=embedder,
        vector_store=vstore,
        kb_repo=repo,
        reranker=reranker,
        query_expander=query_expander,
    )

async def get_chat_service(
    chat_repo: PostgresChatRepository = Depends(get_chat_repo),
    llm_conn: LLMConnection = Depends(get_llm_connection),
    search_service: SearchService = Depends(get_search_service),
    ivm_service: IVMService = Depends(get_ivm_service),
    relevance_service: RelevanceService = Depends(get_relevance_service),
    ram_service: RAMService = Depends(get_ram_service)
) -> ChatService:
    """Provides ``ChatService``, the top-level application service assembling all chat-pipeline collaborators."""
    config = get_chat_config()
    # Reuses the generation connection/model: condensation is a short,
    # deterministic rewrite, so it doesn't warrant a separately configured LLM.
    condenser = (
        QueryCondenser(
            llm_connection=llm_conn,
            model=config.llm_model,
            system_prompt=config.condense_prompt,
            user_template=config.condense_user_template,
        )
        if config.condense_query
        else None
    )
    return ChatService(
        chat_repo=chat_repo,
        llm_conn=llm_conn,
        search_service=search_service,
        ivm_service=ivm_service,
        relevance_service=relevance_service,
        ram_service=ram_service,
        model_name=config.llm_model,
        system_prompt=config.system_prompt,
        temperature=config.llm_temperature,
        attachment_search_excerpt_chars=config.attachment_search_excerpt_chars,
        history_max_tokens=config.history_max_tokens,
        query_condenser=condenser,
        refusal_message=config.refusal_message,
        safety_block_message=config.safety_block_message,
        internal_error_message=config.internal_error_message,
    )
