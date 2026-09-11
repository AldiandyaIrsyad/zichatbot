from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache

from app.rag.chunking.config import (
    DEFAULT_FIXED_CHILD_MAX_TOKENS,
    DEFAULT_FIXED_CHILD_OVERLAP_TOKENS,
    DEFAULT_FIXED_PARENT_MAX_TOKENS,
    ChunkingConfig,
)
from app.rag.chunking.logic import (
    DEFAULT_CHILD_MAX_CHARS,
    DEFAULT_CHILD_OVERLAP_CHARS,
    DEFAULT_PARENT_MAX_CHARS,
)
from app.rag.chunking.page_classifier import VLM_PAGE_EXTRACTION_PROMPT
from app.rag.vlm.client import DEFAULT_VLM_PROMPT
from app.kb.application.retrieval_strategies import DEFAULT_DATE_PRIORITY_LAMBDA

class QdrantSettings(BaseSettings):
    """Connection settings for the Qdrant vector store (``QdrantStore``)."""

    host: str = Field(default="127.0.0.1")
    port: int = Field(default=6333)
    grpc_port: int = Field(default=6334)
    collection_name: str = Field(default="knowledge_base")
    model_config = SettingsConfigDict(env_file=".env", env_prefix="QDRANT_", extra="ignore")

@lru_cache
def get_qdrant_settings() -> QdrantSettings: return QdrantSettings()


class UnstructuredSettings(BaseSettings):
    """Connection/auth settings for ``UnstructuredClient`` (local Docker
    unstructured-api when ``api_key`` is empty, Unstructured Cloud otherwise)."""

    base_url: str = Field(default="http://localhost:8001")
    port: int = Field(default=8001)
    api_key: str = Field(
        default="",
        description="API key for Unstructured Cloud (Bearer token). Empty for local self-hosted.",
    )
    extract_images: bool = Field(
        default=True,
        description="Whether to extract image/figure elements during PDF parsing.",
    )
    model_config = SettingsConfigDict(env_file=".env", env_prefix="UNSTRUCTURED_", extra="ignore")

@lru_cache
def get_unstructured_settings() -> UnstructuredSettings: return UnstructuredSettings()

class MinerUSettings(BaseSettings):
    """Hosted MinerU parser settings, and the switch between parsers.

    ``PARSER_BACKEND`` selects the ``IDocumentParser`` implementation:
      unstructured — Unstructured (default). Returns table HTML but garbles cell
                     content on this corpus ("Kelompok 1}", dropped "Rp.",
                     [merged] cells).
      mineru       — hosted MinerU. Recovered 96 programme codes and 714 tariff
                     values on the UKT schedule where Unstructured recovered
                     none. Its per-file caps (200 pages / 200 MB — the API
                     rejects over 200 despite docs advertising 600) are handled
                     by splitting into page windows, so any page count parses;
                     see ``pages_per_request``.
      mineru_local — the same parser run locally from its own virtualenv. No
                     upload, no rate limit, no page cap. Preferred when the
                     hosted API is stalling uploads; needs the GPU free (stop
                     the TEI reranker for the duration of a parse).
    """

    backend: Literal["unstructured", "mineru", "mineru_local"] = Field(
        default="unstructured", validation_alias="PARSER_BACKEND"
    )
    local_binary: str = Field(
        default=".venv-mineru/bin/mineru", validation_alias="MINERU_LOCAL_BINARY"
    )
    local_device: str = Field(default="cuda", validation_alias="MINERU_LOCAL_DEVICE")
    local_backend: str = Field(default="pipeline", validation_alias="MINERU_LOCAL_BACKEND")
    local_method: str = Field(default="ocr", validation_alias="MINERU_LOCAL_METHOD")
    api_key: str = Field(default="", validation_alias="MINERU_API_KEY")
    base_url: str = Field(default="https://mineru.net", validation_alias="MINERU_BASE_URL")
    batch_path: str = Field(default="/api/v4/file-urls/batch", validation_alias="MINERU_BATCH")
    language: str = Field(default="id", validation_alias="MINERU_LANGUAGE")
    poll_interval_sec: float = Field(default=10.0, validation_alias="MINERU_POLL_INTERVAL_SEC")
    max_wait_sec: float = Field(default=1800.0, validation_alias="MINERU_MAX_WAIT_SEC")
    # Pages per hosted request. The API rejects a file over 200 pages; anything
    # above this value is sliced into windows and re-joined with original page
    # numbers.
    pages_per_request: int = Field(
        default=180, validation_alias="MINERU_PAGES_PER_REQUEST"
    )
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

@lru_cache
def get_mineru_settings() -> MinerUSettings: return MinerUSettings()


class KBStorageSettings(BaseSettings):
    """Filesystem locations for uploaded PDFs and extracted page images."""

    upload_dir: str = Field(default="./uploads/knowledge_base")
    image_dir: str = Field(
        default="./uploads/knowledge_base/images",
        description="Directory for extracted page/region images.",
    )
    model_config = SettingsConfigDict(env_file=".env", env_prefix="STORAGE_KB_", extra="ignore")

@lru_cache
def get_storage_settings() -> KBStorageSettings: return KBStorageSettings()


class ChunkingSettings(BaseSettings):
    """Which chunking strategy the ingestion pipeline runs, and its sizes.

    ``hierarchical`` (default) follows the document's heading structure and is
    domain-aware (BAB/Pasal); ``fixed`` cuts the document into fixed BGE-M3
    token windows and is domain-agnostic. Character sizes apply to the former,
    token sizes to the latter. Consumed by
    ``app/kb/application/ingest_worker.py`` via
    ``app/thesis/chunking/strategy.py``.
    """

    strategy: Literal["hierarchical", "fixed"] = Field(
        default="hierarchical",
        description="Chunking strategy: 'hierarchical' (heading-aware) or 'fixed' (token windows).",
    )

    parent_max_chars: int = Field(
        default=DEFAULT_PARENT_MAX_CHARS,
        description="hierarchical: max characters per parent chunk.",
    )
    child_max_chars: int = Field(
        default=DEFAULT_CHILD_MAX_CHARS,
        description="hierarchical: max characters per child chunk.",
    )
    child_overlap_chars: int = Field(
        default=DEFAULT_CHILD_OVERLAP_CHARS,
        description="hierarchical: character overlap between adjacent child chunks.",
    )
    table_child_mode: Literal["rows", "summary", "both"] = Field(
        default="both",
        description=(
            "What gets embedded for a table: 'rows' (row groups, legacy), "
            "'summary' (a description only — the full table still reaches the "
            "LLM because retrieval hydrates the parent), or 'both'."
        ),
    )

    fixed_parent_max_tokens: int = Field(
        default=DEFAULT_FIXED_PARENT_MAX_TOKENS,
        description="fixed: max tokens per parent window (non-overlapping).",
    )
    fixed_child_max_tokens: int = Field(
        default=DEFAULT_FIXED_CHILD_MAX_TOKENS,
        description="fixed: max tokens per child window.",
    )
    fixed_child_overlap_tokens: int = Field(
        default=DEFAULT_FIXED_CHILD_OVERLAP_TOKENS,
        description="fixed: token overlap between adjacent child windows.",
    )

    model_config = SettingsConfigDict(env_file=".env", env_prefix="CHUNKING_", extra="ignore")

    def to_chunking_config(self) -> ChunkingConfig:
        """Map these settings onto the thesis-layer :class:`ChunkingConfig`."""
        return ChunkingConfig(
            strategy=self.strategy,
            parent_max_chars=self.parent_max_chars,
            child_max_chars=self.child_max_chars,
            child_overlap_chars=self.child_overlap_chars,
            table_child_mode=self.table_child_mode,
            fixed_parent_max_tokens=self.fixed_parent_max_tokens,
            fixed_child_max_tokens=self.fixed_child_max_tokens,
            fixed_child_overlap_tokens=self.fixed_child_overlap_tokens,
        )

@lru_cache
def get_chunking_settings() -> ChunkingSettings: return ChunkingSettings()

# KB only uses Infinity's reranking (embeddings run in-process via BGEM3).
class KBInfinitySettings(BaseSettings):
    """Infinity server settings for the KB context — only the reranking
    model/toggle is used here; embeddings run in-process via ``BGEM3Embeddings``.
    """

    base_url: str = Field(default="http://127.0.0.1:7997")
    embedding_model: str = Field(default="BAAI/bge-m3")
    reranker_model: str = Field(default="BAAI/bge-reranker-v2-m3")
    reranker_enabled: bool = Field(default=True, description="Toggle reranking of retrieved documents.")
    model_config = SettingsConfigDict(env_file=".env", env_prefix="INFINITY_", extra="ignore")

@lru_cache
def get_infinity_settings() -> KBInfinitySettings: return KBInfinitySettings()


class RerankerSettings(BaseSettings):
    """Which server backs the ``IReranker`` port.

    ``tei`` (default) is HuggingFace Text Embeddings Inference — continuous
    batching, actively maintained, no transformers pin. ``infinity`` is the
    legacy backend, kept so the two can be A/B'd with
    ``evals/exp2_retrieval/run.py`` before Infinity is removed.

    Both serve the same model, so switching backends is expected to be
    behaviour-neutral; the eval is how that gets verified rather than assumed.
    """

    backend: Literal["tei", "qwen3", "infinity"] = Field(
        default="tei",
        description="The single reranker selector:\n"
        "  tei      — BAAI/bge-reranker-v2-m3 on text-embeddings-inference (default,\n"
        "             best measured: MRR@5 0.8691 hybrid)\n"
        "  qwen3    — Qwen3-Reranker-0.6B, in-process. Measurably worse on this\n"
        "             corpus (6/6 losses, see post_sidang_exp/reranker_new)\n"
        "  infinity — legacy michaelf34/infinity; identical scores to tei, retired\n"
        "Supersedes QWEN3_RERANKER_ENABLED, which is still honoured for "
        "backwards compatibility but should be considered deprecated.",
    )
    base_url: str = Field(default="http://127.0.0.1:7996")
    model: str = Field(
        default="BAAI/bge-reranker-v2-m3",
        description="Informational for TEI (the server picks its own model via "
        "--model-id); sent in the request body for Infinity.",
    )
    model_config = SettingsConfigDict(env_file=".env", env_prefix="RERANKER_", extra="ignore")

@lru_cache
def get_reranker_settings() -> RerankerSettings: return RerankerSettings()


class BGEM3Settings(BaseSettings):
    """Configuration for the in-process BGE-M3 dense+sparse embedder.

    Used instead of Infinity for embeddings because Infinity serves
    BAAI/bge-m3 as dense-only; BAAI's own FlagEmbedding.BGEM3FlagModel also
    computes the sparse (lexical-weight) vectors hybrid search needs.
    """

    model: str = Field(default="BAAI/bge-m3")
    device: str = Field(
        default="cuda",
        description="'cuda' or 'cpu'. Fall back to 'cpu' if the GPU lacks free "
        "VRAM alongside Infinity's models — ingestion is bottlenecked by "
        "external API calls (Unstructured parsing, VLM), not embedding.",
    )
    use_fp16: bool = Field(default=True)
    batch_size: int = Field(default=12)
    model_config = SettingsConfigDict(env_file=".env", env_prefix="BGE_M3_", extra="ignore")

@lru_cache
def get_bge_m3_settings() -> BGEM3Settings: return BGEM3Settings()


class Qwen3Settings(BaseSettings):
    """Configuration for the in-process Qwen3 embedder + reranker.

    Both run in-process rather than on Infinity because the pinned Infinity
    image ships transformers 4.49 and Qwen3 needs >= 4.51 (michaelfeil/infinity#611).

    ``embedder_enabled`` selects the whole dense stack: off keeps BGE-M3,
    on switches to Qwen3 + ``sparse_encoder``. Point ``QDRANT_COLLECTION_NAME``
    at a fresh collection when enabling — Qwen3 is also 1024-dim, so Qdrant will
    silently accept Qwen3 vectors into a BGE-M3 collection and mix two
    incompatible embedding spaces with no error.
    """

    embedder_enabled: bool = Field(
        default=False,
        description="Use Qwen3-Embedding instead of BGE-M3 for dense vectors.",
    )
    embedder_model: str = Field(default="Qwen/Qwen3-Embedding-0.6B")
    sparse_encoder: str = Field(
        default="bm25",
        description="Sparse producer when Qwen3 is active: 'bm25' (FastEmbed, "
        "CPU) or 'bge-m3' (BGE-M3 lexical weights on CPU, identical to the "
        "current collection's sparse channel).",
    )
    reranker_enabled: bool = Field(
        default=False,
        description="Use the in-process Qwen3 reranker instead of Infinity's "
        "bge-reranker-v2-m3. Independent of embedder_enabled — the reranker is "
        "query-time only and needs no reindex.",
    )
    reranker_model: str = Field(default="Qwen/Qwen3-Reranker-0.6B")
    # Both instructions are query-time only — documents are embedded and scored
    # without them — so changing either is a restart, never a re-embed. That
    # makes them cheap to ablate against an existing collection.
    # Empty = the module default in qwen3_embeddings / qwen3_reranker.
    query_instruction: str = Field(
        default="",
        description="Task description for Qwen3-Embedding's 'Instruct:' query "
        "prefix. Omitting the prefix entirely costs 1-5% retrieval performance "
        "per the model card.",
    )
    rerank_instruction: str = Field(
        default="",
        description="Task description injected as <Instruct> in the reranker's "
        "yes/no prompt.",
    )
    device: str = Field(default="cuda", description="'cuda' or 'cpu'.")
    use_fp16: bool = Field(default=True)
    embed_batch_size: int = Field(default=8)
    rerank_batch_size: int = Field(default=4)
    # 16k, not the model's 32k: parents are ~4096 chars (~1.2k tokens), but the
    # HyDE probe concatenates several passages on the query side, and anything
    # over the budget is silently truncated away rather than erroring.
    rerank_max_length: int = Field(default=16384)
    model_config = SettingsConfigDict(env_file=".env", env_prefix="QWEN3_", extra="ignore")

@lru_cache
def get_qwen3_settings() -> Qwen3Settings: return Qwen3Settings()


class VLMSettings(BaseSettings):
    """Configuration for Vision-Language Model enrichment of visual elements.

    Three modes: ``cloud`` (OpenRouter: Gemini, GPT-4o), ``local`` (Ollama:
    LLaVA, Qwen-VL), or ``fallback`` (no VLM — PyMuPDF drawing analysis for
    text-only heuristic figure descriptions). When enrichment fails or is
    disabled, figures are described heuristically (fallback) or skipped (empty
    text, filtered out by the chunker).
    """

    strict: bool = Field(
        default=True,
        description=(
            "Fail loudly when VLM is unavailable or a call errors, instead of "
            "degrading to heuristic text. Ingestion is write-once and its output "
            "is what every later stage reads, so a silent fallback bakes "
            "unusable OCR into the index and looks like a retrieval problem "
            "months later. Set false only for a deliberate no-VLM run."
        ),
        validation_alias="VLM_STRICT",
    )
    mode: str = Field(
        default="fallback",
        description="VLM provider mode: 'cloud', 'local', or 'fallback'.",
    )
    # Cloud (OpenRouter) settings
    cloud_api_key: str = Field(default="", description="OpenRouter API key for cloud VLM.")
    cloud_model: str = Field(
        default="google/gemini-2.0-flash-001",
        description="Cloud VLM model identifier.",
    )
    cloud_base_url: str = Field(
        default="https://openrouter.ai/api/v1",
        description="OpenRouter API base URL.",
    )
    # Local (Ollama) settings
    local_base_url: str = Field(
        default="http://localhost:11434",
        description="Ollama API base URL for local VLM.",
    )
    local_model: str = Field(
        default="llava:13b",
        description="Local VLM model name in Ollama.",
    )
    # Behaviour
    timeout: float = Field(
        default=120.0,
        description="Timeout in seconds for VLM API calls.",
    )
    enabled: bool = Field(
        default=True,
        description="Master toggle. If False, figures are skipped (no enrichment).",
    )
    page_image_ratio_threshold: float = Field(
        default=0.5,
        description="Min fraction of a page's elements that must be images for VISUAL page classification.",
    )
    page_garbage_ratio_threshold: float = Field(
        default=0.7,
        description="Min fraction of a page's image elements with garbage (<=3 char) OCR text for VISUAL classification.",
    )
    image_description_prompt: str = Field(
        default=DEFAULT_VLM_PROMPT,
        description="Prompt used for single-figure/image enrichment description.",
    )
    page_extraction_prompt: str = Field(
        default=VLM_PAGE_EXTRACTION_PROMPT,
        description="Prompt used for full-page VLM extraction on VISUAL-classified pages.",
    )

    model_config = SettingsConfigDict(env_file=".env", env_prefix="VLM_", extra="ignore")


@lru_cache
def get_vlm_settings() -> VLMSettings:
    """Returns the cached VLMSettings singleton.

    Returns:
        VLMSettings: The cached settings instance.
    """
    return VLMSettings()


class RetrievalSettings(BaseSettings):
    """Retrieval strategy selection — a global config default, not a
    per-request parameter. Future strategies are added to the ``strategy``
    literal and implemented in ``app/kb/application/retrieval_strategies.py``.
    """

    strategy: Literal["baseline", "date_priority"] = Field(
        default="baseline",
        description="Retrieval strategy: 'baseline' (score only) or "
        "'date_priority' (penalize older documents by release date).",
    )
    date_priority_lambda: float = Field(
        default=DEFAULT_DATE_PRIORITY_LAMBDA,
        description="Decay strength for the 'date_priority' strategy.",
    )
    rerank_probe: Literal["query", "hyde"] = Field(
        default="hyde",
        description=(
            "What the cross-encoder scores candidates against when HyDE ran: "
            "'query' (the raw user question) or 'hyde' (the generated passage). "
            "HyDE is the point of HyDE — the hypothetical answer is a better "
            "probe than the question. Measured on this KB, 'hyde' recovered two "
            "queries the raw probe lost entirely ('berapa UKT 2026', 'berapa "
            "biaya UKT di UPI' — both from absent to rank 1) at the cost of one "
            "rank-1→rank-2 slip, with six ties. Falls back to the raw query "
            "whenever no expansion ran, so this is a no-op with HyDE disabled."
        ),
    )
    model_config = SettingsConfigDict(env_file=".env", env_prefix="RETRIEVAL_", extra="ignore")


@lru_cache
def get_retrieval_settings() -> RetrievalSettings:
    """Returns the cached RetrievalSettings singleton."""
    return RetrievalSettings()


