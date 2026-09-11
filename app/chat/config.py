from functools import lru_cache
from typing import Literal
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.chat.application.query_condenser import (
    DEFAULT_CONDENSER_PROMPT,
    DEFAULT_CONDENSER_USER_TEMPLATE,
)
from app.rag.prompts import DEFAULT_SYSTEM_PROMPT_ID
from app.guardrails.ivm.judge import (
    DEFAULT_RELEVANCE_JUDGE_PROMPT,
    DEFAULT_RELEVANCE_JUDGE_USER_TEMPLATE,
)

class ChatConfig(BaseSettings):
    """Configuration for the Chat module.

    Every field uses a ``validation_alias`` mapping its canonical env var
    (``CHAT_*`` / ``INFINITY_*`` / ``PROMPT_GUARD_*``) to the attribute, so
    ``get_chat_config()`` reads real credentials instead of the dead Ollama
    defaults. LLM settings read ``CHAT_LLM_BASE_URL`` / ``CHAT_LLM_API_KEY`` /
    ``CHAT_LLM_MODEL`` — rename any legacy ``LLM_*`` env vars to these.
    """

    # LLM Settings
    llm_base_url: str = Field(
        default="http://localhost:11434/v1",
        description="OpenAI-compatible base URL",
        validation_alias="CHAT_LLM_BASE_URL",
    )
    llm_api_key: SecretStr = Field(
        default=SecretStr("dummy"),
        description="API key for LLM",
        validation_alias="CHAT_LLM_API_KEY",
    )
    llm_model: str = Field(
        default="llama3.1:8b",
        description="Model name to use for generation",
        validation_alias="CHAT_LLM_MODEL",
    )
    llm_temperature: float = Field(
        default=0.0,
        description="Sampling temperature for generation (0.0 = deterministic)",
        validation_alias="CHAT_LLM_TEMPERATURE",
    )
    llm_provider_order: str = Field(
        default="",
        description=(
            "OpenRouter provider preference, highest first (e.g. 'DeepInfra'). "
            "A gateway model is served by many providers at different prices — "
            "deepseek-v4-flash-0731 spans $0.0798-$0.14 per M input tokens — so "
            "leaving this empty means the bill depends on who happens to be "
            "routed to. Comma-separated; empty disables pinning (and any "
            "non-OpenRouter backend ignores it)."
        ),
        validation_alias="CHAT_LLM_PROVIDER_ORDER",
    )
    llm_provider_quantizations: str = Field(
        default="",
        description=(
            "Restrict routing to these quantizations (e.g. 'fp4'), "
            "comma-separated. Quantization changes both price and output "
            "quality, so pinning it keeps a measured cost figure reproducible."
        ),
        validation_alias="CHAT_LLM_PROVIDER_QUANTIZATIONS",
    )
    llm_allow_fallbacks: bool = Field(
        default=True,
        description=(
            "Let OpenRouter fall back to an unpinned provider when the preferred "
            "one is unavailable. True favours availability; False favours a "
            "predictable price and is what a cost measurement needs."
        ),
        validation_alias="CHAT_LLM_ALLOW_FALLBACKS",
    )
    system_prompt: str = Field(
        default=DEFAULT_SYSTEM_PROMPT_ID,
        description="Default system prompt (Bahasa Indonesia)",
        validation_alias="CHAT_SYSTEM_PROMPT",
    )

    # Safety/Relevance (IVM)
    infinity_url: str = Field(default="http://localhost:7997", description="Infinity server URL", validation_alias="INFINITY_BASE_URL")
    prompt_guard_model: str = Field(default="meta-llama/Llama-Prompt-Guard-2-86M", validation_alias="INFINITY_PROMPT_GUARD_MODEL")
    nli_model: str = Field(default="StevenLimcorn/indo-roberta-indonli", validation_alias="INFINITY_NLI_MODEL")
    security_threshold: float = Field(
        default=0.75,
        description="Threshold for prompt injection detection",
        validation_alias="CHAT_SECURITY_THRESHOLD",
    )

    # --- NLI backend selection (RAM + IVM nli_entailment) ---
    nli_model_kind: Literal["indo_roberta", "mmbert", "zeroshot"] = Field(
        default="mmbert",
        description=(
            "Which NLI backend RAM and the IVM nli_entailment checker use. "
            "'mmbert' = the IndoNLI fine-tune on the dedicated NLI service "
            "(default: macro-F1 0.763 lay / 0.572 expert, against "
            "indo-roberta's 0.727 / 0.554); 'indo_roberta' = "
            "indo-roberta-indonli on the same server; 'zeroshot' = "
            "bge-m3-zeroshot-v2.0-c (binary baseline, benchmark only). "
            "The kind also selects the wire format — mmbert posts "
            "{premise, hypothesis} pairs, indo_roberta posts one joined string — "
            "so it must match the container that is actually running, or every "
            "request returns HTTP 422."
        ),
        validation_alias="CHAT_NLI_MODEL_KIND",
    )
    nli_base_url: str = Field(
        default="http://localhost:8002",
        description=(
            "Canonical base URL of the active NLI server (default "
            "http://localhost:8002, matching CHAT_NLI_PORT). All three NLI "
            "backends (indo_roberta, mmbert, zeroshot) share this endpoint."
        ),
        validation_alias="CHAT_NLI_BASE_URL",
    )
    nli_mmbert_model: str = Field(
        default="/models/mmbert_nli_id",
        description="Model id/path for the mmBERT fine-tune on the NLI server.",
        validation_alias="CHAT_NLI_MMBERT_MODEL",
    )
    nli_zeroshot_model: str = Field(
        default="MoritzLaurer/bge-m3-zeroshot-v2.0-c",
        description="Model id for the zero-shot NLI baseline.",
        validation_alias="CHAT_NLI_ZEROSHOT_MODEL",
    )

    @property
    def nli_indo_roberta_base_url(self) -> str:
        """Compatibility shim for legacy per-kind endpoint field."""
        return self.nli_base_url

    @property
    def nli_mmbert_base_url(self) -> str:
        """Compatibility shim for legacy per-kind endpoint field."""
        return self.nli_base_url

    @property
    def nli_zeroshot_base_url(self) -> str:
        """Compatibility shim for legacy per-kind endpoint field."""
        return self.nli_base_url

    # --- RAM claim decomposition ---
    ram_dependency_parse: bool = Field(
        default=True,
        description=(
            "Use an Indonesian dependency parser (Stanza 'id') to split "
            "compound sentences into atomic clauses before NLI verification. "
            "Falls back to sentence-level splitting when disabled or the "
            "parser is unavailable."
        ),
        validation_alias="CHAT_RAM_DEPENDENCY_PARSE",
    )

    # --- RAM NLI confidence gates ---
    ram_entailment_threshold: float = Field(
        default=0.5,
        description=(
            "Minimum NLI entailment score for a claim to be labelled Supported. "
            "A model-entailment below this is downgraded to Neutral (scores are "
            "still surfaced in the citation badge)."
        ),
        validation_alias="CHAT_RAM_ENTAILMENT_THRESHOLD",
    )
    ram_contradiction_threshold: float = Field(
        default=0.7,
        description=(
            "Minimum NLI contradiction score for a claim to be labelled "
            "Contradicted. A model-contradiction below this is downgraded to "
            "Neutral (scores are still surfaced in the citation badge)."
        ),
        validation_alias="CHAT_RAM_CONTRADICTION_THRESHOLD",
    )

    # --- Outbound concurrency caps (backpressure) ---
    llm_max_concurrency: int = Field(
        default=4,
        description=(
            "Max concurrent in-flight LLM requests across all chat callers "
            "(generation, condenser, HyDE, judge). Excess requests await a "
            "semaphore slot."
        ),
        validation_alias="CHAT_LLM_MAX_CONCURRENCY",
    )
    nli_max_concurrency: int = Field(
        default=8,
        description=(
            "Max concurrent in-flight NLI requests to the Infinity server "
            "(per-sentence RAM verification). Kept separate from the LLM cap "
            "so a RAM burst doesn't starve generation."
        ),
        validation_alias="CHAT_NLI_MAX_CONCURRENCY",
    )

    # --- Safety backend selection (IVM) ---
    # Mirrors ``ood_method``: one env var swaps the ISafetyModel adapter so the
    # off-the-shelf classifier, its Indonesian fine-tune, and a hosted
    # generative guard can all be compared without a code change.
    safety_backend: Literal["prompt_guard", "qwen3guard"] = Field(
        default="prompt_guard",
        description=(
            "IVM safety backend. 'prompt_guard' = local sequence classifier "
            "(base or fine-tuned, chosen via prompt_guard_model); 'qwen3guard' = "
            "hosted generative guard over an OpenAI-compatible API."
        ),
        validation_alias="CHAT_SAFETY_BACKEND",
    )
    prompt_guard_url: str = Field(
        default="http://localhost:7999",
        description=(
            "Base URL of the dedicated prompt-guard server. Separate from "
            "``infinity_url`` so the guard can be swapped or restarted without "
            "reloading the reranker and NLI models."
        ),
        validation_alias="PROMPT_GUARD_BASE_URL",
    )
    safety_api_base_url: str = Field(
        default="https://router.huggingface.co/v1",
        description="OpenAI-compatible base URL for the hosted safety backend",
        validation_alias="CHAT_SAFETY_API_BASE_URL",
    )
    safety_api_key: str = Field(
        default="",
        description="API key for the hosted safety backend",
        validation_alias="CHAT_SAFETY_API_KEY",
    )
    safety_api_model: str = Field(
        default="Qwen/Qwen3Guard-Gen-0.6B:featherless-ai",
        description="Model identifier for the hosted safety backend",
        validation_alias="CHAT_SAFETY_API_MODEL",
    )
    safety_controversial_is_unsafe: bool = Field(
        default=True,
        description=(
            "Qwen3Guard emits three tiers (Safe/Controversial/Unsafe). True maps "
            "Controversial to unsafe, matching the fail-closed posture of the rest "
            "of the IVM. Fix this before running an experiment — changing it "
            "afterwards turns the label boundary into a tuned parameter."
        ),
        validation_alias="CHAT_SAFETY_CONTROVERSIAL_IS_UNSAFE",
    )
    relevance_judge_prompt: str = Field(
        default=DEFAULT_RELEVANCE_JUDGE_PROMPT,
        description="System prompt for the IVM LLM-as-judge relevance checker",
        validation_alias="CHAT_RELEVANCE_JUDGE_PROMPT",
    )
    relevance_judge_user_template: str = Field(
        default=DEFAULT_RELEVANCE_JUDGE_USER_TEMPLATE,
        description="User-turn template for the IVM LLM-as-judge relevance checker. {context} and {query} are replaced.",
        validation_alias="CHAT_RELEVANCE_JUDGE_USER_TEMPLATE",
    )
    ood_method: Literal["llm_judge", "similarity_threshold", "nli_entailment"] = Field(
        default="llm_judge",
        description="IVM relevance/OOD backend (see app/thesis/ivm/checkers.py)",
        validation_alias="CHAT_OOD_METHOD",
    )
    ood_similarity_threshold: float = Field(
        default=0.15,
        description=(
            "Min top-1 retrieval (RRF fusion) score for the 'similarity_threshold' "
            "OOD method. Calibrated against this KB: 10 in-domain queries scored "
            "0.28-0.83, while off-topic ones split into a clearly-out cluster "
            "(0.08-0.10) and a confusable tail that still scores 0.33-0.50. The "
            "two distributions overlap, so this gate is deliberately lenient — it "
            "rejects only the clearly-out cluster and lets a confusable query "
            "through to the LLM, which will decline from the retrieved context. "
            "Recalibrate after any change to chunking, embeddings, or fusion."
        ),
        validation_alias="CHAT_OOD_SIMILARITY_THRESHOLD",
    )
    ood_nli_entailment_threshold: float = Field(
        default=0.5,
        description="Min NLI entailment_score for the 'nli_entailment' OOD method",
        validation_alias="CHAT_OOD_NLI_THRESHOLD",
    )

    # --- Refusal messages ---
    # Surfaced verbatim to the user, so they are Indonesian like the rest of the
    # UI and the KB. Overridable because the right wording depends on which
    # corpus is loaded.
    refusal_message: str = Field(
        default=(
            "Maaf, saya hanya dapat menjawab berdasarkan dokumen peraturan UPI "
            "(JDIH) yang tersedia. Pertanyaan ini sepertinya tidak tercakup dalam "
            "dokumen tersebut."
        ),
        description="Shown when the IVM relevance gate judges a query out-of-domain",
        validation_alias="CHAT_REFUSAL_MESSAGE",
    )
    safety_block_message: str = Field(
        default=(
            "Maaf, permintaan ini diblokir oleh filter keamanan dan tidak dapat "
            "diproses."
        ),
        description="Shown when the IVM safety classifier blocks a prompt",
        validation_alias="CHAT_SAFETY_BLOCK_MESSAGE",
    )
    internal_error_message: str = Field(
        default="Maaf, terjadi kesalahan saat memproses permintaan Anda.",
        description="Shown when the chat pipeline fails unexpectedly",
        validation_alias="CHAT_INTERNAL_ERROR_MESSAGE",
    )

    # --- Conversation history ---
    history_max_tokens: int = Field(
        default=3_000,
        description=(
            "Token budget for replayed conversation history (see "
            "app/chat/application/history.py). Covers history only — the system "
            "prompt, retrieved context, and current turn are separate."
        ),
        validation_alias="CHAT_HISTORY_MAX_TOKENS",
    )
    context_max_tokens: int = Field(
        default=6_000,
        description=(
            "Approximate token budget for the retrieved context block fed to "
            "the LLM. Contexts are kept in retrieval order and truncated once "
            "the budget is exceeded (the RAM citation mapping uses the same "
            "clamped list, so [CIT:N] numbering stays consistent)."
        ),
        validation_alias="CHAT_CONTEXT_MAX_TOKENS",
    )
    condense_query: bool = Field(
        default=True,
        description=(
            "Rewrite an elliptical follow-up into a standalone question before "
            "retrieval and the relevance gate. Affects the search query only; "
            "the safety check and the LLM prompt always see the raw message."
        ),
        validation_alias="CHAT_CONDENSE_QUERY",
    )
    condense_prompt: str = Field(
        default=DEFAULT_CONDENSER_PROMPT,
        description="System prompt for the follow-up query condenser",
        validation_alias="CHAT_CONDENSE_PROMPT",
    )
    condense_user_template: str = Field(
        default=DEFAULT_CONDENSER_USER_TEMPLATE,
        description="User-turn template for the condenser. {history} and {question} are replaced.",
        validation_alias="CHAT_CONDENSE_USER_TEMPLATE",
    )

    # Chat PDF Attachments
    attachment_max_file_size_mb: int = Field(
        default=15,
        description="Max upload size (MB) for a chat-attached PDF",
        validation_alias="CHAT_ATTACHMENT_MAX_FILE_SIZE_MB",
    )
    attachment_max_pages: int = Field(
        default=40,
        description="Max page count for a chat-attached PDF",
        validation_alias="CHAT_ATTACHMENT_MAX_PAGES",
    )
    attachment_max_chars: int = Field(
        default=50_000,
        description="Max extracted character count kept from a chat-attached PDF (truncated beyond this)",
        validation_alias="CHAT_ATTACHMENT_MAX_CHARS",
    )
    attachment_search_excerpt_chars: int = Field(
        default=4_000,
        description=(
            "Max characters of attachment text folded into the KB relevance/"
            "retrieval search query (kept short so it doesn't degrade "
            "embedding/HyDE quality — full text still goes to the LLM prompt)"
        ),
        validation_alias="CHAT_ATTACHMENT_SEARCH_EXCERPT_CHARS",
    )

    # HyDE (Hypothetical Document Embeddings) Settings
    hyde_enabled: bool = Field(
        default=False,
        description="Enable HyDE: use a hypothetical answer only for dense retrieval while retaining raw-query sparse weights",
        validation_alias="CHAT_HYDE_ENABLED",
    )
    hyde_max_tokens: int = Field(
        default=256,
        description="Max tokens for the hypothetical document generation",
        validation_alias="CHAT_HYDE_MAX_TOKENS",
    )
    hyde_temperature: float = Field(
        default=0.0,
        description=(
            "Temperature for HyDE generation. 0.0 is deterministic, which makes the "
            "hyde_num_passages 'variants' near-identical and the ensemble mean no better "
            "than one passage; raise it (~0.7) when generating more than one."
        ),
        validation_alias="CHAT_HYDE_TEMPERATURE",
    )
    hyde_num_passages: int = Field(
        default=1,
        ge=1,
        le=5,
        description=(
            "Number of independently generated HyDE passages whose dense vectors are "
            "averaged. Each costs one LLM round-trip, and they are the dominant cost of "
            "a chat request, so the default is a single passage. Raise it (with "
            "hyde_temperature > 0, or the variants come back near-identical) to trade "
            "cost for a smoother mean query vector."
        ),
        validation_alias="CHAT_HYDE_NUM_PASSAGES",
    )
    hyde_system_prompt: str = Field(
        default=(
            "Tulis satu paragraf yang berbunyi seperti kutipan langsung dari pasal, ayat, "
            "atau lampiran peraturan/keputusan institusi Indonesia yang MEMUAT jawaban "
            "pertanyaan. Tulis isi ketentuannya sendiri — kategori, golongan, besaran, "
            "syarat, atau langkahnya — bukan penjelasan tentang di mana informasi itu bisa "
            "dicari, bukan deskripsi dokumen, dan bukan rujukan ke surat edaran atau situs "
            "resmi. Pertahankan entitas yang disebut dalam pertanyaan (nama institusi, "
            "peran, prosedur). Gunakan istilah baku yang akan muncul dalam dokumen aslinya. "
            "Jangan sebut nomor dokumen atau tanggal. Keluarkan paragrafnya saja."
        ),
        description=(
            "System prompt for HyDE hypothetical document generation — sets the domain/register "
            "the model should imitate. May contain a {kb_context} placeholder, filled with the "
            "active KB document titles/descriptions.\n\n"
            "The prompt must push the model to write a passage that *contains* the answer. The "
            "earlier default forbade inventing facts, so the model hedged — 'informasi dapat "
            "dilihat dalam Surat Keputusan Rektor Nomor [X]' — producing meta-text *about* a "
            "decree rather than its provisions. That retrieves the corpus's VLM page "
            "descriptions (also meta-text about decrees) instead of the article that answers "
            "the question. Prior default, kept for ablation: 'Write a short Indonesian "
            "institutional-legal document passage for retrieval. Preserve entities explicitly "
            "present in the question... Do not invent article numbers, dates, document titles, "
            "or facts...'"
        ),
        validation_alias="CHAT_HYDE_SYSTEM_PROMPT",
    )
    hyde_prompt_template: str = Field(
        default=(
            "Question: {query}\n\n"
            "{variant_instruction}\n\n"
            "Retrieval passage:"
        ),
        description="User-turn template for HyDE hypothetical document generation. {query} is replaced with the user query.",
        validation_alias="CHAT_HYDE_PROMPT_TEMPLATE",
    )
    hyde_context_enabled: bool = Field(
        default=False,
        description="Optional non-original variant: ground HyDE with KB titles/descriptions via {kb_context}. Disabled by default for paper-faithful HyDE.",
        validation_alias="CHAT_HYDE_CONTEXT_ENABLED",
    )
    hyde_context_max_docs: int = Field(
        default=20,
        description="Max number of active KB documents listed in the {kb_context} grounding block.",
        validation_alias="CHAT_HYDE_CONTEXT_MAX_DOCS",
    )
    hyde_context_refresh_seconds: int = Field(
        default=300,
        description="TTL (seconds) for the cached {kb_context} grounding block before it's refetched from the KB.",
        validation_alias="CHAT_HYDE_CONTEXT_REFRESH_SECONDS",
    )

    def provider_routing(self) -> dict:
        """OpenRouter's provider-selection block, or ``{}`` when unpinned.

        Returned empty (rather than `{"allow_fallbacks": true}`) when nothing is
        pinned, so a non-OpenRouter backend never receives a field it would have
        to ignore.
        """
        order = [p.strip() for p in self.llm_provider_order.split(",") if p.strip()]
        quants = [q.strip() for q in self.llm_provider_quantizations.split(",") if q.strip()]
        if not order and not quants:
            return {}
        routing: dict = {"allow_fallbacks": self.llm_allow_fallbacks}
        if order:
            routing["order"] = order
        if quants:
            routing["quantizations"] = quants
        return routing

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", populate_by_name=True)

@lru_cache
def get_chat_config() -> ChatConfig:
    """Return the cached :class:`ChatConfig` singleton."""
    return ChatConfig()
