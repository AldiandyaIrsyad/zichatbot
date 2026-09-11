# JDIH Chatbot - UPI Legal-Document RAG with Guardrails

A deployment-focused RAG chatbot over UPI's internal legal and regulatory documents (JDIH corpus: Peraturan Rektor, SK Rektor, Statuta, pedoman). It answers user questions grounded in the knowledge base and applies two robust guardrails:

- **IVM** (Input Validation Module): safety (prompt injection) and relevance/out-of-domain (OOD) gating before retrieval.
- **RAM** (Response Assessment Module): NLI-based faithfulness and hallucination verification on generated answers, featuring Stanza Indonesian dependency parsing for compound legal clauses and 3-way citation badges.

This repository represents the production deployment tree, migrated and cleaned from earlier research archives: evaluation harnesses consolidated under `evals/`, and high-performance inference services decoupled into dedicated containers.

## Layout

```
app/            runtime: chat, kb, guardrails, rag, shared config/db, main.py, frontend
app/guardrails/ IVM (safety, relevance) + RAM (Stanza clause splitting, claim parsing) + NLI client adapters
app/rag/        chunking (hierarchical, fixed, table captions), prompt building, VLM visual enrichment
services/       sidecar inference microservices (HuggingFace TEI reranker, dedicated NLI, prompt guard, ollama)
models/         local model checkpoints (fine-tuned prompt-guard and mmBERT NLI; gitignored)
evals/          unified evaluation harnesses, benchmark datasets, and training pipelines (see evals/README.md)
observability/  Loki + Vector + Grafana logging and telemetry configuration
test/           comprehensive pytest test suite mirroring the app structure
```

## Quickstart

### Everything in Docker

```bash
cp .env.example .env      # Fill in credentials (CHAT_LLM_API_KEY, MINERU_API_KEY, etc.)
```

```bash
COMPOSE_PROFILES=guard-ft,nli-indoroberta,app docker compose up -d --build
```

This starts PostgreSQL, Qdrant, the HuggingFace TEI reranker, the fine-tuned Indonesian prompt guard, the dedicated NLI verification service, and the FastAPI application on <http://localhost:8000>.

#### Profile Orchestration

Compose profiles ensure that only the required inference models run at any given time:

- **Safety Guard Selection**: Exactly one guard runs:
  * `guard-base`: Stock `meta-llama/Llama-Prompt-Guard-2-86M` (port 7998).
  * `guard-ft`: Indonesian fine-tuned checkpoint mounted from `./models/prompt_guard_id` (port 7998).
- **NLI Model Selection**: Exactly one NLI backend runs, which must match `CHAT_NLI_MODEL_KIND` in `.env`:
  * `nli-indoroberta` (recommended default): `StevenLimcorn/indo-roberta-indonli` on port 8002 (`CHAT_NLI_MODEL_KIND=indo_roberta`).
  * `nli-mmbert`: Fine-tuned `mmbert_nli_id` checkpoint on port 8002 (`CHAT_NLI_MODEL_KIND=mmbert`).
  * `nli-zeroshot`: Zero-shot `MoritzLaurer/bge-m3-zeroshot-v2.0-c` baseline on port 8002 (`CHAT_NLI_MODEL_KIND=zeroshot`).
  * Note: All three NLI services bind the exact same host port 8002 (`CHAT_NLI_PORT`), so switching models is a clean stop-then-up profile operation without altering host port mappings.
- **Reranker Backend**: Cross-encoder reranking runs on HuggingFace Text Embeddings Inference (`ghcr.io/huggingface/text-embeddings-inference:86-1.9`) on port 7996. The legacy Infinity service is retained only behind the optional `legacy` profile for A/B benchmarking.
- **Application**: The `app` profile runs the FastAPI backend. Inside the container, dependencies resolve using compose service names, while the same `.env` addresses `127.0.0.1` when running locally on the host.

#### First-Run Tips

1. **Free port 8000 before starting the container**: If a host `uvicorn` instance is listening on port 8000, the container will start and pass internal healthchecks, but the published port will fail to bind. Stop any host process on port 8000 first, then run `docker compose up -d --force-recreate app`.
2. **BGE-M3 Download (~2.3 GB)**: In-process embeddings require downloading BGE-M3 on the first request. The weights persist in the `hf_cache` Docker volume across rebuilds. To avoid waiting on download, seed the volume from host cache:
   ```bash
   docker run --rm -v skripsi_hf_cache:/dst -v "$HOME/.cache/huggingface/hub:/src:ro" alpine sh -c "mkdir -p /dst/hub && cp -r /src/models--BAAI--bge-m3 /dst/hub/"
   ```
3. **GPU passthrough**: The `app` service reserves 1 NVIDIA GPU for fast in-process embeddings. To run on CPU, set `BGE_M3_DEVICE=cpu` and remove the `deploy.resources` block under `app` in `docker-compose.yaml`.

### App on the Host (Development)

Running the application directly on the host enables fast iteration, live reloads, and step-through debugging. Leave the `app` profile off in Docker Compose so port 8000 and GPU memory remain free:

```bash
# Start background databases and inference services
COMPOSE_PROFILES=guard-ft,nli-indoroberta docker compose up -d

# Run local FastAPI dev server with auto-reload
mise run dev              # Executes: uvicorn app.main:fastapi_app --reload
```

### Smoke Test

```bash
curl -s http://localhost:8000/api/admin/pdfs | head -c 200
```

## Tests

The repository includes a comprehensive pytest suite covering chat workflows, retrieval strategies, NLI verification, Stanza clause parsing, VLM cleanup, and table extraction. Run all tests with:

```bash
.venv/bin/pytest -q
```

A project-level `pytest.ini` automatically sets `pythonpath = .` and targets `test/`.

## Configuration

All application settings are defined using [pydantic-settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/) and loaded from `.env` (copy from `.env.example`). Variables marked **[REQUIRED]** must be configured for the application to function.

### Application (`APP_`)

- `APP_TITLE`: Service title in OpenAPI documentation.
- `APP_HOST` / `APP_PORT` / `APP_RELOAD`: FastAPI server host, port, and auto-reload toggle.

### PostgreSQL Database (`POSTGRES_`)

- `POSTGRES_USER` / `POSTGRES_PASSWORD` / `POSTGRES_DB` / `POSTGRES_HOST` / `POSTGRES_PORT`: Connection parameters for PostgreSQL 17. Automatically migrates schema tables and alters columns (e.g. `released_date`) on startup.

### Qdrant Vector Store (`QDRANT_`)

- `QDRANT_HOST` / `QDRANT_PORT` / `QDRANT_GRPC_PORT`: Connection settings for the Qdrant vector database.
- `QDRANT_COLLECTION_NAME`: Target hybrid collection (default: `knowledge_base`).

### Document Ingestion and Parser (`PARSER_` / `MINERU_` / `UNSTRUCTURED_`)

The document parsing backend is selected via `PARSER_BACKEND`:

- `PARSER_BACKEND`: Parser implementation:
  * `mineru` (recommended): Hosted MinerU API (`https://mineru.net`). Preserves complex legal schedules, tables, currency symbols, and multi-column formatting. Files exceeding 200 pages are automatically chunked into overlapping page windows.
  * `mineru_local`: Local MinerU CLI executed inside an isolated virtual environment (`.venv-mineru/bin/mineru`), avoiding PyTorch version conflicts and hosted rate limits.
  * `unstructured`: Local or cloud Unstructured API container (port 8001).
- `MINERU_BASE_URL`: Endpoint for hosted MinerU (default: `https://mineru.net`).
- `MINERU_API_KEY`: API authentication key for hosted MinerU extraction.
- `MINERU_LOCAL_BINARY`: Path to isolated local MinerU executable (default: `.venv-mineru/bin/mineru`).
- `MINERU_PAGES_PER_REQUEST`: Page window limit before auto-splitting large PDFs (default: 180).
- `UNSTRUCTURED_BASE_URL` / `UNSTRUCTURED_PORT`: Endpoint for the Unstructured API fallback container.

### Cross-Encoder Reranker (`RERANKER_`)

- `RERANKER_BACKEND`: Selection of cross-encoder inference engine:
  * `tei` (default and recommended): HuggingFace Text Embeddings Inference (`ghcr.io/huggingface/text-embeddings-inference:86-1.9`) on port 7996. Features dynamic batching and continuous throughput.
  * `qwen3`: In-process `Qwen/Qwen3-Reranker-0.6B` model.
  * `infinity`: Legacy `michaelf34/infinity` container.
- `RERANKER_BASE_URL`: Endpoint URL for the reranker service (default: `http://127.0.0.1:7996`).
- `RERANKER_MODEL`: Model identifier passed to the inference backend (default: `BAAI/bge-reranker-v2-m3`).
- **Metadata Context Prefixing**: During retrieval scoring, candidate passages are automatically prefixed with structured document metadata headers (`Nomor`, `Dokumen`, `Tahun`, `Bagian > Pasal`), resolving clause ambiguity and lifting MRR@5.

### BGE-M3 Embedder (`BGE_M3_`)

- `BGE_M3_MODEL`: Dense and sparse multi-lingual embedding model (default: `BAAI/bge-m3`).
- `BGE_M3_DEVICE`: Device for in-process embedding (`cuda` or `cpu`).
- `BGE_M3_USE_FP16`: Use half-precision floating point on CUDA devices (default: `true`).
- `BGE_M3_BATCH_SIZE`: Inference batch size for document chunk embeddings (default: `12`).

### Retrieval Strategy (`RETRIEVAL_`)

- `RETRIEVAL_STRATEGY`: Active scoring strategy:
  * `baseline`: Standard hybrid dense + sparse score fusion with cross-encoder reranking.
  * `date_priority`: Applies an exponential decay age penalty based on a document's official issuance date (`released_date`).
- `RETRIEVAL_DATE_PRIORITY_LAMBDA`: Decay strength parameter for date priority ranking (default: `0.0005`).
- `RETRIEVAL_RERANK_PROBE`: Target candidate comparison probe:
  * `hyde`: Scores candidates against the generated hypothetical passage (default, significantly improves ambiguous queries).
  * `query`: Scores candidates directly against the raw user question.

### Chunking and Table Formatting (`CHUNKING_`)

- `CHUNKING_STRATEGY`: `hierarchical` (default heading-aware BAB/Pasal parent-child segmenter) or `fixed` (token-window baseline). Changing this setting affects newly ingested documents.
- `CHUNKING_TABLE_CHILD_MODE`: Table child chunk representation mode:
  * `summary` (recommended): Embeds deterministic structural captions (lead document title, column headers, row counts, and data samples) into child chunks, while small-to-big retrieval hydrates the full parent table.
  * `rows`: Slices table into row groups.
  * `both`: Generates both summary chunks and row chunks.

### VLM Visual Pipeline (`VLM_`)

- `VLM_MODE`: Vision-Language Model enrichment backend: `cloud` (OpenRouter Gemini/GPT-4o), `local` (Ollama), or `fallback` (PyMuPDF vector drawing heuristics).
- `VLM_STRICT`: When set to `true` (default), fails ingestion loudly on VLM errors instead of silently baking unparsed OCR into the vector index.
- `VLM_PAGE_IMAGE_RATIO_THRESHOLD` / `VLM_PAGE_GARBAGE_RATIO_THRESHOLD`: Thresholds for classifying pages as visual raster layouts requiring full-page VLM transcription.

### Chat Generation and Cost Accounting (`CHAT_`)

- `CHAT_LLM_BASE_URL` / `CHAT_LLM_API_KEY` / `CHAT_LLM_MODEL`: OpenAI-compatible endpoint credentials. Works with OpenRouter, DeepInfra, Ollama, or vLLM.
- **Provider Pinning and Reproducibility**:
  * `CHAT_LLM_PROVIDER_ORDER=DeepInfra`: Enforces specific upstream provider routing on OpenRouter to prevent unexpected pricing jumps.
  * `CHAT_LLM_PROVIDER_QUANTIZATIONS=fp8`: Restricts model execution to calibrated precision.
  * `CHAT_LLM_ALLOW_FALLBACKS=false`: Disables silent fallback to expensive unpinned providers.
- **Real-Time Telemetry**: Streaming responses emit real-time `event: usage` SSE payloads capturing prompt, completion, cached token counts, and monetary cost across all sub-calls (generation, HyDE, condenser).
- `CHAT_SYSTEM_PROMPT`: Assistant legal grounding persona in Bahasa Indonesia.
- `CHAT_CONDENSE_QUERY`: Rewrites conversational follow-up questions before retrieval while retaining the raw message for safety checks.

### Input Validation Module - IVM (`CHAT_SAFETY_` / `PROMPT_GUARD_` / `CHAT_OOD_`)

- `CHAT_SAFETY_BACKEND`: `prompt_guard` (dedicated local sequence classifier) or `qwen3guard` (generative guard via OpenAI-compatible endpoint).
- `PROMPT_GUARD_BASE_URL`: Endpoint of the dedicated prompt guard service (port 7998).
- `CHAT_SECURITY_THRESHOLD`: Injection probability threshold for blocking malicious queries.
- `CHAT_OOD_METHOD`: Relevance gating method: `llm_judge` (LLM-as-a-judge), `similarity_threshold` (vector score cutoff), or `nli_entailment` (NLI cross-check).

### Response Assessment Module - RAM (`CHAT_NLI_` / `CHAT_RAM_`)

- `CHAT_NLI_MODEL_KIND`: Active NLI architecture: `indo_roberta` (default 3-way classifier), `mmbert` (fine-tuned mmBERT-small on IndoNLI), or `zeroshot` (BGE-M3 zero-shot adapter).
- `CHAT_NLI_BASE_URL`: Shared base endpoint for the dedicated NLI microservice (default: `http://localhost:8002`).
- `CHAT_RAM_DEPENDENCY_PARSE`: When `true` (default), uses Stanza's Indonesian dependency parser (`tokenize,pos,lemma,depparse`) to decompose compound sentences into atomic clauses at coordinating conjunctions (`cc`).
- `CHAT_RAM_ENTAILMENT_THRESHOLD`: Minimum confidence score to label a claim "Supported" (default: `0.5`).
- `CHAT_RAM_CONTRADICTION_THRESHOLD`: Minimum confidence score to label a claim "Contradicted" (default: `0.7`).
- **Citation Badges**: Answers display 3-way verification badges for full auditability: `*(Supported: X; Neutral: Y; Contradicted: Z; DocID; Page; Evidence)*` and `*(Unverified)*`.

### HyDE Ensemble (`CHAT_HYDE_`)

- `CHAT_HYDE_ENABLED`: Generates hypothetical legal document passages, computes average dense vectors, fuses rankings with raw query hybrid search via Reciprocal Rank Fusion (RRF), and feeds top candidates to the TEI reranker.
- `CHAT_HYDE_CONTEXT_ENABLED`: Dynamically grounds hypothetical passage generation using titles and descriptions from active KB documents (cached in-process with a TTL).

### Outbound Concurrency Caps (`CHAT_LLM_MAX_CONCURRENCY` / `CHAT_NLI_MAX_CONCURRENCY`)

- `CHAT_LLM_MAX_CONCURRENCY`: Caps simultaneous outbound LLM requests (default: 4).
- `CHAT_NLI_MAX_CONCURRENCY`: Caps concurrent NLI verification requests (default: 8) to prevent memory contention.

### Observability (`LOGGER_`)

- `LOGGER_LOKI_PORT` / `LOGGER_VECTOR_PORT` / `LOGGER_GRAFANA_PORT`: Ports for the centralized logging stack in `observability/`.
- `LOGGER_USERNAME` / `LOGGER_PASSWORD`: Credentials for the Grafana monitoring dashboard.

## Ingesting and Managing Documents

Document ingestion and knowledge base administration are performed directly through the built-in Admin UI or programmatically via the REST API, rather than relying on brittle scraping pipelines.

### Admin Web Dashboard

Navigate to <http://localhost:8000/admin> to manage the document repository:

- **Upload PDFs**: Directly upload official regulatory documents with title, description, and issuance date (`released_date`).
- **Ingest Pipeline**: Click `Ingest` to trigger background document processing:
  1. PDF parsing with MinerU (`PARSER_BACKEND=mineru`) to extract clean text, tables, and page imagery.
  2. Hierarchical BAB/Pasal chunking with deterministic table caption summaries.
  3. Dense and sparse vector generation via in-process BGE-M3.
  4. Point storage in Qdrant hybrid index and relational metadata persistence in PostgreSQL.
- **Document Lifecycle & Governance**: Toggle active status (to include or exclude documents from retrieval without deleting) or perform cascade deletion (clearing database records and associated vector chunks).

### Programmatic REST API

The administrative endpoints enable automated pipeline integration:

```bash
# Upload a new decree PDF with release metadata
curl -X POST http://localhost:8000/api/admin/pdfs/upload \
  -F "file=@/path/to/peraturan_rektor.pdf" \
  -F "title=Peraturan Rektor No. 003 Tahun 2024" \
  -F "description=Pedoman Standar Biaya Operasional" \
  -F "released_date=2024-03-15"

# Trigger ingestion for an uploaded document
curl -X POST http://localhost:8000/api/admin/pdfs/{pdf_id}/ingest

# Search and filter documents by date range or status
curl -s "http://localhost:8000/api/admin/pdfs/search?released_from=2024-01-01&active=true"

# Bulk toggle document active status
curl -X POST http://localhost:8000/api/admin/pdfs/bulk/status \
  -H "Content-Type: application/json" \
  -d '{"ids": ["uuid-1", "uuid-2"], "active": true}'

# Bulk delete documents and purge Qdrant vector chunks
curl -X POST http://localhost:8000/api/admin/pdfs/bulk/delete \
  -H "Content-Type: application/json" \
  -d '{"ids": ["uuid-1", "uuid-2"]}'
```

## Evaluations

Comprehensive evaluation harnesses and empirical findings are documented in `evals/README.md`:
- **Experiment 1a-1d**: IVM safety classification, relevance gating, instruction hijacking, and boundary OOD behavior.
- **Experiment 2 & 2a**: Hybrid vs dense vs sparse retrieval performance, and hierarchical vs fixed chunking.
- **Experiment 3**: RAM NLI faithfulness benchmark across Indo-RoBERTa, fine-tuned mmBERT, and zero-shot backends.
- **Experiment 4**: End-to-end RAG pipeline evaluation with blind human auditing and latency telemetry.
