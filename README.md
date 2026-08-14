# JDIH Chatbot — UPI Legal-Document RAG with Guardrails

A deployment-focused RAG chatbot over UPI's internal legal/regulatory documents
(JDIH corpus: Peraturan Rektor, SK Rektor, Statuta, pedoman). It answers
questions grounded in the KB and applies two guardrails:

- **IVM** (Input Validation Module) — safety (prompt injection) + relevance/OOD
  gating before retrieval.
- **RAM** (Response Assessment Module) — NLI-based faithfulness/hallucination
  check on the generated answer.

This repository is the **deployment** tree, migrated and cleaned from the
thesis archive (`../skripsi_app`): eval harnesses consolidated under `evals/`,
operational scripts under `scripts/`, and earlier-version files removed.

## Layout

```
app/            runtime: chat, kb, shared (config/db/logging), main.py, frontend
app/guardrails/ IVM + RAM (safety, relevance, NLI faithfulness)
app/rag/        chunking, prompt building, VLM image/table enrichment
evals/          unified evaluation harnesses + datasets (see evals/README.md)
scripts/        operational tooling, grouped by category (scrapers, ingestion, …)
services/       sidecar servers (prompt-guard, ollama)
models/         local model checkpoints (fine-tune; gitignored)
observability/  Loki + Vector + Grafana config
test/           pytest suite mirroring app/ structure
```

## Quickstart

### 1. Infrastructure

```bash
docker compose up -d postgres qdrant infinity
# optional: the Indonesian prompt-guard fine-tune
docker compose --profile finetuned up -d prompt-guard-ft
```

### 2. App

```bash
cp .env.example .env      # fill in credentials
mise run dev              # uvicorn app.main:fastapi_app --reload
```

### 3. Smoke test

```bash
curl -s http://localhost:8000/api/health
```

## Tests

```bash
.venv/bin/pytest -q
```

## Evaluations

See `evals/README.md` — every experiment (exp1a–1d, exp2, exp2a, exp3, exp4),
the feasibility probes, and the blind human-audit workflow, with exact run
commands and dataset schemas.

## Ingesting documents

1. Scrape the corpus: `scripts/scrapers/download_pdfs_playwright.py`
2. Upload: `scripts/ingestion/bulk_upload_pdfs.py`
3. Ingest (chunk + embed + store) through the KB API.
