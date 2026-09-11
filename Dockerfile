# JDIH chatbot - FastAPI app image.
#
# Runs the API, the admin UI, the ingestion worker and the guardrails. The
# stateful pieces stay outside: Postgres, Qdrant, the TEI reranker, the NLI
# server and the prompt guard are their own compose services.
#
# GPU: torch is installed from PyPI, which bundles its own CUDA runtime, so no
# CUDA base image is needed: only the host NVIDIA driver plus
# nvidia-container-toolkit, and the `deploy.resources` block in
# docker-compose.yaml. Set BGE_M3_DEVICE=cpu to run without a GPU (slower
# embeddings; everything else is unaffected).
#
# Build:  docker compose build app
# Run:    docker compose up -d app

FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    # HF_HOME is a volume in compose: BGE-M3 is ~2.3 GB and must survive a
    # rebuild, or every image build re-downloads it on first request.
    HF_HOME=/cache/huggingface \
    # Baked into the image (see below) so the first RAM verification does not
    # stall on a model download.
    STANZA_RESOURCES_DIR=/opt/stanza_resources

WORKDIR /app

# curl is used by the compose healthcheck; libgomp1 is torch's OpenMP runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Dependencies before source: a code edit must not re-resolve or re-download
# torch and the CUDA wheels (several GB). The BuildKit cache mount keeps the
# downloaded wheels between builds without putting them in the image layer, so
# editing the lock file costs a re-install but not a re-download.
COPY requirements.lock.txt requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.lock.txt

# Stanza's Indonesian pipeline (tokenize,pos,lemma,depparse): the RAM clause
# splitter needs all four; `lemma` is not optional because the `id` depparse
# model takes lemmas as a feature and raises without it. Downloading at build
# time keeps the first request fast and the container runnable offline.
RUN python -c "import stanza; stanza.download('id', processors='tokenize,pos,lemma,depparse', verbose=False)"

COPY app ./app

# uploads/ is written at runtime (PDFs, extracted page images). Owned by the
# non-root user so a bind-mounted host directory does not need root.
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /app/uploads/knowledge_base/images /cache/huggingface \
    && chown -R appuser:appuser /app /cache/huggingface

USER appuser

EXPOSE 8000

# One worker on purpose: BGE-M3 is held in process memory (~3.5 GB on GPU) and
# a second worker would load a second copy and contend for the same card.
CMD ["uvicorn", "app.main:fastapi_app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
