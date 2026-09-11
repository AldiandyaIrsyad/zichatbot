# Purpose-built server for the NLI fine-tune (mmBERT-small on IndoNLI).
#
# mmBERT is a ModernBERT architecture that the pinned Infinity image does not
# serve, so it gets its own transformers-based server — the same pattern as
# services/prompt_guard. It accepts premise/hypothesis pairs so the model's own
# tokenizer joins them with its separator.

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/cache/huggingface

WORKDIR /app

# Installed before the source so a code change does not re-download torch.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

RUN mkdir -p /cache/huggingface

EXPOSE 7997

# Single worker on purpose: the model is held in process memory.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7997", "--workers", "1"]
