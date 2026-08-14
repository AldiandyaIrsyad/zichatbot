# Purpose-built server for the IVM safety classifier.
#
# Replaces running the guard on the shared inference server, which required a
# patched config.json bind-mounted over a pinned model-snapshot path just to
# supply the label names the upstream checkpoint omits. That workaround could
# never survive a model swap; here the mapping is resolved in code.

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/cache/huggingface

WORKDIR /app

# Installed before the source so a code change does not re-download torch.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

# The model cache is a volume; an empty dir here keeps the path valid when it
# is not mounted.
RUN mkdir -p /cache/huggingface

EXPOSE 7997

# Single worker on purpose: the model is held in process memory, and a second
# worker would load a second copy for no throughput gain on a guard whose
# traffic is one short call per chat turn.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7997", "--workers", "1"]
