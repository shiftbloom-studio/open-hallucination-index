# syntax=docker/dockerfile:1

FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md /app/
COPY src /app/src

# Install PyTorch CPU-only first (smaller than full CUDA version)
RUN python -m pip install --upgrade pip \
    && python -m pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip wheel --no-cache-dir --wheel-dir /wheels .


FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Set HuggingFace cache path (replaces TRANSFORMERS_CACHE)
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/.cache/huggingface \
    && chown -R appuser:appuser /app/.cache

COPY --from=builder /wheels /wheels
RUN python -m pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install --no-cache-dir /wheels/* \
    && rm -rf /wheels

# Pre-download the embedding model (speeds up first request and bakes it into image)
# We run this as root but ensure permissions are correct afterwards
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')" \
    && chown -R appuser:appuser /app/.cache

EXPOSE 8080

USER appuser

CMD ["ohi-server"]
