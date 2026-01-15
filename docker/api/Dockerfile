# syntax=docker/dockerfile:1

FROM python:3.12-slim AS builder

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy

WORKDIR /app

# Install uv for faster, more reliable package installation
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY src/api/pyproject.toml /app/pyproject.toml
COPY README.md /README.md
COPY src/api/src /app/src

# Install PyTorch CPU-only first, then build wheel using uv
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system torch --index-url ${TORCH_INDEX_URL} \
    && uv build --wheel --out-dir /wheels


FROM python:3.12-slim

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124

# Install uv for faster package installation
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    # Set HuggingFace cache path (replaces TRANSFORMERS_CACHE)
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/.cache/huggingface /app/benchmark_results \
    && chown -R appuser:appuser /app/.cache /app/benchmark_results

COPY --from=builder /wheels /wheels
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system torch --index-url ${TORCH_INDEX_URL} \
    && uv pip install --system /wheels/* \
    && rm -rf /wheels

# Benchmark package + dataset (used by benchmark-runner)
COPY src/benchmark /app/benchmark
COPY src/benchmark/benchmark_dataset.csv /app/benchmark/benchmark_dataset.csv
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system /app/benchmark

# Pre-download the embedding model (speeds up first request and bakes it into image)
# Download directly to the app cache directory with proper ownership
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L12-v2')" \
    && chown -R appuser:appuser /app/.cache

EXPOSE 8080

USER appuser

CMD ["ohi-server"]
