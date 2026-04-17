# Small embedding service — runs on the PC alongside neo4j/qdrant/pg/redis,
# exposed to Lambda via the existing cloudflared tunnel as
# embed.ohi.shiftbloom.studio. Exists so the Lambda image doesn't need torch
# or sentence-transformers (which bloat the image to ~2GB and blow Lambda's
# 10s INIT cap).

FROM public.ecr.aws/docker/library/python:3.12-slim AS builder

WORKDIR /build

# CPU-only torch (no CUDA stubs); FastAPI + sentence-transformers pinned.
RUN pip install --no-cache-dir --target /deps \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    "fastapi>=0.128,<1.0" \
    "uvicorn[standard]>=0.40" \
    "pydantic>=2.12" \
    "torch>=2.0,<3.0" \
    "sentence-transformers>=2.2"

# --- Runtime --------------------------------------------------------------
FROM public.ecr.aws/docker/library/python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    HF_HOME=/app/hf-cache \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_OFFLINE=1 \
    PORT=8080

WORKDIR /app

COPY --from=builder /deps /app
COPY docker/pc-embed/app.py /app/app.py

# Pre-download the embedding model into the image. Flags above keep runtime
# fully offline; flip them off only for this build-time RUN.
RUN TRANSFORMERS_OFFLINE=0 HF_HUB_OFFLINE=0 \
    python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L12-v2')"

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health/live').read()" || exit 1

CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
