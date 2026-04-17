# Multi-stage build for the OHI v2 Lambda container.
# Uses AWS Lambda Web Adapter pattern with a plain Python base image —
# NOT the public.ecr.aws/lambda/python base, whose entrypoint expects a
# function-handler contract that conflicts with running a web server.
#
# Lambda gets ORCHESTRATION dependencies only (FastAPI + HTTP clients).
# Heavy ML (torch + sentence-transformers) lives in the PC-side pc-embed
# container, reached via CF tunnel. See docker/pc-embed/ and
# src/api/adapters/embeddings.py.

# --- Stage 1: dependency install -------------------------------------------
FROM public.ecr.aws/docker/library/python:3.12-slim AS builder

WORKDIR /build

RUN pip install --no-cache-dir --target /deps \
    "fastapi>=0.128,<1.0" \
    "uvicorn[standard]>=0.40" \
    "pydantic>=2.12" \
    "pydantic-settings>=2.12" \
    "httpx>=0.28" \
    "openai>=2.14" \
    "neo4j>=6.0" \
    "qdrant-client>=1.16" \
    "redis>=7.1" \
    "pyyaml>=6.0" \
    "orjson>=3.11" \
    "structlog>=24.1" \
    "boto3>=1.34" \
    "wikitextparser>=0.56" \
    "mcp>=1.25"

# --- Stage 2: runtime ------------------------------------------------------
FROM public.ecr.aws/docker/library/python:3.12-slim

# AWS Lambda Web Adapter — intercepts Lambda events, forwards HTTP to PORT=8080.
# Placed at /opt/extensions/ so Lambda auto-loads it as an extension.
COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:0.8.4 /lambda-adapter /opt/extensions/lambda-adapter

ENV PORT=8080
ENV READINESS_CHECK_PATH=/health/ready
ENV PYTHONUNBUFFERED=1

# App code lives at /var/task (Lambda runtime convention).
WORKDIR /var/task

# Copy deps
COPY --from=builder /deps /var/task

# Copy app code. src/api is the *package root* in the flat layout; its immediate
# children (config/, server/, models/, adapters/, pipeline/, interfaces/, services/)
# each become importable top-level packages under /var/task.
COPY src/api/ /var/task/
# Strip files we don't want in the image (pyproject, README, tests).
RUN rm -f /var/task/pyproject.toml /var/task/README.md && rm -rf /var/task/tests

# Make app code importable.
ENV PYTHONPATH=/var/task

# Use `python -m uvicorn` so we don't depend on pip's bin/ being on PATH
# (with --target installs, the bin/ directory isn't searched by default).
# Bare imports: `server.app:app`, NOT `api.server.app:app`.
CMD ["python", "-m", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
