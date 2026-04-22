# Multi-stage build for the OHI v2 Lambda container.
# Uses AWS Lambda Web Adapter pattern with a plain Python base image —
# NOT the public.ecr.aws/lambda/python base, whose entrypoint expects a
# function-handler contract that conflicts with running a web server.
#
# Lambda gets ORCHESTRATION dependencies only (FastAPI + HTTP clients).
# Heavy ML (embeddings + rerank) runs on AWS Bedrock (Titan v2 + Cohere
# rerank-v3-5); see src/api/adapters/embeddings.py and bedrock_rerank.py.
#
# Phase 2 of the Neo4j migration adds a tiny tsnet-based Go sidecar
# (docker/lambda/tsproxy/) that joins the Tailnet at cold start and
# forwards bolt:7687 from 127.0.0.1 to the PC-hosted Neo4j over the
# Tailnet. See docker/lambda/entrypoint.sh for the startup choreography.

# --- Stage 0: build tsnet proxy (Go) ---------------------------------------
FROM public.ecr.aws/docker/library/golang:1.23-alpine AS tsproxy-builder

WORKDIR /src
# Copy go.mod + go.sum together so `go mod download` uses the pinned
# hashes from go.sum and `go build` can verify without re-resolving.
COPY docker/lambda/tsproxy/go.mod docker/lambda/tsproxy/go.sum ./
RUN go mod download
COPY docker/lambda/tsproxy/main.go ./
# CGO_ENABLED=0 to keep the binary static and copy-able into the slim
# Python runtime without also pulling in glibc. -s -w strips debug info
# to shave a few MB off the image (tsnet pulls in a lot of Go code).
RUN CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o /out/tsproxy .

# --- Stage 1: Python dependency install ------------------------------------
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
    "mcp>=1.25" \
    "numpy>=2.0"

# --- Stage 2: runtime ------------------------------------------------------
FROM public.ecr.aws/docker/library/python:3.12-slim

# AWS Lambda Web Adapter — intercepts Lambda events, forwards HTTP to PORT=8080.
# Placed at /opt/extensions/ so Lambda auto-loads it as an extension.
COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:0.8.4 /lambda-adapter /opt/extensions/lambda-adapter

# tsnet-based bolt forwarder for the PC-local Neo4j (Phase 2 of the
# Aura -> PC-Tailscale migration). Inactive (exits 0) if TS_AUTHKEY or
# TS_UPSTREAM is unset, so the image is safe to run even before
# Tailscale is provisioned.
COPY --from=tsproxy-builder /out/tsproxy /opt/tsproxy

# Entrypoint choreography: start tsproxy in the background, then exec
# uvicorn. See docker/lambda/entrypoint.sh for the details + the
# Lambda-init-budget rationale.
COPY docker/lambda/entrypoint.sh /opt/entrypoint.sh
RUN chmod +x /opt/entrypoint.sh /opt/tsproxy

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

# /opt/entrypoint.sh handles tsproxy boot + hands off to uvicorn via exec.
# Using the script (not a CMD array) means Lambda signals still reach
# uvicorn because of the final `exec` in the script.
ENTRYPOINT ["/opt/entrypoint.sh"]
