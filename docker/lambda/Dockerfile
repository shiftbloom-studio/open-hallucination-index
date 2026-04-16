# Multi-stage build for the OHI v2 Lambda container.
# Stage 1 installs dependencies against a vanilla Python image for speed.
# Stage 2 uses the AWS Lambda Python base, adds the Lambda Web Adapter layer,
# and copies in the app code.

# --- Stage 1: dependency install -------------------------------------------
FROM public.ecr.aws/docker/library/python:3.12-slim AS builder

WORKDIR /build

# IMPORTANT: base pyproject.toml pulls sentence-transformers + torch (~800MB).
# Those belong on the PC/ingestion side, NOT in Lambda. We install a TRIMMED
# runtime set directly instead of `pip install /build`. If you need a new dep
# in the Lambda image, add it here explicitly — the full pyproject is for
# local dev + ingestion, not for Lambda.
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
    "boto3>=1.34"

# --- Stage 2: runtime ------------------------------------------------------
FROM public.ecr.aws/lambda/python:3.12

# Lambda Web Adapter — intercepts Lambda events, forwards HTTP to PORT=8080
COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:0.8.4 /lambda-adapter /opt/extensions/lambda-adapter

ENV AWS_LAMBDA_EXEC_WRAPPER=/opt/extensions/lambda-adapter
ENV PORT=8080
ENV READINESS_CHECK_PATH=/health/ready
ENV PYTHONUNBUFFERED=1

# Copy deps first (cached layer when only app code changes)
COPY --from=builder /deps ${LAMBDA_TASK_ROOT}

# Copy app code. src/api is the *package root* in the flat layout; its immediate
# children (config/, server/, models/, adapters/, pipeline/, interfaces/, services/)
# each become importable top-level packages under ${LAMBDA_TASK_ROOT}.
COPY src/api/ ${LAMBDA_TASK_ROOT}/
# Strip files we don't want in the image (pyproject, README, tests).
RUN rm -f ${LAMBDA_TASK_ROOT}/pyproject.toml ${LAMBDA_TASK_ROOT}/README.md \
 && rm -rf ${LAMBDA_TASK_ROOT}/tests

# Bare imports: `uvicorn server.app:app`, NOT `api.server.app:app`.
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
