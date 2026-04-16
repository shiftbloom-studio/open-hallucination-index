"""
FastAPI Application Factory (v2 skeleton)
==========================================

v2 OHI is open-access: no JWT, no API keys on the public surface. The v1
API-key / admin / mock-key machinery has been removed. Traffic protection
arrives in Phase 1 Tasks 1.10-1.11 (rate limit, cost ceiling, retention).

v2 routes land incrementally:
  - health       : already mounted (Task 1.12 adds /health/deep)
  - verify       : Task 1.8
  - verify/stream: Task 1.9
  - feedback     : Task 4.2
  - calibration  : Task 4.8

This module stays a thin factory so route / middleware tasks can land
independently.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse

from config.dependencies import lifespan_manager
from config.settings import get_settings
from server.middleware import RetentionMiddleware
from server.routes import health_router, verify_router


def create_app() -> FastAPI:
    """Build the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title="Open Hallucination Index",
        description=(
            "Open-access fact-checking API for LLM outputs. Verifies atomic "
            "claims against curated knowledge sources and returns calibrated "
            "trust scores with full provenance."
        ),
        version="0.2.0",
        default_response_class=ORJSONResponse,
        lifespan=lifespan_manager,
    )

    # CORS — open by default; infra sub-project may lock down in production.
    cors_origins = getattr(settings.api, "cors_origins", ["*"])
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    # TODO(Task 1.10): PerIPTokenBucketMiddleware (per-IP rate limit).
    # TODO(Task 1.10): CostCeilingMiddleware (daily $ ceiling).
    # TODO(Task 1.10): InternalAuthMiddleware (internal bearer token).

    # Retention policy (Task 1.11): raw text is NOT persisted unless the
    # caller explicitly opts in via ?retain=true. See spec §11.
    app.add_middleware(RetentionMiddleware)

    # Routers (more added in Phase 1 / 4 tasks)
    app.include_router(health_router, prefix="/health", tags=["health"])
    app.include_router(verify_router, prefix="/api/v2", tags=["verify"])

    # TODO(Task 1.9): app.include_router(stream_router, prefix="/api/v2", tags=["stream"])
    # TODO(Task 4.2): app.include_router(feedback_router, prefix="/api/v2", tags=["feedback"])
    # TODO(Task 4.8): app.include_router(calibration_router, prefix="/api/v2", tags=["calibration"])

    return app


# Module-level instance for `uvicorn server.app:app` and the ohi-server console
# entrypoint defined in src/api/pyproject.toml.
app = create_app()
