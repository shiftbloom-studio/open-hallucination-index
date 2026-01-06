"""
FastAPI Application Factory
===========================

Creates and configures the FastAPI application with routers and middleware.
"""

from __future__ import annotations

import yaml
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from open_hallucination_index.api.routes import health, verification
from open_hallucination_index.infrastructure.config import get_settings
from open_hallucination_index.infrastructure.dependencies import lifespan_manager


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    Returns:
        Configured FastAPI application instance.
    """
    settings = get_settings()

    app = FastAPI(
        title=settings.api.title,
        version=settings.api.version,
        description=(
            "Middleware API for verifying LLM-generated text against trusted "
            "knowledge sources. Detects hallucinations by decomposing text into "
            "claims and validating each against graph and vector knowledge stores."
        ),
        debug=settings.api.debug,
        lifespan=lifespan_manager,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(health.router, prefix="/health", tags=["Health"])
    app.include_router(verification.router, prefix="/api/v1", tags=["Verification"])

    @app.get("/openapi.yaml", include_in_schema=False)
    def openapi_yaml() -> Response:
        schema = app.openapi()
        content = yaml.safe_dump(schema, sort_keys=False, allow_unicode=True)
        return Response(content=content, media_type="application/yaml")

    return app
