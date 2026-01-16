"""
FastAPI Application Factory
===========================

Creates and configures the FastAPI application with routers and middleware.
"""

from __future__ import annotations

import asyncio
import secrets

import yaml
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse, Response
from fastapi.security import APIKeyHeader

from config.dependencies import lifespan_manager
from config.settings import get_settings
from server.routes import health_router, track_router, verify_router

# API Key header
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str | None = Depends(api_key_header)):
    """Verify API key if authentication is enabled."""
    settings = get_settings()

    # If no API key configured, skip auth
    if not settings.api.api_key:
        return True

    # Check API key
    if not api_key or not secrets.compare_digest(api_key, settings.api.api_key):
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "X-API-Key"},
        )
    return True


def create_app(*, enable_lifespan: bool = True) -> FastAPI:
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
        lifespan=lifespan_manager if enable_lifespan else None,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        default_response_class=ORJSONResponse,
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Request Timeout Middleware (240s global hard limit)
    @app.middleware("http")
    async def timeout_middleware(request: Request, call_next):
        try:
            return await asyncio.wait_for(call_next(request), timeout=240.0)
        except TimeoutError:
            return ORJSONResponse(
                status_code=504,
                content={
                    "detail": "Request timed out",
                    "message": "Operation took longer than 240 seconds",
                },
            )

    # Include routers
    app.include_router(health_router, prefix="/health", tags=["Health"])
    app.include_router(
        verify_router,
        prefix="/api/v1",
        tags=["Verification"],
        dependencies=[Depends(verify_api_key)],  # Require API key for verification
    )
    app.include_router(
        track_router,
        prefix="/api/v1",
        tags=["Knowledge Track"],
        dependencies=[Depends(verify_api_key)],  # Require API key for knowledge-track
    )

    @app.get("/openapi.yaml", include_in_schema=False)
    def openapi_yaml() -> Response:
        schema = app.openapi()
        content = yaml.safe_dump(schema, sort_keys=False, allow_unicode=True)
        return Response(content=content, media_type="application/yaml")

    return app
