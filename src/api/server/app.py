"""
FastAPI Application Factory
===========================

Creates and configures the FastAPI application with routers and middleware.
"""

from __future__ import annotations

import asyncio
import secrets
import time
from datetime import datetime, timezone
from typing import Any

import yaml
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse, Response
from fastapi.security import APIKeyHeader
from starlette.middleware.base import BaseHTTPMiddleware

from config.dependencies import lifespan_manager
from config.settings import get_settings
from server.routes import admin_router, health_router, track_router, verify_router
from server.services.live_logs import live_log_service

# API Key header
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# In-memory API key store (shared with admin routes for now)
# In production, this would be Supabase queries
from server.routes.admin import _mock_api_keys, _hash_key


class ApiKeyInfo:
    """Information about a validated API key."""

    def __init__(
        self,
        key_id: str | None,
        key_type: str,
        user_id: str | None,
        token_limit: int | None,
        tokens_used: int,
        is_env_key: bool = False,
    ):
        self.key_id = key_id
        self.key_type = key_type
        self.user_id = user_id
        self.token_limit = token_limit
        self.tokens_used = tokens_used
        self.is_env_key = is_env_key

    @property
    def tokens_remaining(self) -> int | None:
        """Get remaining tokens (None for unlimited)."""
        if self.token_limit is None:
            return None
        return max(0, self.token_limit - self.tokens_used)

    def can_use_tokens(self, amount: int) -> bool:
        """Check if the key has enough tokens for an operation."""
        if self.token_limit is None:
            return True  # Unlimited
        return self.tokens_remaining >= amount  # type: ignore


def _validate_db_key(api_key: str) -> ApiKeyInfo | None:
    """
    Validate API key against the database.

    Returns ApiKeyInfo if valid, None otherwise.
    """
    key_hash = _hash_key(api_key)

    for key_data in _mock_api_keys.values():
        if secrets.compare_digest(key_data["key_hash"], key_hash):
            # Check if active
            if not key_data["is_active"]:
                return None

            # Check expiry
            expires_at = key_data.get("expires_at")
            if expires_at and expires_at < datetime.now(timezone.utc):
                return None

            # Update last used
            key_data["last_used_at"] = datetime.now(timezone.utc)

            return ApiKeyInfo(
                key_id=key_data["id"],
                key_type=key_data["type"],
                user_id=key_data.get("user_id"),
                token_limit=key_data.get("token_limit"),
                tokens_used=key_data.get("tokens_used", 0),
                is_env_key=False,
            )

    return None


async def verify_api_key(
    request: Request,
    api_key: str | None = Depends(api_key_header),
) -> ApiKeyInfo | bool:
    """
    Verify API key if authentication is enabled.

    Checks in order:
    1. Database-backed API keys (preferred)
    2. Environment variable API key (fallback/bootstrap)

    Returns ApiKeyInfo for DB keys or True for env key.
    """
    settings = get_settings()

    # If no API key configured and no DB keys exist, skip auth
    if not settings.api.api_key and not _mock_api_keys:
        return True

    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="Missing API key",
            headers={"WWW-Authenticate": "X-API-Key"},
        )

    # Try DB-backed key first
    key_info = _validate_db_key(api_key)
    if key_info:
        # Store key info in request state for later use
        request.state.api_key_info = key_info
        return key_info

    # Fallback to environment API key
    if settings.api.api_key and secrets.compare_digest(api_key, settings.api.api_key):
        env_key_info = ApiKeyInfo(
            key_id=None,
            key_type="master",
            user_id=None,
            token_limit=None,
            tokens_used=0,
            is_env_key=True,
        )
        request.state.api_key_info = env_key_info
        return env_key_info

    raise HTTPException(
        status_code=401,
        detail="Invalid or missing API key",
        headers={"WWW-Authenticate": "X-API-Key"},
    )


def increment_token_usage(key_id: str, amount: int) -> None:
    """Increment token usage for a DB-backed API key."""
    if key_id in _mock_api_keys:
        _mock_api_keys[key_id]["tokens_used"] = _mock_api_keys[key_id].get("tokens_used", 0) + amount


class TokenUsageMiddleware(BaseHTTPMiddleware):
    """Middleware to add token usage headers to responses."""

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        response = await call_next(request)

        # Add token headers if we have key info
        if hasattr(request.state, "api_key_info"):
            key_info: ApiKeyInfo = request.state.api_key_info
            if key_info.tokens_remaining is not None:
                response.headers["X-Tokens-Remaining"] = str(key_info.tokens_remaining)
                response.headers["X-Tokens-Limit"] = str(key_info.token_limit)
            response.headers["X-Key-Type"] = key_info.key_type

        return response


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware to log all API requests and responses to the live log service."""

    # Paths to exclude from logging (to avoid noise)
    EXCLUDED_PATHS = {
        "/health",
        "/health/live",
        "/health/ready",
        "/docs",
        "/redoc",
        "/openapi.json",
        "/openapi.yaml",
        "/api/v1/admin/logs/stream",  # Don't log the log stream itself
    }

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        # Skip excluded paths
        if request.url.path in self.EXCLUDED_PATHS:
            return await call_next(request)

        # Extract key prefix for logging (anonymized)
        key_prefix = None
        api_key = request.headers.get("X-API-Key")
        if api_key and len(api_key) >= 12:
            key_prefix = api_key[:12]

        user_id = request.headers.get("X-User-Id")

        # Record start time
        start_time = time.perf_counter()

        # Process request
        try:
            response = await call_next(request)
            duration_ms = (time.perf_counter() - start_time) * 1000

            # Log the response (fire and forget)
            asyncio.create_task(
                live_log_service.log_response(
                    method=request.method,
                    path=request.url.path,
                    status_code=response.status_code,
                    duration_ms=duration_ms,
                    user_id=user_id,
                    key_prefix=key_prefix,
                )
            )

            return response

        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000

            # Log the error
            asyncio.create_task(
                live_log_service.log_error(
                    message=f"{type(e).__name__}: {str(e)}",
                    path=request.url.path,
                    details={"method": request.method, "duration_ms": duration_ms},
                )
            )
            raise


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

    # Token usage headers middleware
    app.add_middleware(TokenUsageMiddleware)

    # Request logging middleware for live admin dashboard
    app.add_middleware(RequestLoggingMiddleware)

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
    app.include_router(
        admin_router,
        prefix="/api/v1/admin",
        tags=["Admin"],
        # Admin routes handle their own auth via verify_admin_access dependency
    )

    @app.get("/openapi.yaml", include_in_schema=False)
    def openapi_yaml() -> Response:
        schema = app.openapi()
        content = yaml.safe_dump(schema, sort_keys=False, allow_unicode=True)
        return Response(content=content, media_type="application/yaml")

    return app
