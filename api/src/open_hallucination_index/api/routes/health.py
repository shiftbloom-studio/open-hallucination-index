"""
Health Check Endpoints
======================

Liveness and readiness probes for Kubernetes/container orchestration.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from fastapi import APIRouter, status
from pydantic import BaseModel, Field

from open_hallucination_index.infrastructure.config import get_settings
from open_hallucination_index.infrastructure.dependencies import (
    get_cache_provider,
    get_graph_store,
    get_llm_provider,
    get_vector_store,
)

router = APIRouter()


class HealthStatus(BaseModel):
    """Health check response model."""

    status: Literal["healthy", "degraded", "unhealthy"]
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    version: str
    environment: str
    checks: dict[str, bool] = Field(default_factory=dict)


class ReadinessStatus(BaseModel):
    """Readiness check response with service details."""

    ready: bool
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    services: dict[str, dict[str, bool | str]] = Field(default_factory=dict)


@router.get(
    "/live",
    response_model=HealthStatus,
    status_code=status.HTTP_200_OK,
    summary="Liveness probe",
    description="Check if the API is alive and responding.",
)
async def liveness() -> HealthStatus:
    """
    Liveness probe for container orchestration.

    Always returns healthy if the service is running.
    """
    settings = get_settings()
    return HealthStatus(
        status="healthy",
        version=settings.api.version,
        environment=settings.environment,
    )


@router.get(
    "/ready",
    response_model=ReadinessStatus,
    status_code=status.HTTP_200_OK,
    summary="Readiness probe",
    description="Check if all dependencies are connected and ready.",
)
async def readiness() -> ReadinessStatus:
    """
    Readiness probe checking all service dependencies.

    Returns ready=True only if all critical services are available.
    """
    settings = get_settings()

    async def check_service(
        getter,
        *,
        enabled: bool = True,
    ) -> tuple[dict[str, bool | str], bool]:
        if not enabled:
            return {"connected": False, "status": "disabled"}, True

        try:
            instance = await getter()
        except Exception:
            return {"connected": False, "status": "not_initialized"}, False

        if instance is None:
            return {"connected": False, "status": "not_initialized"}, False

        try:
            is_healthy = await instance.health_check()
            return (
                {"connected": bool(is_healthy), "status": "healthy" if is_healthy else "unhealthy"},
                bool(is_healthy),
            )
        except Exception:
            return {"connected": False, "status": "error"}, False

    services: dict[str, dict[str, bool | str]] = {}
    ready = True

    llm_status, llm_ready = await check_service(get_llm_provider)
    services["llm"] = llm_status
    ready = ready and llm_ready

    neo4j_status, neo4j_ready = await check_service(get_graph_store)
    services["neo4j"] = neo4j_status
    ready = ready and neo4j_ready

    qdrant_status, qdrant_ready = await check_service(get_vector_store)
    services["qdrant"] = qdrant_status
    ready = ready and qdrant_ready

    redis_status, redis_ready = await check_service(
        get_cache_provider,
        enabled=settings.redis.enabled,
    )
    services["redis"] = redis_status
    if settings.redis.enabled:
        ready = ready and redis_ready

    return ReadinessStatus(ready=ready, services=services)


@router.get(
    "",
    response_model=HealthStatus,
    status_code=status.HTTP_200_OK,
    summary="Basic health check",
    description="Simple health check endpoint.",
)
async def health() -> HealthStatus:
    """Basic health check - alias for liveness."""
    return await liveness()
