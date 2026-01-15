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
    # TODO: Implement actual health checks for each adapter
    # This is a stub that reports the expected services
    services = {
        "llm": {"connected": False, "status": "not_configured"},
        "neo4j": {"connected": False, "status": "not_configured"},
        "qdrant": {"connected": False, "status": "not_configured"},
        "redis": {"connected": False, "status": "not_configured"},
    }

    # All services must be connected for readiness
    all_ready = all(svc.get("connected", False) for svc in services.values())

    return ReadinessStatus(
        ready=all_ready,
        services=services,
    )


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
