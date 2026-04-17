"""
Health Check Endpoints
======================

Liveness and readiness probes for Kubernetes/container orchestration.
"""

from __future__ import annotations

from datetime import UTC, datetime
from time import perf_counter
from typing import Literal

from fastapi import APIRouter, Depends, status
from pydantic import BaseModel, Field

from config.dependencies import (
    get_graph_store,
    get_llm_provider,
    get_pipeline,
    get_trace_store,
    get_vector_store,
)
from config.settings import get_settings
from pipeline.pipeline import Pipeline

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

    # Redis readiness via the trace store (v1 cache provider has been
    # removed; Task 1.10 re-adds a v2 Redis cache for DocumentVerdict).
    redis_status, redis_ready = await check_service(
        get_trace_store,
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


# ---------------------------------------------------------------------------
# Deep health check (Task 1.12, spec §10)
# ---------------------------------------------------------------------------


class LayerStatus(BaseModel):
    """Per-layer health telemetry."""

    status: Literal["ok", "degraded", "down", "skipped"]
    latency_ms: float | None = None
    detail: str | None = None


class DeepHealthStatus(BaseModel):
    """Detailed per-layer health + model versions. Matches spec §10."""

    status: Literal["ok", "degraded", "down"]
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    layers: dict[str, LayerStatus] = Field(default_factory=dict)
    model_versions: dict[str, str] = Field(default_factory=dict)
    calibration_freshness_hours: float | None = None


@router.get(
    "/deep",
    response_model=DeepHealthStatus,
    status_code=status.HTTP_200_OK,
    summary="Deep health — per-layer status + latency + model versions",
    description=(
        "Exercises every pipeline layer with a synthetic payload and "
        "returns per-layer latency + status, plus model versions and "
        "calibration freshness. Used by ops for SLO monitoring and by "
        "the frontend's status page."
    ),
)
async def deep_health(
    pipeline: Pipeline = Depends(get_pipeline),  # noqa: B008
) -> DeepHealthStatus:
    """Run a synthetic verify through the real pipeline to probe each layer."""
    from config.dependencies import get_verdict_store

    settings = get_settings()
    layers: dict[str, LayerStatus] = {}
    overall: Literal["ok", "degraded", "down"] = "ok"

    # Simple probes for the infra layers (graph / vector / llm / trace).
    async def probe(
        name: str,
        getter,
        *,
        enabled: bool = True,
    ) -> None:
        nonlocal overall
        if not enabled:
            layers[name] = LayerStatus(status="skipped", detail="disabled by config")
            return
        start = perf_counter()
        try:
            # get_* dependency functions are synchronous — they return the
            # singleton adapter instance populated during lifespan startup.
            instance = getter()
        except Exception as exc:
            layers[name] = LayerStatus(status="down", detail=f"init: {exc}")
            overall = "degraded"
            return
        if instance is None:
            layers[name] = LayerStatus(status="down", detail="not_initialized")
            overall = "degraded"
            return
        try:
            ok = await instance.health_check()
            latency = (perf_counter() - start) * 1000.0
            layers[name] = LayerStatus(
                status="ok" if ok else "down",
                latency_ms=round(latency, 2),
                detail=None if ok else "health_check returned False",
            )
            if not ok:
                overall = "degraded"
        except Exception as exc:
            layers[name] = LayerStatus(status="down", detail=f"probe: {exc}")
            overall = "degraded"

    await probe("L1.decompose", get_llm_provider)
    await probe("L1.retrieve.neo4j", get_graph_store)
    await probe("L1.retrieve.qdrant", get_vector_store)
    await probe(
        "L1.retrieve.trace",
        get_trace_store,
        enabled=settings.redis.enabled,
    )

    # Pipeline-level probe: verify that the orchestrator resolves
    # and that a synthetic empty-text verify returns a valid result.
    start = perf_counter()
    try:
        synth = await pipeline.verify("")
        layers["pipeline.orchestrator"] = LayerStatus(
            status="ok",
            latency_ms=round((perf_counter() - start) * 1000.0, 2),
            detail=None,
        )
        model_versions = dict(synth.model_versions)
    except Exception as exc:
        layers["pipeline.orchestrator"] = LayerStatus(status="down", detail=str(exc))
        overall = "degraded"
        model_versions = {}

    # Verdict store — always in-memory in Phase 1, should never fail
    start = perf_counter()
    try:
        get_verdict_store()
        layers["L7.verdict_store"] = LayerStatus(
            status="ok",
            latency_ms=round((perf_counter() - start) * 1000.0, 2),
        )
    except Exception as exc:
        layers["L7.verdict_store"] = LayerStatus(status="down", detail=str(exc))
        overall = "degraded"

    return DeepHealthStatus(
        status=overall,
        layers=layers,
        model_versions=model_versions,
        # Phase 4 fills this once calibration refits have a persistent timestamp.
        calibration_freshness_hours=None,
    )
