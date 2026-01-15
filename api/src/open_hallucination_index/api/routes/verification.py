"""
Verification API Endpoints
==========================

Primary API for text verification against knowledge sources.
"""

from __future__ import annotations

import logging
import time
from itertools import count
from typing import Annotated, Literal
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, Header, HTTPException, status
from pydantic import BaseModel, Field

from open_hallucination_index.application.verify_text import VerifyTextUseCase
from open_hallucination_index.domain.results import (
    CitationTrace,
    TrustScore,
    VerificationResult,
    VerificationStatus,
)
from open_hallucination_index.infrastructure.dependencies import get_verify_use_case
from open_hallucination_index.ports.verification_oracle import VerificationStrategy

router = APIRouter()
logger = logging.getLogger(__name__)
audit_logger = logging.getLogger("ohi.audit")
_request_counter = count(1000)


def _next_request_id() -> int:
    return next(_request_counter)


def _format_mode(target_sources: int | None) -> str:
    if target_sources == 6:
        return "STANDARD"
    if target_sources == 18:
        return "EXPERT"
    if target_sources is None:
        return "DEFAULT"
    return "CUSTOM"


def _format_text_preview(text: str, max_len: int = 160) -> str:
    normalized = " ".join(text.strip().split())
    if len(normalized) <= max_len:
        return normalized
    return f"{normalized[:max_len].rstrip()}…"


def _count_evidence(result: VerificationResult) -> int:
    return sum(
        len(verification.trace.supporting_evidence) + len(verification.trace.refuting_evidence)
        for verification in result.claim_verifications
    )


# -----------------------------------------------------------------------------
# Request/Response Schemas (API layer DTOs)
# -----------------------------------------------------------------------------


class VerifyTextRequest(BaseModel):
    """Request body for text verification."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=100_000,
        description="Text content to verify for factual accuracy.",
        examples=["The Eiffel Tower is located in Paris and was built in 1889."],
    )
    context: str | None = Field(
        default=None,
        max_length=10_000,
        description="Optional context to help with claim disambiguation.",
    )
    strategy: (
        Literal[
            "graph_exact",
            "vector_semantic",
            "hybrid",
            "cascading",
            "mcp_enhanced",
            "adaptive",
        ]
        | None
    ) = Field(
        default=None,
        description="Verification strategy. Defaults to configured strategy if not specified.",
    )
    use_cache: bool = Field(
        default=True,
        description="Whether to use cached results for identical/similar inputs.",
    )
    target_sources: int | None = Field(
        default=None,
        ge=1,
        le=20,
        description="Preferred number of sources to query during verification.",
    )
    skip_decomposition: bool = Field(
        default=False,
        description="Skip LLM claim decomposition and treat input text as a single claim.",
    )


class ClaimSummary(BaseModel):
    """Summarized claim for API response."""

    id: UUID
    text: str
    status: VerificationStatus
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning: str
    trace: CitationTrace | None = None


class VerifyTextResponse(BaseModel):
    """Response from text verification endpoint."""

    id: UUID
    trust_score: TrustScore
    summary: str | None
    claims: list[ClaimSummary]
    processing_time_ms: float
    cached: bool


class BatchVerifyRequest(BaseModel):
    """Request for batch verification of multiple texts."""

    texts: list[str] = Field(
        ...,
        min_length=1,
        max_length=50,
        description="List of texts to verify (max 50).",
    )
    strategy: (
        Literal[
            "graph_exact",
            "vector_semantic",
            "hybrid",
            "cascading",
            "mcp_enhanced",
            "adaptive",
        ]
        | None
    ) = None
    use_cache: bool = True


class BatchVerifyResponse(BaseModel):
    """Response from batch verification."""

    results: list[VerifyTextResponse]
    total_processing_time_ms: float


# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------


@router.post(
    "/verify",
    response_model=VerifyTextResponse,
    status_code=status.HTTP_200_OK,
    summary="Verify text for factual accuracy",
    description=(
        "Analyze text by decomposing it into claims and verifying each "
        "against the knowledge base. Returns a trust score and detailed traces."
    ),
)
async def verify_text(
    request: VerifyTextRequest,
    use_case: Annotated[VerifyTextUseCase, Depends(get_verify_use_case)],
    user_id: Annotated[str | None, Header(alias="X-User-Id")] = None,
) -> VerifyTextResponse:
    """
    Verify a text for factual accuracy.

    Decomposes the input into atomic claims, verifies each against
    graph and vector knowledge stores, and returns an aggregated
    trust score with explanations.
    """
    request_id = uuid4()
    request_seq = _next_request_id()
    start = time.perf_counter()

    # Map string strategy to enum
    strategy = None
    if request.strategy:
        strategy = VerificationStrategy(request.strategy)

    audit_logger.info(
        f'[REQUEST] ID: {request_seq} - UserID: "{user_id or "unknown"}" - '
        f'Mode: "{_format_mode(request.target_sources)}" - '
        f'Text: "{_format_text_preview(request.text)}"'
    )

    logger.info(
        "Verify request started",
        extra={
            "request_id": str(request_id),
            "text_length": len(request.text),
            "has_context": bool(request.context),
            "strategy": strategy.value if strategy else "default",
            "use_cache": request.use_cache,
            "target_sources": request.target_sources,
        },
    )

    try:
        result: VerificationResult = await use_case.execute(
            text=request.text,
            strategy=strategy,
            use_cache=request.use_cache,
            context=request.context,
            target_sources=request.target_sources,
            skip_decomposition=request.skip_decomposition,
        )
    except Exception as e:
        audit_logger.warning(f"[OUTPUT] ID: {request_seq} - RESULT: ERROR - SOURCES: 0")
        logger.error(
            "Verify request failed",
            extra={
                "request_id": str(request_id),
                "stage": "use_case.execute",
                "error_type": type(e).__name__,
            },
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Verification failed: {e!s}",
        ) from e

    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "Verify request completed",
        extra={
            "request_id": str(request_id),
            "verification_id": str(result.id),
            "processing_time_ms": round(elapsed_ms, 2),
            "cached": result.cached,
        },
    )

    audit_logger.info(
        f"[OUTPUT] ID: {request_seq} - RESULT: {result.trust_score.overall:.2f} "
        f"- SOURCES: {_count_evidence(result)}"
    )

    # Verification response trace logging
    for cv in result.claim_verifications:
        if cv.trace:
            logger.debug(f"Claim {cv.claim.id} has trace with {len(cv.trace.supporting_evidence)} supporting items")
        else:
            logger.warning(f"Claim {cv.claim.id} has NO TRACE")

    # Transform to API response
    claims = [
        ClaimSummary(
            id=cv.claim.id,
            text=cv.claim.text,
            status=cv.status,
            confidence=cv.trace.confidence,
            reasoning=cv.trace.reasoning,
            trace=cv.trace,
        )
        for cv in result.claim_verifications
    ]

    return VerifyTextResponse(
        id=result.id,
        trust_score=result.trust_score,
        summary=result.summary,
        claims=claims,
        processing_time_ms=result.processing_time_ms,
        cached=result.cached,
    )


@router.post(
    "/verify/batch",
    response_model=BatchVerifyResponse,
    status_code=status.HTTP_200_OK,
    summary="Batch verify multiple texts",
    description="Verify multiple texts in a single request for efficiency.",
)
async def verify_batch(
    request: BatchVerifyRequest,
    use_case: Annotated[VerifyTextUseCase, Depends(get_verify_use_case)],
) -> BatchVerifyResponse:
    """
    Batch verification of multiple texts.

    Processes each text independently and returns aggregated results.
    Uses a semaphore to limit concurrency and protect downstream services.
    """
    import asyncio

    # Concurrency limit to protect downstream services (LLM, vector DB, etc.)
    BATCH_CONCURRENCY_LIMIT = 10

    start = time.perf_counter()
    request_id = uuid4()

    strategy = None
    if request.strategy:
        strategy = VerificationStrategy(request.strategy)

    logger.info(
        "Batch verify request started",
        extra={
            "request_id": str(request_id),
            "batch_size": len(request.texts),
            "strategy": strategy.value if strategy else "default",
            "use_cache": request.use_cache,
        },
    )

    # Semaphore to limit concurrent verification operations
    semaphore = asyncio.Semaphore(BATCH_CONCURRENCY_LIMIT)

    async def verify_with_limit(text: str) -> VerificationResult:
        """Execute verification with concurrency limiting."""
        async with semaphore:
            return await use_case.execute(
                text=text,
                strategy=strategy,
                use_cache=request.use_cache,
            )

    # Process all texts with concurrency limit
    tasks = [verify_with_limit(text) for text in request.texts]
    gathered = await asyncio.gather(*tasks, return_exceptions=True)

    # Transform results
    responses: list[VerifyTextResponse] = []
    for item in gathered:
        if isinstance(item, BaseException):
            logger.error(
                "Batch verify item failed",
                extra={
                    "request_id": str(request_id),
                    "error_type": type(item).__name__,
                },
            )
            # Skip failed verifications
            continue

        # item is now narrowed to VerificationResult
        claims = [
            ClaimSummary(
                id=cv.claim.id,
                text=cv.claim.text,
                status=cv.status,
                confidence=cv.trace.confidence,
                reasoning=cv.trace.reasoning,
                trace=cv.trace,
            )
            for cv in item.claim_verifications
        ]

        responses.append(
            VerifyTextResponse(
                id=item.id,
                trust_score=item.trust_score,
                summary=item.summary,
                claims=claims,
                processing_time_ms=item.processing_time_ms,
                cached=item.cached,
            )
        )

    total_time = (time.perf_counter() - start) * 1000

    logger.info(
        "Batch verify request completed",
        extra={
            "request_id": str(request_id),
            "batch_size": len(request.texts),
            "completed_count": len(responses),
            "processing_time_ms": round(total_time, 2),
        },
    )

    return BatchVerifyResponse(
        results=responses,
        total_processing_time_ms=total_time,
    )


@router.get(
    "/strategies",
    response_model=list[str],
    status_code=status.HTTP_200_OK,
    summary="List available verification strategies",
)
async def list_strategies() -> list[str]:
    """Return list of available verification strategies."""
    return [s.value for s in VerificationStrategy]
