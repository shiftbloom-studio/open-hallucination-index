"""POST /api/v2/verify + GET /api/v2/verdict/{request_id}. Spec §10."""

from __future__ import annotations

import hashlib
import logging
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, Request, status

from config.dependencies import get_pipeline, get_verdict_store
from interfaces.verdict_store import VerdictStore
from models.results import DocumentVerdict
from pipeline.pipeline import Pipeline
from server.schemas.verify import VerifyRequest

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/verify",
    response_model=DocumentVerdict,
    status_code=status.HTTP_200_OK,
    summary="Verify a piece of text against curated knowledge sources",
    description=(
        "Runs the v2 pipeline (L1 decompose → L2 domain route → L3 NLI → "
        "L4 PCG → L5 conformal → L6 active-learning hook → L7 assembly) "
        "and returns a calibrated DocumentVerdict per spec §9."
    ),
)
async def verify(
    body: VerifyRequest,
    request: Request,
    pipeline: Pipeline = Depends(get_pipeline),  # noqa: B008
    store: VerdictStore = Depends(get_verdict_store),  # noqa: B008
) -> DocumentVerdict:
    """Run the pipeline, persist the verdict, return it."""
    request_id = body.request_id or uuid4()
    try:
        verdict = await pipeline.verify(
            body.text,
            context=body.context,
            domain_hint=body.domain_hint,
            rigor=body.options.rigor,
            retrieval_tier=body.options.tier,
            max_claims=body.options.max_claims,
            request_id=request_id,
        )
    except NotImplementedError as exc:
        # Phase 2 hook landed without its concrete implementation. Surface
        # explicitly; don't silently degrade.
        logger.warning("Pipeline returned NotImplementedError: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": "phase_not_implemented", "message": str(exc)},
        ) from exc
    except Exception as exc:
        logger.exception("Pipeline failed for request %s", request_id)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": "pipeline_error", "message": str(exc)},
        ) from exc

    text_hash = hashlib.sha256(body.text.encode("utf-8")).hexdigest()
    # Retention: only persist the raw text if the retention middleware
    # tagged this request with retain_text=True (opt-in; default False
    # per spec §11).
    retain_text = bool(getattr(request.state, "retain_text", False))
    await store.put(
        request_id,
        verdict,
        text_hash=text_hash,
        text=body.text if retain_text else None,
    )

    return verdict


@router.get(
    "/verdict/{request_id}",
    response_model=DocumentVerdict,
    status_code=status.HTTP_200_OK,
    summary="Retrieve a previously computed verdict by request_id",
    description=(
        "Returns the DocumentVerdict stored for a given request_id. "
        "In-memory store with no TTL in Phase 1; Phase 4 adds durable "
        "storage with spec §11 retention."
    ),
)
async def get_verdict(
    request_id: UUID,
    store: VerdictStore = Depends(get_verdict_store),  # noqa: B008
) -> DocumentVerdict:
    verdict = await store.get(request_id)
    if verdict is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "verdict_not_found",
                "message": f"No verdict stored for request_id {request_id}",
            },
        )
    return verdict
