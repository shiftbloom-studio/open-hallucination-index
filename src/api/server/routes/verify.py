"""Public /api/v2/verify surface — async polling flow (Stream D2).

Pre-D2 POST /verify blocked on ``pipeline.verify()`` and returned the
``DocumentVerdict`` synchronously. E1 observed ~5s latency on a 2-claim
doc, and on longer inputs the API Gateway 30s integration timeout would
504 even though the Lambda kept running (CLAUDE.md trap + D1 handoff).

Post-D2 POST /verify is just the *producer* side of an async handshake:

1. Write a pending record to DynamoDB with a fresh job_id + ticket.
2. Self-async-invoke the Lambda with a synthetic APIGW v2 event whose
   body carries the sentinel + job_id + text + ticket.
3. Return 202 + ``{"job_id": ...}`` in well under 200 ms.

The caller then polls ``GET /api/v2/verify/status/{job_id}`` at ~1Hz
until terminal (done / error). See ``src/api/server/jobs.py`` for the
DynamoDB helpers and the synthetic-event rationale.

``GET /verdict/{request_id}`` is preserved for the v1 → v2 back-compat
shim that still surfaces in ``ohi-client.ts``; no consumer of the D2
polling path hits it.
"""

from __future__ import annotations

import logging
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import ORJSONResponse

from config.dependencies import get_verdict_store
from interfaces.verdict_store import VerdictStore
from models.results import DocumentVerdict
from server import jobs
from server.schemas.verify import VerifyRequest

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/verify",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Enqueue an async verification job; returns a job_id to poll.",
    description=(
        "Accepts the verification request, writes a pending record to the "
        "ohi-verify-jobs DynamoDB table, fires a Lambda self-async-invoke "
        "that runs the full L1→L7 pipeline, and returns a job_id. "
        "Poll GET /api/v2/verify/status/{job_id} until status transitions "
        "from 'pending' to 'done' or 'error'."
    ),
    response_class=ORJSONResponse,
)
async def verify(body: VerifyRequest, request: Request) -> ORJSONResponse:
    job_id = jobs.new_job_id()
    ticket = jobs.new_invoke_ticket()

    # Retention (spec §11): raw text is NOT persisted unless ?retain=true
    # tagged request.state. The async-invoke payload (below) always carries
    # the real text because the handler needs it to run the pipeline, but
    # that payload is ephemeral — it dies with the Lambda invocation. What
    # ends up in DynamoDB honours the retention flag.
    retain_text = bool(getattr(request.state, "retain_text", False))
    persisted_text = body.text if retain_text else ""

    try:
        jobs.create_job(job_id=job_id, text=persisted_text, ticket=ticket)
    except Exception as exc:
        # DynamoDB write failure is the only thing that can stop us
        # pre-invoke. Surface as 503 so the client can retry.
        logger.exception("jobs.create_job failed for %s", job_id)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": "jobs_write_failed", "message": str(exc)},
        ) from exc

    try:
        invoke_status = jobs.async_invoke_verify(
            job_id=job_id, text=body.text, ticket=ticket, depth=1
        )
    except Exception as exc:
        # If the async-invoke fails the DynamoDB record is already in
        # 'pending' — flip it to error so the poller sees a terminal
        # state immediately rather than timing out after 3 min.
        logger.exception("jobs.async_invoke_verify failed for %s", job_id)
        try:
            jobs.fail_job(job_id, f"async_invoke_failed: {exc}")
        except Exception:  # noqa: BLE001
            logger.exception(
                "follow-up jobs.fail_job also failed for %s (record will TTL)",
                job_id,
            )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": "async_invoke_failed", "message": str(exc)},
        ) from exc

    # The boto3 invoke call itself can succeed (no exception) while the
    # async queue still refuses the payload — throttling, IAM regression,
    # reserved-concurrency ceiling hit. When that happens the downstream
    # pipeline never runs and the DynamoDB record stays at 'pending'
    # forever. Returning 202 to the caller in that state is exactly the
    # silent-fail mode D2's first deploy exhibited via a different
    # (missing edge-secret) cause. Inspect StatusCode and surface the
    # failure both to the record and to the client.
    if invoke_status != 202:
        logger.error(
            "jobs.async_invoke_verify returned non-202 StatusCode=%s "
            "for %s — failing record and returning 503.",
            invoke_status,
            job_id,
        )
        try:
            jobs.fail_job(
                job_id,
                f"async_invoke_rejected: status={invoke_status}",
            )
        except Exception:  # noqa: BLE001
            logger.exception(
                "follow-up jobs.fail_job also failed for %s (record will TTL)",
                job_id,
            )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "async_invoke_rejected",
                "message": (
                    f"async queue returned StatusCode={invoke_status}; "
                    "expected 202"
                ),
            },
        )

    return ORJSONResponse(
        status_code=status.HTTP_202_ACCEPTED,
        content={"job_id": job_id},
    )


@router.get(
    "/verify/status/{job_id}",
    status_code=status.HTTP_200_OK,
    summary="Poll the async /verify job. 404 if unknown or TTL-reaped.",
    description=(
        "Returns the current DynamoDB record for a job in a native dict "
        "shape: {status, phase, created_at, updated_at, result?, error?, "
        "completed_at?}. Polling cadence is the client's choice; the "
        "frontend uses ~1s with exponential backoff on transient errors."
    ),
    response_class=ORJSONResponse,
)
async def verify_status(job_id: UUID) -> ORJSONResponse:
    # UUID-typed path param gives us free validation; malformed → 422.
    try:
        record = jobs.get_job(str(job_id))
    except Exception as exc:
        logger.exception("jobs.get_job failed for %s", job_id)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": "jobs_read_failed", "message": str(exc)},
        ) from exc

    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "job_not_found",
                "message": (
                    f"No verify job for id {job_id} (never existed, or TTL "
                    "reaped the record)."
                ),
            },
        )

    # Drop the ticket + raw text from the response — they are internal.
    out = {k: v for k, v in record.items() if k not in {"invoke_ticket", "text"}}
    return ORJSONResponse(status_code=status.HTTP_200_OK, content=out)


@router.get(
    "/verdict/{request_id}",
    response_model=DocumentVerdict,
    status_code=status.HTTP_200_OK,
    summary="Retrieve a previously computed verdict by request_id (back-compat).",
    description=(
        "V1 back-compat shim. The D2 polling flow uses GET "
        "/verify/status/{job_id} instead. Kept because ohi-client.ts still "
        "exports .verdict(id); safe to remove once the frontend drops it."
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
