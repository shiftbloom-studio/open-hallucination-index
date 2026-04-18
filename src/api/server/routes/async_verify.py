"""Internal ``POST /_internal/async-verify`` route — consumer side of the
async polling handshake (Stream D2).

Reached ONLY via Lambda self-async-invoke. The producer
(``POST /api/v2/verify``) writes a pending DynamoDB record, fires a
``lambda:InvokeFunction`` at this Lambda with a synthetic APIGW v2
event, and returns 202 to its own caller. Lambda Web Adapter routes
that event through FastAPI to this handler; the handler drives
``pipeline.verify()`` with a phase callback that advances the
DynamoDB ``phase`` field at each of the five natural boundaries, and
writes the terminal ``done`` or ``error`` record when verify returns.

Defences against misuse (external CF traffic routed here, IAM
regression, replay, runaway self-invoke):

* **Depth guard.** The producer sends ``depth=1``; any ``depth > 1``
  is refused and a terminal error record lands immediately. Hard cap
  prevents exponential self-invoke if the IAM policy ever widens.
* **Ticket check.** The per-job random ticket must match what POST
  /verify stored; otherwise the handler refuses without running the
  pipeline. An external caller who has the edge secret still cannot
  forge invocations for jobs they didn't create.
* **Replay guard.** The handler refuses if ``status != pending``; once
  complete_job / fail_job runs it clears ``invoke_ticket`` so a
  replayed async-invoke lands the already-consumed branch.

Every refuse / error path returns HTTP 200 with a small JSON payload.
Lambda's async-invoke retry policy treats non-200 as "retry up to 2
times with exponential backoff" — exactly the wrong thing to do for a
verdict that already landed an error record.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, status
from fastapi.responses import ORJSONResponse
from pydantic import BaseModel, ConfigDict, Field

from config.dependencies import get_pipeline
from pipeline.pipeline import Pipeline
from server import jobs

logger = logging.getLogger(__name__)

router = APIRouter()


class _AsyncVerifyRequest(BaseModel):
    """Body shape that the POST /api/v2/verify producer packs into a
    synthetic APIGW v2 event. The leading-double-underscore sentinel is
    aliased because pydantic (like Python itself) reserves ``__*`` field
    names for itself — see ``jobs.async_invoke_verify`` for the
    producer side."""

    sentinel: bool = Field(alias="__async_verify")
    depth: int
    job_id: str
    text: str
    ticket: str

    model_config = ConfigDict(populate_by_name=True, extra="forbid")


def _refuse(reason: str) -> ORJSONResponse:
    return ORJSONResponse(
        status_code=status.HTTP_200_OK,
        content={"refused": True, "reason": reason},
    )


@router.post(
    "/_internal/async-verify",
    status_code=status.HTTP_200_OK,
    include_in_schema=False,  # Internal — don't surface in OpenAPI.
)
async def async_verify(
    body: _AsyncVerifyRequest,
    pipeline: Pipeline = Depends(get_pipeline),  # noqa: B008
) -> ORJSONResponse:
    # Entry log: the ABSENCE of this line in CloudWatch is the clearest
    # possible signal that the handler is not being reached (e.g. middleware
    # 403'ing before dispatch, Web Adapter routing mismatch, Lambda warm-pool
    # stuck on an older image). D2's first deploy silent-failed for exactly
    # this reason; keeping the log line before every guard means regressions
    # on "request never reached the handler" show up loudly in observability.
    logger.info(
        "async-verify entry: depth=%s job_id=%s", body.depth, body.job_id
    )

    # 1. Depth guard FIRST — before any DynamoDB read, before anything
    #    that could itself self-invoke. This is a hard $-cap.
    if body.depth > 1:
        logger.error(
            "async_verify: refusing depth=%d invocation for %s",
            body.depth,
            body.job_id,
        )
        try:
            jobs.fail_job(body.job_id, "async_depth_exceeded")
        except Exception:  # noqa: BLE001
            logger.exception("fail_job itself failed for %s", body.job_id)
        return _refuse("depth_exceeded")

    # 2. Record lookup + capability check.
    try:
        record = jobs.get_job(body.job_id)
    except Exception:  # noqa: BLE001
        logger.exception("jobs.get_job failed for %s", body.job_id)
        return _refuse("jobs_read_failed")
    if record is None:
        logger.warning("async_verify: unknown job %s", body.job_id)
        return _refuse("not_found")
    if record.get("invoke_ticket") != body.ticket:
        logger.warning(
            "async_verify: ticket mismatch for %s (possibly replayed)",
            body.job_id,
        )
        return _refuse("ticket_mismatch")
    if record.get("status") != "pending":
        logger.warning(
            "async_verify: %s already in status=%s — refusing replay",
            body.job_id,
            record.get("status"),
        )
        return _refuse("already_consumed")

    # 3. Happy path — run the pipeline with a phase callback that
    #    ADVANCES the DynamoDB record in parallel.
    async def _advance_phase(phase: str) -> None:
        try:
            jobs.update_phase(body.job_id, phase)
        except Exception:  # noqa: BLE001
            # _safe_fire in pipeline.py swallows, but belt-and-braces here
            # so a local raise never leaks up to pipeline.verify.
            logger.exception(
                "update_phase(%s, %s) failed", body.job_id, phase
            )

    try:
        verdict = await pipeline.verify(body.text, phase_callback=_advance_phase)
    except Exception as exc:  # noqa: BLE001
        logger.exception("pipeline.verify failed for %s", body.job_id)
        try:
            jobs.fail_job(body.job_id, f"{type(exc).__name__}: {exc}")
        except Exception:  # noqa: BLE001
            logger.exception("fail_job after pipeline failure also failed")
        return ORJSONResponse(
            status_code=status.HTTP_200_OK,
            content={"status": "error"},
        )

    # 4. Terminal success write.
    try:
        serialised: dict[str, Any] = verdict.model_dump(mode="json")
    except Exception:  # noqa: BLE001
        # If model_dump itself fails the verdict is unusable; treat as error.
        logger.exception("verdict.model_dump failed for %s", body.job_id)
        try:
            jobs.fail_job(body.job_id, "verdict_serialisation_failed")
        except Exception:  # noqa: BLE001
            logger.exception("fail_job after model_dump failure also failed")
        return ORJSONResponse(
            status_code=status.HTTP_200_OK,
            content={"status": "error"},
        )

    try:
        jobs.complete_job(body.job_id, serialised)
    except Exception:  # noqa: BLE001
        logger.exception("complete_job failed for %s", body.job_id)
        return ORJSONResponse(
            status_code=status.HTTP_200_OK,
            content={"status": "error"},
        )

    return ORJSONResponse(
        status_code=status.HTTP_200_OK,
        content={"status": "done"},
    )
