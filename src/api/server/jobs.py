"""Async /verify job helpers — DynamoDB state + Lambda self-async-invoke.

Stream D2 split ``POST /api/v2/verify`` into an async handshake:

1. ``POST /api/v2/verify`` writes a pending record here, fires a
   ``lambda:InvokeFunction`` (InvocationType=Event) at the Lambda itself
   with a synthetic APIGW v2 event whose body carries the job details,
   and returns ``202 {"job_id": ...}``.
2. Lambda Web Adapter routes the event through FastAPI exactly like any
   other request. The internal route ``/_internal/async-verify`` reads
   the body sentinel, validates the ticket, and drives
   ``pipeline.verify()`` with phase callbacks that update this record at
   each of D1's five natural boundaries.
3. The caller polls ``GET /api/v2/verify/status/{job_id}`` until terminal.

Why the synthetic APIGW v2 event (not a raw JSON payload): the Lambda
uses AWS Lambda Web Adapter, which only understands HTTP-shaped events
(APIGW v1/v2, ALB, Function URL). A raw ``{"__async_verify": true, ...}``
payload would confuse the adapter and 4xx before any of our code runs.
Wrapping the sentinel inside an APIGW v2 event body keeps the
adapter happy, routes into FastAPI normally, and lets the internal
route handler do the depth-guard + ticket check in one place.

Why a ticket (capability) stored in the record rather than a long-lived
shared secret: the edge-secret middleware already gates the public
surface from the internet. Any request that reaches FastAPI at all has
already passed that gate. The ticket defends against a caller who has
the edge secret but should not be able to reach the internal path:
they cannot know the per-job random ticket unless they themselves
wrote the record. After the async handler consumes the ticket it
clears the field, so replay is single-shot.

Boto3 clients are module-level via lazy accessors so warm-start
invocations reuse the same TCP pools. Tests monkey-patch the accessors
to install in-memory stubs — that is the only supported extension
point for injecting a custom client.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import time
from typing import Any
from uuid import uuid4

import boto3

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lazy module-level boto3 clients.
#
# Creating a boto3 client pulls in botocore's slow `session.create_client`
# code path (~200 ms cold). Deferring to first use keeps module import
# time out of the Lambda cold-start critical path and, crucially, keeps
# test files importable without AWS credentials. Once created the client
# persists for the life of the Lambda sandbox — AWS recommends module-
# level clients for exactly this reuse.
# ---------------------------------------------------------------------------

_dynamodb_client: Any | None = None
_lambda_client: Any | None = None
_edge_secret_cache: str | None = None


def _get_dynamodb_client() -> Any:
    global _dynamodb_client
    if _dynamodb_client is None:
        _dynamodb_client = boto3.client("dynamodb")
    return _dynamodb_client


def _get_lambda_client() -> Any:
    global _lambda_client
    if _lambda_client is None:
        _lambda_client = boto3.client("lambda")
    return _lambda_client


def _get_edge_secret() -> str | None:
    """Return the CF edge secret value, or None in local dev.

    The edge secret is what ``EdgeSecretMiddleware`` compares requests
    against: any request reaching the Lambda over HTTP must carry
    ``X-OHI-Edge-Secret: <value>`` or the middleware 403s before the
    route handler runs. The D2 polling flow's self-async-invoke sends a
    synthetic APIGW v2 event back through the SAME Lambda — which means
    it too traverses the middleware and must carry the header. (D2's
    first prod deploy shipped without this and every async-verify 403'd
    silently because ``InvocationType=Event`` swallows downstream
    failures; see the D3 handoff for the forensic write-up.)

    Caching semantics:

    * Module-level cache + lazy resolution so the fast path on a warm
      Lambda sandbox is zero syscalls. Combined with
      ``SecretsLoader``'s own 600 s TTL cache, the secret is fetched
      once per warm container and, in practice, once per cold start.
    * Rotation: the module-level cache lives for the container's
      lifetime (minutes to ~hours). A rotated secret propagates on the
      next cold start; we do NOT actively invalidate mid-warm. If that
      becomes a problem a future stream can switch to relying solely
      on ``SecretsLoader`` TTL — but middleware reads the same
      ``get_loader().get(...)`` surface and has the same property, so
      there is nothing to gain from invalidating here alone.
    * Local dev: when ``OHI_CF_EDGE_SECRET_ARN`` is unset,
      ``EdgeSecretMiddleware`` isn't registered (see ``server.app``)
      and we return None so the caller can omit the header rather than
      send a bogus one.
    """
    global _edge_secret_cache
    if _edge_secret_cache is not None:
        return _edge_secret_cache
    if not os.environ.get("OHI_CF_EDGE_SECRET_ARN"):
        return None
    # Lazy imports: the config.* modules pull in pydantic settings and
    # boto3's SecretsManager client. Deferring until first actual use
    # keeps module-import cost off the Lambda cold-start path and keeps
    # tests that don't exercise this code able to import server.jobs
    # without AWS credentials in scope.
    from config.infra_env import edge_secret_arn
    from config.secrets_loader import get_loader

    _edge_secret_cache = get_loader().get(edge_secret_arn())
    return _edge_secret_cache


def _get_table_name() -> str:
    try:
        return os.environ["JOBS_TABLE_NAME"]
    except KeyError as exc:
        raise RuntimeError(
            "JOBS_TABLE_NAME env var not set — compute TF must surface the "
            "jobs/ layer's verify_jobs_table_name output."
        ) from exc


def _get_self_function_name() -> str:
    # AWS Lambda runtime sets this automatically on every invocation.
    try:
        return os.environ["AWS_LAMBDA_FUNCTION_NAME"]
    except KeyError as exc:
        raise RuntimeError(
            "AWS_LAMBDA_FUNCTION_NAME env var not set — this should only be "
            "reachable inside the Lambda runtime. Tests must stub it."
        ) from exc


def _get_ttl_seconds() -> int:
    raw = os.environ.get("OHI_ASYNC_VERIFY_TTL_SECONDS", "3600")
    try:
        return int(raw)
    except ValueError:
        logger.warning(
            "OHI_ASYNC_VERIFY_TTL_SECONDS=%r is not an int; falling back to 3600",
            raw,
        )
        return 3600


# ---------------------------------------------------------------------------
# Identifier helpers — one call per POST /verify.
# ---------------------------------------------------------------------------


def new_job_id() -> str:
    """UUID4 string, the primary key for the DynamoDB record."""
    return str(uuid4())


def new_invoke_ticket() -> str:
    """Random 32-byte url-safe token — see module docstring for the threat
    model. Do not log this value."""
    return secrets.token_urlsafe(32)


# ---------------------------------------------------------------------------
# Record lifecycle.
#
# DynamoDB attribute names include three SDK-reserved words (``status``,
# ``result``, ``error``) so every UpdateExpression uses
# ExpressionAttributeNames to alias them; it keeps the call shape uniform
# and means we never have to remember which of them the reserved list
# actually covers.
# ---------------------------------------------------------------------------

# Sentinel phase strings — kept here so every caller uses the same vocabulary.
PHASE_QUEUED = "queued"
PHASE_DECOMPOSING = "decomposing"
PHASE_RETRIEVING_EVIDENCE = "retrieving_evidence"
PHASE_CLASSIFYING = "classifying"
PHASE_CALIBRATING = "calibrating"
PHASE_ASSEMBLING = "assembling"


def create_job(*, job_id: str, text: str, ticket: str) -> None:
    """Write the initial ``pending`` record. Producer-only call."""
    now = int(time.time())
    item = {
        "job_id": {"S": job_id},
        "status": {"S": "pending"},
        "phase": {"S": PHASE_QUEUED},
        "text": {"S": text},
        "invoke_ticket": {"S": ticket},
        "created_at": {"N": str(now)},
        "updated_at": {"N": str(now)},
        "ttl": {"N": str(now + _get_ttl_seconds())},
    }
    _get_dynamodb_client().put_item(TableName=_get_table_name(), Item=item)


def update_phase(job_id: str, phase: str) -> None:
    """Advance ``phase`` without changing ``status``. Cheap idempotent
    write used by the pipeline's five boundary callbacks."""
    now = int(time.time())
    _get_dynamodb_client().update_item(
        TableName=_get_table_name(),
        Key={"job_id": {"S": job_id}},
        UpdateExpression="SET #phase = :phase, updated_at = :now",
        ExpressionAttributeNames={"#phase": "phase"},
        ExpressionAttributeValues={
            ":phase": {"S": phase},
            ":now": {"N": str(now)},
        },
    )


def complete_job(job_id: str, result: dict[str, Any]) -> None:
    """Terminal write for the success path. Serialises the verdict as JSON
    into a single String attribute (the whole payload is ~1–5 KB for OHI's
    typical document size so DynamoDB's 400 KB item limit is nowhere near
    binding). Clears ``invoke_ticket`` so a replayed async-invoke cannot
    re-run a finished job."""
    now = int(time.time())
    _get_dynamodb_client().update_item(
        TableName=_get_table_name(),
        Key={"job_id": {"S": job_id}},
        UpdateExpression=(
            "SET #status = :status, #result = :result, "
            "completed_at = :now, updated_at = :now, "
            "invoke_ticket = :empty"
        ),
        ExpressionAttributeNames={"#status": "status", "#result": "result"},
        ExpressionAttributeValues={
            ":status": {"S": "done"},
            ":result": {"S": json.dumps(result, default=str)},
            ":now": {"N": str(now)},
            ":empty": {"S": ""},
        },
    )


def fail_job(job_id: str, error: str) -> None:
    """Terminal write for the failure path. Same ticket-clear semantic as
    ``complete_job``."""
    now = int(time.time())
    _get_dynamodb_client().update_item(
        TableName=_get_table_name(),
        Key={"job_id": {"S": job_id}},
        UpdateExpression=(
            "SET #status = :status, #error = :error, "
            "completed_at = :now, updated_at = :now, "
            "invoke_ticket = :empty"
        ),
        ExpressionAttributeNames={"#status": "status", "#error": "error"},
        ExpressionAttributeValues={
            ":status": {"S": "error"},
            ":error": {"S": error},
            ":now": {"N": str(now)},
            ":empty": {"S": ""},
        },
    )


def get_job(job_id: str) -> dict[str, Any] | None:
    """Read the record. Returns a native-dict shape with JSON-parsed
    ``result``; returns ``None`` on cache miss (either the job never
    existed, or TTL reaped it)."""
    resp = _get_dynamodb_client().get_item(
        TableName=_get_table_name(),
        Key={"job_id": {"S": job_id}},
    )
    item = resp.get("Item")
    if not item:
        return None
    return _deserialize_job(item)


def _deserialize_job(item: dict[str, Any]) -> dict[str, Any]:
    """DynamoDB low-level JSON → plain dict.

    Only surfaces fields that exist; absent fields (``result`` / ``error``
    / ``completed_at`` on a pending record, empty ``invoke_ticket`` after
    terminal write) are dropped rather than rendered as None, so the
    GET /verify/status response stays tight."""
    out: dict[str, Any] = {
        "job_id": item["job_id"]["S"],
        "status": item["status"]["S"],
        "phase": item["phase"]["S"],
        "created_at": int(item["created_at"]["N"]),
        "updated_at": int(item["updated_at"]["N"]),
    }
    if "text" in item and item["text"].get("S"):
        out["text"] = item["text"]["S"]
    if "invoke_ticket" in item and item["invoke_ticket"].get("S"):
        out["invoke_ticket"] = item["invoke_ticket"]["S"]
    if "completed_at" in item and item["completed_at"].get("N"):
        out["completed_at"] = int(item["completed_at"]["N"])
    if "result" in item and item["result"].get("S"):
        out["result"] = json.loads(item["result"]["S"])
    if "error" in item and item["error"].get("S"):
        out["error"] = item["error"]["S"]
    return out


# ---------------------------------------------------------------------------
# Self-async-invoke — the synthetic APIGW v2 event.
# ---------------------------------------------------------------------------


_ASYNC_ROUTE = "/_internal/async-verify"


def async_invoke_verify(
    *, job_id: str, text: str, ticket: str, depth: int = 1
) -> int:
    """Fire a fire-and-forget Lambda invocation at this function itself.

    ``depth`` is relayed inside the body so the internal handler can
    refuse recursion past depth 1 — a defensive backstop in case the
    self-invoke IAM policy is ever widened or a misconfiguration makes
    the handler re-enter.

    Returns the ``StatusCode`` from boto3's invoke response. For a
    successful ``InvocationType=Event`` accept this is 202. Anything
    else means the async queue refused the payload (throttle, IAM
    regression, quota) and the caller MUST treat it as a failure —
    silently returning 202 to the user's client when the async path
    never fired is exactly how D2's first deploy hid the
    edge-secret-missing silent 403. See the log lines below for the
    CloudWatch signal an E2 smoke uses to confirm the invoke succeeded.
    """
    payload = _build_async_event(
        job_id=job_id, text=text, ticket=ticket, depth=depth
    )
    logger.info(
        "async_invoke_verify: invoking self job_id=%s depth=%s",
        job_id,
        depth,
    )
    resp = _get_lambda_client().invoke(
        FunctionName=_get_self_function_name(),
        InvocationType="Event",
        Payload=json.dumps(payload).encode("utf-8"),
    )
    status_code = int(resp.get("StatusCode", 0))
    logger.info(
        "async_invoke_verify: result status_code=%s job_id=%s",
        status_code,
        job_id,
    )
    return status_code


def _build_async_event(
    *, job_id: str, text: str, ticket: str, depth: int
) -> dict[str, Any]:
    """Synthetic APIGW v2 event targeting ``POST /_internal/async-verify``.

    Only the fields AWS Lambda Web Adapter actually reads are populated;
    other fields are left out so a future Web Adapter change that depends
    on, say, ``requestContext.accountId`` fails loudly rather than silently
    pretending to succeed."""
    body = json.dumps(
        {
            "__async_verify": True,
            "depth": depth,
            "job_id": job_id,
            "text": text,
            "ticket": ticket,
        }
    )
    headers: dict[str, str] = {
        "content-type": "application/json",
        "x-ohi-async-job-id": job_id,
    }
    # The synthetic event travels back through the SAME Lambda via the
    # async queue, which means it traverses EdgeSecretMiddleware exactly
    # like an external HTTP request. Omitting this header is why D2's
    # first prod deploy silently 403'd every self-invoke and DynamoDB
    # jobs never advanced past 'queued'. Lambda Web Adapter normalises
    # header keys to lowercase on the wire; Starlette's ``request.headers``
    # is case-insensitive so either case works but we use lowercase to
    # match what Web Adapter produces for real APIGW v2 events.
    edge_secret = _get_edge_secret()
    if edge_secret is not None:
        headers["x-ohi-edge-secret"] = edge_secret
    return {
        "version": "2.0",
        "routeKey": f"POST {_ASYNC_ROUTE}",
        "rawPath": _ASYNC_ROUTE,
        "rawQueryString": "",
        "headers": headers,
        "requestContext": {
            "http": {
                "method": "POST",
                "path": _ASYNC_ROUTE,
                "protocol": "HTTP/1.1",
                "sourceIp": "127.0.0.1",
                "userAgent": "ohi-lambda-self-invoke",
            },
            "stage": "$default",
            "requestId": f"self-{job_id}",
        },
        "body": body,
        "isBase64Encoded": False,
    }
