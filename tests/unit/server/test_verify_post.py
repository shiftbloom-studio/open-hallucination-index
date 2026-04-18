"""Unit tests for the refactored ``POST /api/v2/verify`` (Stream D2).

Pre-D2 the route blocked on ``pipeline.verify()`` and returned a 200 +
DocumentVerdict. Post-D2 it is the *producer* side of the async
handshake: write a pending DynamoDB record, self-async-invoke the Lambda
to run the pipeline, return 202 + ``{"job_id": ...}`` in under the API
Gateway 30s cap (E1 observed ~5s for a 2-claim doc; this refactor
shrinks the sync portion to <200 ms regardless of document length).

Fields under test:

* HTTP status 202 on success.
* Response body carries a UUID4-looking ``job_id``.
* ``jobs.create_job`` is called exactly once with the request text and
  the same ``job_id``/``ticket`` pair that is subsequently handed to
  ``jobs.async_invoke_verify``.
* ``jobs.async_invoke_verify`` is called exactly once.
* Ticket is freshly generated per request (two back-to-back POSTs
  produce two distinct tickets). Reusing a ticket across jobs would
  break the single-shot consumption semantic the async handler relies
  on.

All boto3 side effects are intercepted at the ``jobs.*`` module level;
no AWS env vars, no boto3 clients, no pipeline.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

_SRC_API = Path(__file__).resolve().parents[3] / "src" / "api"
if str(_SRC_API) not in sys.path:
    sys.path.insert(0, str(_SRC_API))

_FLAT_PACKAGES = {
    "adapters",
    "interfaces",
    "models",
    "pipeline",
    "config",
    "server",
    "services",
}
_cached_iface = sys.modules.get("interfaces")
_cached_iface_file = getattr(_cached_iface, "__file__", "") or ""
if _cached_iface is None or str(_SRC_API) not in _cached_iface_file:
    for _cached_name in list(sys.modules):
        _root = _cached_name.split(".", 1)[0]
        if _root in _FLAT_PACKAGES:
            del sys.modules[_cached_name]


from server import jobs as _jobs  # noqa: E402
from server.middleware import RetentionMiddleware  # noqa: E402
from server.routes import verify as _verify_mod  # noqa: E402


@pytest.fixture
def verify_client(monkeypatch: pytest.MonkeyPatch) -> tuple[Any, dict[str, Any]]:
    """Build a minimal FastAPI app mounting the verify router + retention
    middleware (which the retention tests below exercise), with the
    ``jobs.*`` helpers stubbed out so we can observe calls without
    touching AWS.

    Patching the module-level ``_jobs`` binding ensures we target the
    same object ``server.routes.verify`` imported — necessary because a
    conftest-triggered sys.modules purge between collection and test
    runtime can otherwise leave us patching a different ``server.jobs``
    module than the one the route uses.
    """
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    recorded: dict[str, Any] = {
        "create": [],
        "invoke": [],
    }

    def fake_create_job(*, job_id: str, text: str, ticket: str) -> None:
        recorded["create"].append({"job_id": job_id, "text": text, "ticket": ticket})

    def fake_async_invoke(
        *, job_id: str, text: str, ticket: str, depth: int = 1
    ) -> None:
        recorded["invoke"].append(
            {"job_id": job_id, "text": text, "ticket": ticket, "depth": depth}
        )

    monkeypatch.setattr(_jobs, "create_job", fake_create_job)
    monkeypatch.setattr(_jobs, "async_invoke_verify", fake_async_invoke)

    app = FastAPI()
    app.add_middleware(RetentionMiddleware)
    app.include_router(_verify_mod.router, prefix="/api/v2")
    client = TestClient(app)
    return client, recorded


def test_post_verify_returns_202_with_job_id(
    verify_client: tuple[Any, dict[str, Any]],
) -> None:
    client, _ = verify_client
    resp = client.post("/api/v2/verify", json={"text": "Marie Curie won Nobel."})
    assert resp.status_code == 202
    body = resp.json()
    assert "job_id" in body
    # UUID4 canonical form
    from uuid import UUID

    UUID(body["job_id"])


def test_post_verify_writes_pending_record_then_async_invokes(
    verify_client: tuple[Any, dict[str, Any]],
) -> None:
    client, recorded = verify_client
    resp = client.post("/api/v2/verify", json={"text": "hello"})
    assert resp.status_code == 202
    returned_job_id = resp.json()["job_id"]

    assert len(recorded["create"]) == 1
    assert len(recorded["invoke"]) == 1
    create = recorded["create"][0]
    invoke = recorded["invoke"][0]
    # Same job_id + ticket threaded end-to-end
    assert create["job_id"] == returned_job_id
    assert invoke["job_id"] == returned_job_id
    assert create["ticket"] == invoke["ticket"]
    # Retention default is OFF → DynamoDB record carries empty text, but
    # the async-invoke payload (ephemeral) still carries the real text so
    # the handler can run the pipeline. Retention-on case lives in
    # test_post_verify_persists_text_when_retain_true below.
    assert create["text"] == ""
    assert invoke["text"] == "hello"
    # Initial depth is 1
    assert invoke["depth"] == 1


def test_post_verify_persists_text_when_retain_true(
    verify_client: tuple[Any, dict[str, Any]],
) -> None:
    client, recorded = verify_client
    resp = client.post(
        "/api/v2/verify?retain=true", json={"text": "retained data"}
    )
    assert resp.status_code == 202
    create = recorded["create"][0]
    invoke = recorded["invoke"][0]
    # Retain-opt-in → both the persisted record and the in-flight payload
    # carry the real text.
    assert create["text"] == "retained data"
    assert invoke["text"] == "retained data"


def test_post_verify_generates_fresh_ticket_per_call(
    verify_client: tuple[Any, dict[str, Any]],
) -> None:
    client, recorded = verify_client
    client.post("/api/v2/verify", json={"text": "one"})
    client.post("/api/v2/verify", json={"text": "two"})
    tickets = [c["ticket"] for c in recorded["create"]]
    assert len(tickets) == 2
    assert tickets[0] != tickets[1]
    # Each ticket is long enough to be a secrets.token_urlsafe
    assert all(len(t) >= 32 for t in tickets)
