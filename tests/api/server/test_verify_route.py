"""Integration tests for the async POST /api/v2/verify surface (D2).

Pre-D2 this file tested the blocking /verify that returned a
DocumentVerdict. Post-D2 (plan §5.2) POST is a 202 producer — the
pipeline runs on a self-async-invoke, not on the request thread.

What still matches the pre-D2 file:
* pydantic validation rails (422 on oversize / unknown option / bad
  rigor enum)
* GET /api/v2/verdict/{id} 404 shape

What changed:
* Success is 202 + ``{"job_id": "<uuid>"}``, not 200 + DocumentVerdict.
* There is no "pipeline failure → 503" path from POST /verify
  anymore — the pipeline only runs inside the async handler.
* Retention is enforced at the ``jobs.create_job`` write, not at
  ``verdict_store.put``. Retain=false → DynamoDB record has text="".

Tests stub ``jobs.create_job`` / ``jobs.async_invoke_verify`` so no
boto3 client ever spins up.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient

_SRC_API = Path(__file__).resolve().parents[3] / "src" / "api"
if str(_SRC_API) not in sys.path:
    sys.path.insert(0, str(_SRC_API))

# Cross-worktree import guard (same pattern as tests/unit/conftest.py).
# tests/api/ does not have its own conftest, and an earlier test file in
# the api/ subtree may have imported ``server.*`` / ``config.*`` from the
# main checkout before THIS worktree's src/api was put on sys.path. Purge
# the cached flat packages so the imports below reload from this
# worktree; guard on ``interfaces`` so we don't re-purge a cache that's
# already correct (which would break earlier files' bound class
# identities — see D1 handoff §6 for the symptom).
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

from adapters.verdict_store_memory import InMemoryVerdictStore  # noqa: E402
from config.dependencies import get_pipeline, get_verdict_store  # noqa: E402
from models.results import DocumentVerdict  # noqa: E402
from server.app import create_app  # noqa: E402

# Import jobs at module level so the binding shares identity with the
# `from server import jobs` in server.routes.verify — if we imported it
# inside the fixture, a tests/unit conftest purge that ran between
# collection and runtime could leave us patching a DIFFERENT server.jobs
# module than the one the route uses.
from server import jobs as _jobs  # noqa: E402


# ---------------------------------------------------------------------------
# Harness: capture jobs.* calls so we don't need boto3 / DynamoDB.
# ---------------------------------------------------------------------------


class _JobsRecorder:
    def __init__(self) -> None:
        self.created: list[dict[str, Any]] = []
        self.invoked: list[dict[str, Any]] = []


@pytest.fixture
def jobs_env(monkeypatch: pytest.MonkeyPatch) -> _JobsRecorder:
    """Replace jobs.create_job + jobs.async_invoke_verify with recorders.
    Patches the ``_jobs`` module bound at file-top import time, which is
    the same object ``server.routes.verify`` and
    ``server.routes.async_verify`` hold under their own ``jobs`` name.

    ``fake_async_invoke`` returns 202 to mirror a healthy async-queue
    accept — D3 added a StatusCode check on the POST handler so any
    non-202 is treated as an async-invoke failure."""
    recorder = _JobsRecorder()

    def fake_create_job(*, job_id: str, text: str, ticket: str) -> None:
        recorder.created.append({"job_id": job_id, "text": text, "ticket": ticket})

    def fake_async_invoke(
        *, job_id: str, text: str, ticket: str, depth: int = 1
    ) -> int:
        recorder.invoked.append(
            {"job_id": job_id, "text": text, "ticket": ticket, "depth": depth}
        )
        return 202

    def fake_fail_job(job_id: str, error: str) -> None:
        # Belt-and-braces: the POST handler's failure path calls fail_job,
        # which otherwise hits the real DynamoDB KeyError on JOBS_TABLE_NAME.
        pass

    monkeypatch.setattr(_jobs, "create_job", fake_create_job)
    monkeypatch.setattr(_jobs, "async_invoke_verify", fake_async_invoke)
    monkeypatch.setattr(_jobs, "fail_job", fake_fail_job)
    return recorder


@pytest.fixture
def app_client(jobs_env: _JobsRecorder):
    app = create_app()
    # The pipeline / verdict-store dependency only matters for routes that
    # still touch them (GET /verdict legacy + /health/deep). Use the
    # lightweight in-memory store; no real pipeline is needed for the
    # POST /verify async path since we never call it.
    app.dependency_overrides[get_pipeline] = lambda: _DummyPipeline()
    app.dependency_overrides[get_verdict_store] = lambda: InMemoryVerdictStore()
    client = TestClient(app, raise_server_exceptions=False)
    return client, jobs_env


class _DummyPipeline:
    """Used only by non-verify routes (health, legacy /verdict). POST
    /verify no longer depends on a Pipeline instance at all."""

    async def verify(self, *args: Any, **kwargs: Any) -> DocumentVerdict:
        raise AssertionError(
            "POST /verify must not call the pipeline — the async handler "
            "is the only pipeline consumer under D2."
        )


# ---------------------------------------------------------------------------
# POST /api/v2/verify — new async contract.
# ---------------------------------------------------------------------------


def test_verify_returns_202_with_job_id(app_client) -> None:
    client, jobs_env = app_client
    resp = client.post("/api/v2/verify", json={"text": "Einstein was born 1879."})
    assert resp.status_code == 202
    body = resp.json()
    UUID(body["job_id"])  # valid UUID
    # create_job + async_invoke fired, exactly once each, same job_id/ticket.
    assert len(jobs_env.created) == 1
    assert len(jobs_env.invoked) == 1
    assert jobs_env.created[0]["job_id"] == body["job_id"]
    assert jobs_env.invoked[0]["job_id"] == body["job_id"]
    assert jobs_env.created[0]["ticket"] == jobs_env.invoked[0]["ticket"]


def test_verify_empty_text_is_valid(app_client) -> None:
    client, _ = app_client
    resp = client.post("/api/v2/verify", json={"text": ""})
    assert resp.status_code == 202


def test_verify_rejects_oversized_text(app_client) -> None:
    client, _ = app_client
    resp = client.post("/api/v2/verify", json={"text": "x" * 50_001})
    assert resp.status_code == 422


def test_verify_rejects_unknown_option_keys(app_client) -> None:
    client, _ = app_client
    resp = client.post(
        "/api/v2/verify",
        json={"text": "hi", "options": {"unknown_key": True}},
    )
    assert resp.status_code == 422


def test_verify_rigor_rejects_invalid_values(app_client) -> None:
    client, _ = app_client
    resp = client.post(
        "/api/v2/verify",
        json={"text": "hi", "options": {"rigor": "turbo"}},
    )
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Retention — enforced at jobs.create_job, not verdict_store.
# ---------------------------------------------------------------------------


def test_retention_default_does_not_persist_raw_text(app_client) -> None:
    client, jobs_env = app_client
    resp = client.post(
        "/api/v2/verify", json={"text": "sensitive info goes here"}
    )
    assert resp.status_code == 202
    # DynamoDB record carries empty text; async-invoke payload carries the
    # real text (that one is ephemeral — it dies with the Lambda
    # invocation and does not hit persistent storage).
    assert jobs_env.created[0]["text"] == ""
    assert jobs_env.invoked[0]["text"] == "sensitive info goes here"


def test_retention_opt_in_persists_raw_text(app_client) -> None:
    client, jobs_env = app_client
    resp = client.post(
        "/api/v2/verify?retain=true", json={"text": "retained data"}
    )
    assert resp.status_code == 202
    assert jobs_env.created[0]["text"] == "retained data"
    assert jobs_env.invoked[0]["text"] == "retained data"


# ---------------------------------------------------------------------------
# GET /api/v2/verdict/{request_id} — legacy back-compat shim (unchanged).
# ---------------------------------------------------------------------------


def test_verdict_unknown_id_is_404(app_client) -> None:
    client, _ = app_client
    resp = client.get(f"/api/v2/verdict/{uuid4()}")
    assert resp.status_code == 404
    assert resp.json()["detail"]["code"] == "verdict_not_found"


def test_verdict_invalid_uuid_is_422(app_client) -> None:
    client, _ = app_client
    resp = client.get("/api/v2/verdict/not-a-uuid")
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# /health/deep — unchanged by D2.
# ---------------------------------------------------------------------------


def test_health_deep_reports_per_layer_status(app_client) -> None:
    client, _ = app_client
    response = client.get("/health/deep")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] in ("ok", "degraded", "down")
    assert "pipeline.orchestrator" in body["layers"]
    assert "L7.verdict_store" in body["layers"]
