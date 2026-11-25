"""Unit tests for ``GET /api/v2/verify/status/{job_id}`` (Stream D2).

Three terminal shapes + one unknown shape + one malformed-id shape:

* pending     → 200, {status, phase, created_at, updated_at}, no result
* done        → 200, adds {result, completed_at}
* error       → 200, adds {error, completed_at}
* 404         → jobs.get_job returned None (unknown id OR TTL-reaped)
* 422         → path param isn't a valid UUID

Also pinned: the response never leaks ``invoke_ticket`` or ``text``.
Those fields are internal — ``invoke_ticket`` is a capability used by
the async handler; ``text`` is user content that retention middleware
already gates.
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


_JOB_ID = "11111111-2222-4333-8444-555555555555"


@pytest.fixture
def status_client(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Any, dict[str, Any]]:
    """TestClient mounted with the verify router and ``jobs.get_job``
    replaced by a queue-backed stub. ``recorded`` lets tests prime the
    next return value and inspect what was asked for."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from server import jobs
    from server.routes import verify as verify_mod

    recorded: dict[str, Any] = {"asked": [], "next": None}

    def fake_get_job(job_id: str) -> dict[str, Any] | None:
        recorded["asked"].append(job_id)
        return recorded["next"]

    monkeypatch.setattr(jobs, "get_job", fake_get_job)

    app = FastAPI()
    app.include_router(verify_mod.router, prefix="/api/v2")
    return TestClient(app), recorded


def test_status_returns_404_when_unknown(
    status_client: tuple[Any, dict[str, Any]],
) -> None:
    client, recorded = status_client
    recorded["next"] = None
    resp = client.get(f"/api/v2/verify/status/{_JOB_ID}")
    assert resp.status_code == 404
    detail = resp.json()["detail"]
    assert detail["code"] == "job_not_found"
    # Handler looked up by the exact id
    assert recorded["asked"] == [_JOB_ID]


def test_status_returns_pending_shape(
    status_client: tuple[Any, dict[str, Any]],
) -> None:
    client, recorded = status_client
    recorded["next"] = {
        "job_id": _JOB_ID,
        "status": "pending",
        "phase": "classifying",
        "created_at": 1760000000,
        "updated_at": 1760000005,
        "text": "some user text",
        "invoke_ticket": "SECRET",
    }
    resp = client.get(f"/api/v2/verify/status/{_JOB_ID}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "pending"
    assert body["phase"] == "classifying"
    assert body["created_at"] == 1760000000
    assert body["updated_at"] == 1760000005
    # Internal fields are scrubbed
    assert "invoke_ticket" not in body
    assert "text" not in body
    # Terminal-only fields not present on pending
    assert "result" not in body
    assert "error" not in body
    assert "completed_at" not in body


def test_status_returns_done_shape_with_result(
    status_client: tuple[Any, dict[str, Any]],
) -> None:
    client, recorded = status_client
    verdict = {"document_score": 0.87, "claims": [{"p_true": 0.9}]}
    recorded["next"] = {
        "job_id": _JOB_ID,
        "status": "done",
        "phase": "assembling",
        "created_at": 1760000000,
        "updated_at": 1760000010,
        "completed_at": 1760000010,
        "result": verdict,
    }
    resp = client.get(f"/api/v2/verify/status/{_JOB_ID}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "done"
    assert body["result"] == verdict
    assert body["completed_at"] == 1760000010


def test_status_returns_error_shape_with_message(
    status_client: tuple[Any, dict[str, Any]],
) -> None:
    client, recorded = status_client
    recorded["next"] = {
        "job_id": _JOB_ID,
        "status": "error",
        "phase": "classifying",
        "created_at": 1760000000,
        "updated_at": 1760000010,
        "completed_at": 1760000010,
        "error": "NliAdapter: timeout",
    }
    resp = client.get(f"/api/v2/verify/status/{_JOB_ID}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "error"
    assert body["error"] == "NliAdapter: timeout"
    assert body["completed_at"] == 1760000010
    # No result on the error path
    assert "result" not in body


def test_status_422_on_malformed_id(
    status_client: tuple[Any, dict[str, Any]],
) -> None:
    client, _ = status_client
    resp = client.get("/api/v2/verify/status/not-a-uuid")
    assert resp.status_code == 422
