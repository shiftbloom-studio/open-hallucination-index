"""Unit tests for the internal ``POST /_internal/async-verify`` handler
(Stream D2).

This is the *consumer* side of the async handshake. The Lambda
self-invoke sends a synthetic APIGW v2 event whose body is
``{__async_verify, depth, job_id, text, ticket}``; Lambda Web Adapter
forwards the body to this route; the handler drives
``pipeline.verify()`` with a phase callback, persists the result (or
error) into DynamoDB, and returns a small opaque status payload that
only Lambda itself ever sees (the caller has already returned 202 to
its client).

Covered here:

* **Happy path** — pipeline returns a verdict; the handler fires
  ``jobs.update_phase`` at every pipeline boundary and lands
  ``jobs.complete_job`` with a JSON-serialised verdict at the end.
* **Pipeline raises** — the handler captures the exception and calls
  ``jobs.fail_job`` with a ``TypeName: message`` string; it MUST NOT
  let the exception propagate back to Lambda (which would surface as
  a failure in the AsyncInvoke DLQ, not in our own record).
* **Unknown job_id** — handler refuses: no pipeline call, no terminal
  write (the record doesn't exist to write to).
* **Ticket mismatch** — handler refuses: no pipeline call, logs a
  warning. Defence-in-depth against a caller that has the edge secret
  but should not be able to trigger async jobs for a job_id that
  doesn't belong to them.
* **Job already consumed** — handler refuses: replay protection.

Depth-guard ruggedness lives in ``test_async_depth_guard.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from uuid import uuid4

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


# ---------------------------------------------------------------------------
# Stub pipeline.
# ---------------------------------------------------------------------------


class _StubVerdict:
    """Minimal DocumentVerdict stand-in. The handler only needs
    ``model_dump(mode="json")`` to shove it into DynamoDB."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def model_dump(self, *, mode: str = "json") -> dict[str, Any]:
        return dict(self._payload)


class _StubPipeline:
    def __init__(
        self, *, verdict: _StubVerdict | None = None, raises: Exception | None = None
    ) -> None:
        self._verdict = verdict
        self._raises = raises
        self.seen_texts: list[str] = []
        self.phases_seen: list[str] = []

    async def verify(self, text: str, *, phase_callback=None, **_kwargs: Any) -> _StubVerdict:
        self.seen_texts.append(text)
        # Drive the phase_callback through all five boundaries to mimic
        # the real pipeline; this pins the handler-pipeline contract.
        for phase in (
            "decomposing",
            "retrieving_evidence",
            "classifying",
            "calibrating",
            "assembling",
        ):
            self.phases_seen.append(phase)
            if phase_callback is not None:
                await phase_callback(phase)
        if self._raises is not None:
            raise self._raises
        assert self._verdict is not None  # caller promised one
        return self._verdict


# ---------------------------------------------------------------------------
# Fixtures.
# ---------------------------------------------------------------------------


_JOB_ID = str(uuid4())
_TICKET = "ticket-abc"


@pytest.fixture
def async_handler_env(
    monkeypatch: pytest.MonkeyPatch,
) -> Any:
    """Build a FastAPI TestClient with the internal router mounted and
    both ``jobs.*`` and ``get_pipeline`` stubbed."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from config import dependencies as deps
    from server import jobs
    from server.routes import async_verify as async_verify_mod

    recorded: dict[str, Any] = {
        "get_job": [],
        "update_phase": [],
        "complete_job": [],
        "fail_job": [],
    }

    class _Harness:
        pipeline: _StubPipeline | None = None
        job_record: dict[str, Any] | None = None

    harness = _Harness()

    def fake_get_job(job_id: str) -> dict[str, Any] | None:
        recorded["get_job"].append(job_id)
        return harness.job_record

    def fake_update_phase(job_id: str, phase: str) -> None:
        recorded["update_phase"].append((job_id, phase))

    def fake_complete_job(job_id: str, result: dict[str, Any]) -> None:
        recorded["complete_job"].append((job_id, result))

    def fake_fail_job(job_id: str, error: str) -> None:
        recorded["fail_job"].append((job_id, error))

    monkeypatch.setattr(jobs, "get_job", fake_get_job)
    monkeypatch.setattr(jobs, "update_phase", fake_update_phase)
    monkeypatch.setattr(jobs, "complete_job", fake_complete_job)
    monkeypatch.setattr(jobs, "fail_job", fake_fail_job)

    def fake_get_pipeline():
        assert harness.pipeline is not None, "test forgot to set harness.pipeline"
        return harness.pipeline

    app = FastAPI()
    app.include_router(async_verify_mod.router)
    # Override by the *original* function identity — FastAPI's Depends
    # captured that reference at route registration time. Monkey-patching
    # deps.get_pipeline would change what the key points to, breaking the
    # override match.
    app.dependency_overrides[deps.get_pipeline] = fake_get_pipeline

    client = TestClient(app)
    return client, harness, recorded


def _body(**overrides: Any) -> dict[str, Any]:
    base = {
        "__async_verify": True,
        "depth": 1,
        "job_id": _JOB_ID,
        "text": "Marie Curie won Nobel prizes.",
        "ticket": _TICKET,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Happy path.
# ---------------------------------------------------------------------------


def test_happy_path_fires_five_phases_then_completes(async_handler_env) -> None:
    client, harness, recorded = async_handler_env
    harness.job_record = {
        "job_id": _JOB_ID,
        "status": "pending",
        "phase": "queued",
        "invoke_ticket": _TICKET,
    }
    verdict = _StubVerdict({"document_score": 0.87, "claims": []})
    harness.pipeline = _StubPipeline(verdict=verdict)

    resp = client.post("/_internal/async-verify", json=_body())
    assert resp.status_code == 200

    # update_phase fired for every pipeline boundary, in order.
    phases = [p for (_jid, p) in recorded["update_phase"]]
    assert phases == [
        "decomposing",
        "retrieving_evidence",
        "classifying",
        "calibrating",
        "assembling",
    ]
    # complete_job called exactly once with the serialised verdict.
    assert len(recorded["complete_job"]) == 1
    completed_job_id, completed_result = recorded["complete_job"][0]
    assert completed_job_id == _JOB_ID
    assert completed_result == {"document_score": 0.87, "claims": []}
    # fail_job untouched on success.
    assert recorded["fail_job"] == []


def test_pipeline_exception_routes_to_fail_job(async_handler_env) -> None:
    client, harness, recorded = async_handler_env
    harness.job_record = {
        "job_id": _JOB_ID,
        "status": "pending",
        "phase": "queued",
        "invoke_ticket": _TICKET,
    }
    harness.pipeline = _StubPipeline(raises=RuntimeError("pipeline boom"))

    resp = client.post("/_internal/async-verify", json=_body())
    # Handler absorbs the exception — Lambda must see a 200, not a 500,
    # otherwise async-invoke retries kick in per the Lambda default
    # retry policy (up to 2 retries with exponential backoff) and we
    # end up re-running a verdict that was already captured.
    assert resp.status_code == 200

    assert recorded["complete_job"] == []
    assert len(recorded["fail_job"]) == 1
    failed_job_id, failed_err = recorded["fail_job"][0]
    assert failed_job_id == _JOB_ID
    assert "RuntimeError" in failed_err
    assert "pipeline boom" in failed_err


# ---------------------------------------------------------------------------
# Refuse paths — do NOT touch pipeline.
# ---------------------------------------------------------------------------


def test_refuses_when_job_not_found(async_handler_env) -> None:
    client, harness, recorded = async_handler_env
    harness.job_record = None  # jobs.get_job → None
    # Pipeline must never run.
    harness.pipeline = _StubPipeline(verdict=_StubVerdict({}))

    resp = client.post("/_internal/async-verify", json=_body())
    assert resp.status_code == 200
    body = resp.json()
    assert body.get("refused") is True

    assert recorded["update_phase"] == []
    assert recorded["complete_job"] == []
    assert harness.pipeline.seen_texts == []


def test_refuses_on_ticket_mismatch(async_handler_env) -> None:
    client, harness, recorded = async_handler_env
    harness.job_record = {
        "job_id": _JOB_ID,
        "status": "pending",
        "phase": "queued",
        "invoke_ticket": "DIFFERENT",
    }
    harness.pipeline = _StubPipeline(verdict=_StubVerdict({}))

    resp = client.post("/_internal/async-verify", json=_body(ticket="what-i-sent"))
    assert resp.status_code == 200
    assert resp.json().get("refused") is True
    assert harness.pipeline.seen_texts == []
    assert recorded["complete_job"] == []


def test_refuses_when_already_consumed(async_handler_env) -> None:
    client, harness, recorded = async_handler_env
    harness.job_record = {
        "job_id": _JOB_ID,
        "status": "done",
        "phase": "assembling",
        "invoke_ticket": "",  # cleared after complete_job ran
    }
    harness.pipeline = _StubPipeline(verdict=_StubVerdict({}))

    resp = client.post("/_internal/async-verify", json=_body())
    assert resp.status_code == 200
    assert resp.json().get("refused") is True
    assert harness.pipeline.seen_texts == []
    assert recorded["complete_job"] == []
