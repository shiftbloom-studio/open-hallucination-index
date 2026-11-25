"""Depth-guard test for the internal ``POST /_internal/async-verify``
handler (Stream D2).

The handler refuses any invocation with ``depth > 1`` and writes an
``async_depth_exceeded`` error record so the poller sees terminal
state immediately. Without this guard, a misconfiguration (bug in the
async-invoke helper, an extra AWS policy change that makes the
handler recurse, …) could fire N self-invokes for a single request
— quietly burning Lambda invocations at exponential rate. The guard
is a hard 2-invocation cap per request (the initial + one defensive
margin).

Depth-guard is separate from the rest of the async-verify contract
(handler happy-path / refuse-paths live in
``test_async_verify_handler.py``) because it's the one test that MUST
keep working even when future refactors loosen other defences — it's
the last line against a runaway Lambda bill.
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


_JOB_ID = str(uuid4())
_TICKET = "ticket-abc"


class _PoisonPipeline:
    """A pipeline that records if it was ever called. The depth guard
    MUST refuse before the pipeline runs, so ``invocations`` must stay
    zero for every depth-exceeded test."""

    def __init__(self) -> None:
        self.invocations = 0

    async def verify(self, *_args: Any, **_kwargs: Any) -> Any:
        self.invocations += 1
        raise AssertionError("depth guard failed to refuse; pipeline was invoked")


@pytest.fixture
def depth_guard_env(monkeypatch: pytest.MonkeyPatch) -> Any:
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
        "async_invoke": [],
    }

    def fake_get_job(job_id: str) -> dict[str, Any] | None:
        recorded["get_job"].append(job_id)
        return {
            "job_id": _JOB_ID,
            "status": "pending",
            "phase": "queued",
            "invoke_ticket": _TICKET,
        }

    def fake_update_phase(job_id: str, phase: str) -> None:
        recorded["update_phase"].append((job_id, phase))

    def fake_complete_job(job_id: str, result: Any) -> None:
        recorded["complete_job"].append((job_id, result))

    def fake_fail_job(job_id: str, error: str) -> None:
        recorded["fail_job"].append((job_id, error))

    def fake_async_invoke(**kwargs: Any) -> None:
        # A well-behaved handler should never self-invoke from within
        # the async path; surfaces any regression loud and clear.
        recorded["async_invoke"].append(kwargs)

    monkeypatch.setattr(jobs, "get_job", fake_get_job)
    monkeypatch.setattr(jobs, "update_phase", fake_update_phase)
    monkeypatch.setattr(jobs, "complete_job", fake_complete_job)
    monkeypatch.setattr(jobs, "fail_job", fake_fail_job)
    monkeypatch.setattr(jobs, "async_invoke_verify", fake_async_invoke)

    pipeline = _PoisonPipeline()

    app = FastAPI()
    app.include_router(async_verify_mod.router)
    # Override by the original function reference — see the note in
    # test_async_verify_handler.py for why monkey-patching deps.get_pipeline
    # breaks the dependency-override match.
    app.dependency_overrides[deps.get_pipeline] = lambda: pipeline

    return TestClient(app), pipeline, recorded


def test_depth_exceeded_refuses_without_calling_pipeline(depth_guard_env) -> None:
    client, pipeline, recorded = depth_guard_env

    resp = client.post(
        "/_internal/async-verify",
        json={
            "__async_verify": True,
            "depth": 2,  # > 1 triggers the guard
            "job_id": _JOB_ID,
            "text": "anything",
            "ticket": _TICKET,
        },
    )
    # Handler returns 200 OK — Lambda must not see a failure (would
    # trigger retry-on-failure semantics) — but refuses internally.
    assert resp.status_code == 200
    assert resp.json().get("refused") is True

    # Pipeline NEVER ran.
    assert pipeline.invocations == 0
    # No phase updates, no complete.
    assert recorded["update_phase"] == []
    assert recorded["complete_job"] == []
    # Terminal error record written.
    assert len(recorded["fail_job"]) == 1
    failed_job_id, failed_err = recorded["fail_job"][0]
    assert failed_job_id == _JOB_ID
    assert "async_depth_exceeded" in failed_err
    # Handler DID NOT re-invoke itself.
    assert recorded["async_invoke"] == []


def test_depth_much_larger_still_refuses(depth_guard_env) -> None:
    # Guards aren't off-by-one: depth=10 is still refused and doesn't
    # decrement its way to 1 anywhere.
    client, pipeline, recorded = depth_guard_env

    resp = client.post(
        "/_internal/async-verify",
        json={
            "__async_verify": True,
            "depth": 10,
            "job_id": _JOB_ID,
            "text": "anything",
            "ticket": _TICKET,
        },
    )
    assert resp.status_code == 200
    assert resp.json().get("refused") is True
    assert pipeline.invocations == 0
    assert recorded["async_invoke"] == []
    assert len(recorded["fail_job"]) == 1


def test_depth_one_is_allowed(depth_guard_env) -> None:
    # Sanity: the guard must NOT refuse depth=1 (the default). If it
    # does, POST /verify is broken for every real request.
    client, pipeline, recorded = depth_guard_env

    resp = client.post(
        "/_internal/async-verify",
        json={
            "__async_verify": True,
            "depth": 1,
            "job_id": _JOB_ID,
            "text": "anything",
            "ticket": _TICKET,
        },
    )
    # Pipeline will raise (it's _PoisonPipeline) so the handler's
    # exception branch fires fail_job. What matters HERE: the guard
    # did NOT pre-empt and the pipeline WAS attempted — proving depth=1
    # is allowed through the guard.
    assert resp.status_code == 200
    assert pipeline.invocations == 1
    # fail_job gets called because _PoisonPipeline raises.
    assert len(recorded["fail_job"]) == 1
    assert "async_depth_exceeded" not in recorded["fail_job"][0][1]
