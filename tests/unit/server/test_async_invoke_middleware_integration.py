"""Integration test: the full ``create_app()`` stack wrapped around
``/_internal/async-verify`` — closes the test-coverage gap E2 surfaced
in D3's handoff §5.3.

D2's 34 existing tests all mount the internal router directly onto a
bare ``FastAPI()`` instance without any middleware, which is why they
missed the silent-403 path: ``EdgeSecretMiddleware`` gates every
non-``/health/live`` route, and the synthetic self-invoke event that
``jobs.async_invoke_verify`` produces would be rejected there before
reaching the handler. The failure mode in prod was invisible because
``InvocationType=Event`` swallows downstream errors.

This test exercises the real middleware chain by calling
``create_app()`` exactly as ``server.app`` does at import time, with
AWS side effects (Secrets Manager, DynamoDB) stubbed out and
``get_pipeline`` overridden. Three behaviours are pinned:

* No header → 403 ``missing_edge_secret``; handler never runs.
* Matching header → 200; handler advances all five pipeline phases
  and lands a terminal ``complete_job``.
* Mismatched header → 403 ``invalid_edge_secret``; handler never runs.

The fixture intentionally does NOT use ``with TestClient(app):`` so
the lifespan context (which boots the real LLM / Qdrant / Neo4j
adapters and wants live AWS creds) is skipped — dependency overrides
take its place.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# sys.path + cross-worktree sys.modules purge. Same guarded pattern as
# the other tests/unit/server/* files — see tests/unit/conftest.py for
# the session-scoped layer this complements.
# ---------------------------------------------------------------------------
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


TEST_SECRET = "d3-integration-secret-abc-123"
_JOB_ID = "integration-job"
_TICKET = "integration-ticket"


class _StubVerdict:
    """Minimal DocumentVerdict stand-in. The handler only needs
    ``model_dump(mode="json")``."""

    def model_dump(self, *, mode: str = "json") -> dict[str, Any]:
        return {"document_score": 0.5, "claims": []}


class _StubPipeline:
    def __init__(self) -> None:
        self.seen_texts: list[str] = []

    async def verify(
        self, text: str, *, phase_callback: Any = None, **_kwargs: Any
    ) -> _StubVerdict:
        self.seen_texts.append(text)
        if phase_callback is not None:
            for phase in (
                "decomposing",
                "retrieving_evidence",
                "classifying",
                "calibrating",
                "assembling",
            ):
                await phase_callback(phase)
        return _StubVerdict()


@pytest.fixture
def middleware_app(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Any, dict[str, list[Any]], _StubPipeline]:
    """Build ``create_app()`` with ``EdgeSecretMiddleware`` registered
    and all AWS side effects stubbed.

    Order matters: ``OHI_CF_EDGE_SECRET_ARN`` must be set BEFORE
    ``create_app()`` runs (that's where the middleware-registration
    check fires), and the ``secrets_loader.get_loader`` monkeypatch
    must be in place BEFORE the first request dispatch (the middleware
    resolves the secret lazily, on every request).
    """
    # 1. Flip the middleware-registration branch on.
    monkeypatch.setenv(
        "OHI_CF_EDGE_SECRET_ARN",
        "arn:aws:secretsmanager:eu-central-1:000000000000:secret:test-edge-AbCdEf",
    )

    # 2. Stub the SecretsManager read. The middleware's lambda does
    # ``get_loader().get(edge_secret_arn())`` lazily per request; we
    # intercept at the ``get_loader`` factory so every loader call
    # returns our stub.
    from config import secrets_loader

    class _StubLoader:
        def get(self, _arn: str, **_: Any) -> str:
            return TEST_SECRET

    monkeypatch.setattr(secrets_loader, "get_loader", lambda: _StubLoader())

    # 3. Stub jobs.* so no real DynamoDB traffic.
    from server import jobs

    class _State:
        record: dict[str, Any] | None = None

    state = _State()
    state.record = {
        "job_id": _JOB_ID,
        "status": "pending",
        "phase": "queued",
        "invoke_ticket": _TICKET,
    }
    recorded: dict[str, list[Any]] = {
        "get_job": [],
        "update_phase": [],
        "complete_job": [],
        "fail_job": [],
    }

    def fake_get_job(job_id: str) -> dict[str, Any] | None:
        recorded["get_job"].append(job_id)
        return state.record

    def fake_update_phase(job_id: str, phase: str) -> None:
        recorded["update_phase"].append((job_id, phase))

    def fake_complete_job(job_id: str, result: dict[str, Any]) -> None:
        recorded["complete_job"].append((job_id, result))

    def fake_fail_job(job_id: str, err: str) -> None:
        recorded["fail_job"].append((job_id, err))

    monkeypatch.setattr(jobs, "get_job", fake_get_job)
    monkeypatch.setattr(jobs, "update_phase", fake_update_phase)
    monkeypatch.setattr(jobs, "complete_job", fake_complete_job)
    monkeypatch.setattr(jobs, "fail_job", fake_fail_job)

    # 4. Build the actual app factory.
    from server.app import create_app

    app = create_app()

    # 5. Dependency-override get_pipeline (lifespan is intentionally
    # NOT triggered — see module docstring).
    from config import dependencies as deps

    pipeline = _StubPipeline()
    app.dependency_overrides[deps.get_pipeline] = lambda: pipeline

    from fastapi.testclient import TestClient

    client = TestClient(app)
    return client, recorded, pipeline


def _body() -> dict[str, Any]:
    return {
        "__async_verify": True,
        "depth": 1,
        "job_id": _JOB_ID,
        "text": "Marie Curie won Nobel prizes.",
        "ticket": _TICKET,
    }


# ---------------------------------------------------------------------------
# 1. Silent-fail regression — missing header → 403, handler untouched.
# ---------------------------------------------------------------------------


def test_internal_async_verify_403s_without_edge_secret(
    middleware_app: tuple[Any, dict[str, list[Any]], _StubPipeline],
) -> None:
    """The exact silent-fail class D2's first deploy exhibited: the
    self-invoke synthetic event arrives without the header,
    EdgeSecretMiddleware 403s before the handler runs, nothing happens
    to the DynamoDB record, and the poll loop stays on ``pending queued``
    forever."""
    client, recorded, pipeline = middleware_app

    resp = client.post("/_internal/async-verify", json=_body())

    assert resp.status_code == 403
    assert resp.json()["detail"] == "missing_edge_secret"

    # Handler not reached — no ticket check, no phase updates, no
    # terminal write, pipeline never invoked.
    assert recorded["get_job"] == []
    assert recorded["update_phase"] == []
    assert recorded["complete_job"] == []
    assert recorded["fail_job"] == []
    assert pipeline.seen_texts == []


# ---------------------------------------------------------------------------
# 2. Fix verification — correct header → handler runs end to end.
# ---------------------------------------------------------------------------


def test_internal_async_verify_runs_with_edge_secret(
    middleware_app: tuple[Any, dict[str, list[Any]], _StubPipeline],
) -> None:
    """With the matching header the middleware hands off to the route
    handler, which then runs the pipeline and lands a terminal
    ``complete_job``. This is the path ``jobs.async_invoke_verify``
    MUST produce by injecting the header into its synthetic event.
    The companion unit test in ``test_jobs.py`` pins the injection."""
    client, recorded, pipeline = middleware_app

    resp = client.post(
        "/_internal/async-verify",
        json=_body(),
        headers={"x-ohi-edge-secret": TEST_SECRET},
    )

    assert resp.status_code == 200, resp.text
    assert resp.json() == {"status": "done"}

    assert pipeline.seen_texts == ["Marie Curie won Nobel prizes."]
    phases = [p for (_jid, p) in recorded["update_phase"]]
    assert phases == [
        "decomposing",
        "retrieving_evidence",
        "classifying",
        "calibrating",
        "assembling",
    ]
    assert len(recorded["complete_job"]) == 1
    assert recorded["fail_job"] == []


# ---------------------------------------------------------------------------
# 3. Wrong-header defence in depth.
# ---------------------------------------------------------------------------


def test_internal_async_verify_403s_with_wrong_edge_secret(
    middleware_app: tuple[Any, dict[str, list[Any]], _StubPipeline],
) -> None:
    """A header present but mismatched also 403s. Catches a hypothetical
    regression where hmac.compare_digest is replaced with ``==``, or
    where the expected value is hard-coded wrong."""
    client, recorded, pipeline = middleware_app

    resp = client.post(
        "/_internal/async-verify",
        json=_body(),
        headers={"x-ohi-edge-secret": "not-the-real-secret"},
    )

    assert resp.status_code == 403
    assert resp.json()["detail"] == "invalid_edge_secret"
    assert pipeline.seen_texts == []
    assert recorded["get_job"] == []
