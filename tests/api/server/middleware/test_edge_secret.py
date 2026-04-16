"""Tests for EdgeSecretMiddleware."""
from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from server.middleware.edge_secret import EdgeSecretMiddleware


@pytest.fixture
def app_with_middleware():
    app = FastAPI()

    def get_secret() -> str:
        return "correct-secret-abc123"

    app.add_middleware(EdgeSecretMiddleware, get_expected_secret=get_secret)

    @app.get("/hello")
    def hello():
        return {"msg": "hi"}

    return app


def test_rejects_missing_header(app_with_middleware):
    client = TestClient(app_with_middleware)
    r = client.get("/hello")
    assert r.status_code == 403
    assert r.json()["detail"] == "missing_edge_secret"


def test_rejects_wrong_header(app_with_middleware):
    client = TestClient(app_with_middleware)
    r = client.get("/hello", headers={"X-OHI-Edge-Secret": "wrong"})
    assert r.status_code == 403
    assert r.json()["detail"] == "invalid_edge_secret"


def test_accepts_correct_header(app_with_middleware):
    client = TestClient(app_with_middleware)
    r = client.get("/hello", headers={"X-OHI-Edge-Secret": "correct-secret-abc123"})
    assert r.status_code == 200
    assert r.json() == {"msg": "hi"}


def test_health_live_is_exempt(app_with_middleware):
    """`/health/live` must work without the header (Lambda's own readiness probes)."""
    @app_with_middleware.get("/health/live")
    def live():
        return {"status": "live"}

    client = TestClient(app_with_middleware)
    r = client.get("/health/live")
    assert r.status_code == 200


def test_timing_safe_comparison_used(app_with_middleware, monkeypatch):
    """Ensure hmac.compare_digest is used (non-timing-attack-vulnerable)."""
    import hmac

    called = []
    real_compare = hmac.compare_digest

    def spy(a, b):
        called.append((a, b))
        return real_compare(a, b)

    monkeypatch.setattr("server.middleware.edge_secret.hmac.compare_digest", spy)

    client = TestClient(app_with_middleware)
    client.get("/hello", headers={"X-OHI-Edge-Secret": "correct-secret-abc123"})
    assert called, "hmac.compare_digest must be used for the comparison"
