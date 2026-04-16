"""Integration tests for POST /api/v2/verify + GET /api/v2/verdict/{id}.

Drives the real FastAPI app via ASGITransport + httpx, with the pipeline
dependency overridden to a stubbed version that doesn't need Neo4j /
Qdrant / LLM. We're testing the HTTP surface, request-id round-trip,
retention middleware wiring, and error paths — not the pipeline itself
(that has its own orchestrator tests).
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

from adapters.verdict_store_memory import InMemoryVerdictStore  # noqa: E402
from config.dependencies import get_pipeline, get_verdict_store  # noqa: E402
from models.results import ClaimVerdict, DocumentVerdict  # noqa: E402
from models.entities import Claim, ClaimType  # noqa: E402
from server.app import create_app  # noqa: E402


# ---------------------------------------------------------------------------
# Stub pipeline that returns a canned DocumentVerdict
# ---------------------------------------------------------------------------


class _StubPipeline:
    def __init__(self, *, should_fail: bool = False) -> None:
        self._should_fail = should_fail

    async def verify(
        self,
        text: str,
        *,
        context: str | None = None,
        domain_hint: Any = None,
        rigor: str = "balanced",
        retrieval_tier: str = "default",
        max_claims: int = 50,
        request_id: UUID | None = None,
    ) -> DocumentVerdict:
        if self._should_fail:
            raise RuntimeError("simulated pipeline failure")
        rid = request_id or uuid4()
        claim = Claim(text="stub claim", claim_type=ClaimType.UNCLASSIFIED)
        cv = ClaimVerdict(
            claim=claim,
            p_true=0.75,
            interval=(0.0, 1.0),
            coverage_target=None,
            domain="general",
            domain_assignment_weights={"general": 1.0},
            supporting_evidence=[],
            refuting_evidence=[],
            pcg_neighbors=[],
            nli_self_consistency_variance=0.0,
            bp_validated=None,
            information_gain=0.0,
            queued_for_review=False,
            calibration_set_id=None,
            calibration_n=0,
            fallback_used="general",
        )
        return DocumentVerdict(
            document_score=0.75,
            document_interval=(0.0, 1.0),
            internal_consistency=1.0,
            claims=[cv],
            decomposition_coverage=1.0,
            processing_time_ms=5.0,
            rigor=rigor,  # type: ignore[arg-type]
            refinement_passes_executed=0,
            model_versions={"test": "stub"},
            request_id=rid,
        )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def _client_and_store(monkeypatch: pytest.MonkeyPatch):
    """TestClient with real app + stub pipeline + fresh verdict store."""
    app = create_app()
    store = InMemoryVerdictStore()
    stub = _StubPipeline()
    app.dependency_overrides[get_pipeline] = lambda: stub
    app.dependency_overrides[get_verdict_store] = lambda: store

    # create_app wires the lifespan, which would try to connect Neo4j +
    # Qdrant. TestClient as a context manager triggers lifespan, so we
    # use the non-context-manager form which does NOT run lifespan.
    client = TestClient(app, raise_server_exceptions=False)
    yield client, store


@pytest.fixture
def _failing_client(monkeypatch: pytest.MonkeyPatch):
    app = create_app()
    app.dependency_overrides[get_pipeline] = lambda: _StubPipeline(should_fail=True)
    app.dependency_overrides[get_verdict_store] = lambda: InMemoryVerdictStore()
    return TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# POST /api/v2/verify
# ---------------------------------------------------------------------------


def test_verify_returns_document_verdict(_client_and_store) -> None:
    client, _ = _client_and_store
    response = client.post(
        "/api/v2/verify",
        json={"text": "Einstein was born in 1879."},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["pipeline_version"] == "ohi-v2.0"
    assert body["document_score"] == 0.75
    assert len(body["claims"]) == 1


def test_verify_empty_text_is_valid() -> None:
    """Empty text is allowed (returns a trivial verdict with no claims)."""
    app = create_app()
    app.dependency_overrides[get_pipeline] = lambda: _StubPipeline()
    app.dependency_overrides[get_verdict_store] = lambda: InMemoryVerdictStore()
    client = TestClient(app, raise_server_exceptions=False)
    response = client.post("/api/v2/verify", json={"text": ""})
    assert response.status_code == 200


def test_verify_rejects_oversized_text() -> None:
    app = create_app()
    app.dependency_overrides[get_pipeline] = lambda: _StubPipeline()
    app.dependency_overrides[get_verdict_store] = lambda: InMemoryVerdictStore()
    client = TestClient(app, raise_server_exceptions=False)
    payload = {"text": "x" * 50_001}
    response = client.post("/api/v2/verify", json=payload)
    assert response.status_code == 422  # Pydantic validation


def test_verify_rejects_unknown_option_keys() -> None:
    """options has extra='forbid' — unknown keys are a 422."""
    app = create_app()
    app.dependency_overrides[get_pipeline] = lambda: _StubPipeline()
    app.dependency_overrides[get_verdict_store] = lambda: InMemoryVerdictStore()
    client = TestClient(app, raise_server_exceptions=False)
    response = client.post(
        "/api/v2/verify",
        json={"text": "hi", "options": {"unknown_key": True}},
    )
    assert response.status_code == 422


def test_verify_pipeline_failure_returns_503(_failing_client) -> None:
    response = _failing_client.post("/api/v2/verify", json={"text": "hi"})
    assert response.status_code == 503
    body = response.json()
    assert body["detail"]["code"] == "pipeline_error"


def test_verify_rigor_option_preserved(_client_and_store) -> None:
    client, _ = _client_and_store
    response = client.post(
        "/api/v2/verify",
        json={"text": "hi", "options": {"rigor": "maximum"}},
    )
    assert response.status_code == 200
    assert response.json()["rigor"] == "maximum"


def test_verify_rigor_rejects_invalid_values() -> None:
    app = create_app()
    app.dependency_overrides[get_pipeline] = lambda: _StubPipeline()
    app.dependency_overrides[get_verdict_store] = lambda: InMemoryVerdictStore()
    client = TestClient(app, raise_server_exceptions=False)
    response = client.post(
        "/api/v2/verify",
        json={"text": "hi", "options": {"rigor": "turbo"}},
    )
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# GET /api/v2/verdict/{request_id}
# ---------------------------------------------------------------------------


def test_verdict_round_trip(_client_and_store) -> None:
    client, _ = _client_and_store
    # Submit a verify request with an explicit request_id
    rid = str(uuid4())
    post = client.post("/api/v2/verify", json={"text": "hi", "request_id": rid})
    assert post.status_code == 200
    assert post.json()["request_id"] == rid

    # Fetch it back
    get = client.get(f"/api/v2/verdict/{rid}")
    assert get.status_code == 200
    assert get.json()["request_id"] == rid


def test_verdict_unknown_id_is_404(_client_and_store) -> None:
    client, _ = _client_and_store
    response = client.get(f"/api/v2/verdict/{uuid4()}")
    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "verdict_not_found"


def test_verdict_invalid_uuid_is_422(_client_and_store) -> None:
    client, _ = _client_and_store
    response = client.get("/api/v2/verdict/not-a-uuid")
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Retention middleware — store respects the ?retain flag
# ---------------------------------------------------------------------------


class _SpyVerdictStore(InMemoryVerdictStore):
    def __init__(self) -> None:
        super().__init__()
        self.last_text_received: str | None = None

    async def put(self, request_id, verdict, *, text_hash, text) -> None:  # type: ignore[override, no-untyped-def]
        self.last_text_received = text
        await super().put(request_id, verdict, text_hash=text_hash, text=text)


def test_retention_default_does_not_store_raw_text() -> None:
    app = create_app()
    spy = _SpyVerdictStore()
    app.dependency_overrides[get_pipeline] = lambda: _StubPipeline()
    app.dependency_overrides[get_verdict_store] = lambda: spy
    client = TestClient(app, raise_server_exceptions=False)

    response = client.post("/api/v2/verify", json={"text": "sensitive data"})
    assert response.status_code == 200
    # By default retain_text is False → store receives None
    assert spy.last_text_received is None


def test_retention_opt_in_stores_raw_text() -> None:
    app = create_app()
    spy = _SpyVerdictStore()
    app.dependency_overrides[get_pipeline] = lambda: _StubPipeline()
    app.dependency_overrides[get_verdict_store] = lambda: spy
    client = TestClient(app, raise_server_exceptions=False)

    response = client.post("/api/v2/verify?retain=true", json={"text": "retained data"})
    assert response.status_code == 200
    assert spy.last_text_received == "retained data"


# ---------------------------------------------------------------------------
# /health/deep
# ---------------------------------------------------------------------------


def test_health_deep_reports_per_layer_status() -> None:
    app = create_app()
    app.dependency_overrides[get_pipeline] = lambda: _StubPipeline()
    app.dependency_overrides[get_verdict_store] = lambda: InMemoryVerdictStore()
    client = TestClient(app, raise_server_exceptions=False)

    response = client.get("/health/deep")
    assert response.status_code == 200
    body = response.json()
    # Top-level status is always 'ok' / 'degraded' / 'down'
    assert body["status"] in ("ok", "degraded", "down")
    # Per-layer telemetry is present (at least the pipeline layer — infra
    # layers may be 'down' because Neo4j / Qdrant aren't running during
    # unit tests, but their presence is asserted).
    assert "pipeline.orchestrator" in body["layers"]
    assert "L7.verdict_store" in body["layers"]
    assert body["model_versions"]  # populated from the stub pipeline
