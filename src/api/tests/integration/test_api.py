"""
Integration Tests for API Endpoints
====================================

Tests the API endpoints using FastAPI TestClient.
"""

from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from models.entities import Claim, ClaimType
from models.results import (
    CitationTrace,
    ClaimVerification,
    TrustScore,
    VerificationResult,
    VerificationStatus,
)

# Skip if dependencies not installed
pytest.importorskip("fastapi")
pytest.importorskip("httpx")


class TestHealthEndpoints:
    """Tests for health check endpoints."""

    @pytest.fixture
    def client(self) -> TestClient:
        """Create a test client with mocked dependencies."""
        # Import here to avoid import errors during collection
        from open_hallucination_index.api.app import create_app

        app = create_app(enable_lifespan=False)

        # Create test client without running lifespan (skip DI)
        with TestClient(app, raise_server_exceptions=True) as client:
            yield client

    def test_liveness_check(self, client: TestClient) -> None:
        """Test liveness endpoint returns 200."""
        response = client.get("/health/live")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_readiness_check_no_deps(self, client: TestClient) -> None:
        """Test readiness endpoint (may fail without real dependencies)."""
        response = client.get("/health/ready")

        # Without real DI, this might return unhealthy
        assert response.status_code in [200, 503]


class TestVerificationEndpoints:
    """Tests for verification API endpoints."""

    @pytest.fixture
    def mock_verify_use_case(self) -> AsyncMock:
        """Create a mock verification use case."""
        mock = AsyncMock()

        # Set up return value
        claim = Claim(
            id=uuid4(),
            text="Paris is the capital of France.",
            claim_type=ClaimType.SUBJECT_PREDICATE_OBJECT,
            confidence=0.9,
        )

        trace = CitationTrace(
            claim_id=claim.id,
            status=VerificationStatus.SUPPORTED,
            reasoning="Found supporting evidence.",
            supporting_evidence=[],
            refuting_evidence=[],
            confidence=0.85,
            verification_strategy="hybrid",
        )

        verification = ClaimVerification(
            claim=claim,
            status=VerificationStatus.SUPPORTED,
            trace=trace,
            score_contribution=0.9,
        )

        result = VerificationResult(
            input_hash="test123",
            input_length=32,
            claim_verifications=[verification],
            trust_score=TrustScore(
                overall=0.9,
                claims_total=1,
                claims_supported=1,
                claims_refuted=0,
                claims_unverifiable=0,
                confidence=0.85,
                scoring_method="weighted_average",
            ),
            summary="1 claim verified: 1 supported",
            processing_time_ms=50.0,
            cached=False,
        )

        mock.execute.return_value = result
        return mock

    @pytest.fixture
    def client_with_mock(self, mock_verify_use_case: AsyncMock) -> TestClient:
        """Create a test client with mocked use case."""
        from open_hallucination_index.api.app import create_app
        from open_hallucination_index.infrastructure import dependencies

        app = create_app(enable_lifespan=False)

        # Override the dependency
        async def mock_get_verify_use_case():
            return mock_verify_use_case

        app.dependency_overrides[dependencies.get_verify_use_case] = mock_get_verify_use_case

        with TestClient(app) as client:
            yield client

    def test_verify_text_success(
        self,
        client_with_mock: TestClient,
        mock_verify_use_case: AsyncMock,
    ) -> None:
        """Test successful text verification."""
        response = client_with_mock.post(
            "/api/v1/verify",
            json={
                "text": "Paris is the capital of France.",
            },
        )

        assert response.status_code == 200
        data = response.json()

        assert "trust_score" in data
        assert data["trust_score"]["overall"] == 0.9
        assert data["trust_score"]["claims_supported"] == 1
        assert "claims" in data
        assert len(data["claims"]) == 1

    def test_verify_text_empty(self, client_with_mock: TestClient) -> None:
        """Test verification with empty text."""
        response = client_with_mock.post(
            "/api/v1/verify",
            json={"text": ""},
        )

        # Should fail validation
        assert response.status_code == 422

    def test_verify_text_with_strategy(
        self,
        client_with_mock: TestClient,
        mock_verify_use_case: AsyncMock,
    ) -> None:
        """Test verification with explicit strategy."""
        response = client_with_mock.post(
            "/api/v1/verify",
            json={
                "text": "The sky is blue.",
                "strategy": "graph_exact",
            },
        )

        assert response.status_code == 200

    def test_list_strategies(self, client_with_mock: TestClient) -> None:
        """Test listing available strategies."""
        response = client_with_mock.get("/api/v1/strategies")

        assert response.status_code == 200
        data = response.json()

        # Should be a list of strategy strings
        assert isinstance(data, list)
        assert "hybrid" in data
        assert "graph_exact" in data
        assert "vector_semantic" in data
