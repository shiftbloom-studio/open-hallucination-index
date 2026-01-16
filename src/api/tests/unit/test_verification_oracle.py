"""
Tests for Verification Oracle
==============================
"""

from uuid import uuid4

import pytest

from interfaces.stores import (
    GraphKnowledgeStore,
    VectorKnowledgeStore,
    VectorQuery,
)
from interfaces.verification import VerificationStrategy
from models.entities import (
    Claim,
    ClaimType,
    Evidence,
    EvidenceSource,
)
from models.results import VerificationStatus
from pipeline.oracle import (
    HybridVerificationOracle,
)


class MockGraphStore(GraphKnowledgeStore):
    """Mock graph store for testing."""

    def __init__(self, evidence: list[Evidence] | None = None) -> None:
        self._evidence = evidence or []

    async def connect(self) -> None:
        pass

    async def disconnect(self) -> None:
        pass

    async def health_check(self) -> bool:
        return True

    async def query_triplet(
        self, subject: str | None, predicate: str | None, obj: str | None
    ) -> list[dict]:
        return []

    async def find_evidence_for_claim(self, claim: Claim) -> list[Evidence]:
        return self._evidence

    async def entity_exists(self, entity_name: str) -> bool:
        return True

    async def get_entity_properties(self, entity_name: str) -> dict | None:
        return {}


class MockVectorStore(VectorKnowledgeStore):
    """Mock vector store for testing."""

    def __init__(self, evidence: list[Evidence] | None = None) -> None:
        self._evidence = evidence or []

    async def connect(self) -> None:
        pass

    async def disconnect(self) -> None:
        pass

    async def health_check(self) -> bool:
        return True

    async def search_similar(self, query: VectorQuery | str, **kwargs):
        if isinstance(query, VectorQuery):
            return []
        return []

    async def find_evidence_for_claim(self, claim: Claim) -> list[Evidence]:
        return self._evidence

    async def embed_text(self, text: str) -> list[float]:
        return [0.0]


class TestHybridVerificationOracle:
    """Tests for the HybridVerificationOracle."""

    @pytest.fixture
    def supporting_evidence(self) -> list[Evidence]:
        """Create supporting evidence."""
        return [
            Evidence(
                id=uuid4(),
                source=EvidenceSource.GRAPH_EXACT,
                content="Paris is the capital of France.",
                similarity_score=0.95,
                structured_data={"subject": "Paris", "object": "France"},
            ),
        ]

    @pytest.fixture
    def refuting_evidence(self) -> list[Evidence]:
        """Create refuting evidence."""
        return [
            Evidence(
                id=uuid4(),
                source=EvidenceSource.GRAPH_EXACT,
                content="Berlin is not the capital of France.",
                similarity_score=0.85,
            ),
        ]

    @pytest.fixture
    def sample_claim(self) -> Claim:
        """Create a sample claim."""
        return Claim(
            id=uuid4(),
            text="Paris is the capital of France.",
            claim_type=ClaimType.SUBJECT_PREDICATE_OBJECT,
            subject="Paris",
            predicate="is the capital of",
            object="France",
            confidence=0.9,
        )

    @pytest.mark.asyncio
    async def test_verify_with_supporting_evidence(
        self,
        sample_claim: Claim,
        supporting_evidence: list[Evidence],
    ) -> None:
        """Test verification finds supporting evidence."""
        oracle = HybridVerificationOracle(
            graph_store=MockGraphStore(supporting_evidence),
            vector_store=MockVectorStore(),
        )

        status, trace = await oracle.verify_claim(sample_claim)

        assert status == VerificationStatus.SUPPORTED
        assert len(trace.supporting_evidence) > 0
        assert trace.confidence > 0.5

    @pytest.mark.asyncio
    async def test_verify_no_evidence(self, sample_claim: Claim) -> None:
        """Test verification with no evidence."""
        oracle = HybridVerificationOracle(
            graph_store=MockGraphStore([]),
            vector_store=MockVectorStore([]),
        )

        status, trace = await oracle.verify_claim(sample_claim)

        assert status == VerificationStatus.UNVERIFIABLE
        assert "No relevant evidence" in trace.reasoning

    @pytest.mark.asyncio
    async def test_verify_multiple_claims(self, sample_claim: Claim) -> None:
        """Test verifying multiple claims."""
        oracle = HybridVerificationOracle(
            graph_store=MockGraphStore([]),
            vector_store=MockVectorStore([]),
        )

        another_claim = Claim(
            id=uuid4(),
            text=f"{sample_claim.text} (variant)",
            claim_type=sample_claim.claim_type,
            subject=sample_claim.subject,
            predicate=sample_claim.predicate,
            object=sample_claim.object,
        )
        claims = [sample_claim, another_claim]
        results = await oracle.verify_claims(claims)

        assert len(results) == 2
        for status, _trace in results:
            assert status == VerificationStatus.UNVERIFIABLE

    @pytest.mark.asyncio
    async def test_strategy_graph_only(
        self,
        sample_claim: Claim,
        supporting_evidence: list[Evidence],
    ) -> None:
        """Test graph-only strategy."""
        vector_evidence = [
            Evidence(
                id=uuid4(),
                source=EvidenceSource.VECTOR_SEMANTIC,
                content="Different evidence from vector",
                similarity_score=0.8,
            ),
        ]

        oracle = HybridVerificationOracle(
            graph_store=MockGraphStore(supporting_evidence),
            vector_store=MockVectorStore(vector_evidence),
            default_strategy=VerificationStrategy.GRAPH_EXACT,
        )

        status, trace = await oracle.verify_claim(sample_claim)

        # Should only use graph evidence
        assert len(trace.supporting_evidence) == len(supporting_evidence)

    @pytest.mark.asyncio
    async def test_strategy_vector_only(
        self,
        sample_claim: Claim,
        supporting_evidence: list[Evidence],
    ) -> None:
        """Test vector-only strategy."""
        oracle = HybridVerificationOracle(
            graph_store=MockGraphStore([]),
            vector_store=MockVectorStore(supporting_evidence),
            default_strategy=VerificationStrategy.VECTOR_SEMANTIC,
        )

        status, trace = await oracle.verify_claim(sample_claim)

        assert status == VerificationStatus.SUPPORTED

    @pytest.mark.asyncio
    async def test_current_strategy(self) -> None:
        """Test getting/setting strategy."""
        oracle = HybridVerificationOracle(
            default_strategy=VerificationStrategy.HYBRID,
        )

        assert oracle.current_strategy == VerificationStrategy.HYBRID

        await oracle.set_strategy(VerificationStrategy.CASCADING)
        assert oracle.current_strategy == VerificationStrategy.CASCADING

    @pytest.mark.asyncio
    async def test_health_check_no_stores(self) -> None:
        """Test health check with no stores."""
        oracle = HybridVerificationOracle(
            graph_store=None,
            vector_store=None,
            default_strategy=VerificationStrategy.HYBRID,
        )

        result = await oracle.health_check()
        assert result is False

    @pytest.mark.asyncio
    async def test_health_check_with_stores(self) -> None:
        """Test health check with mock stores."""
        oracle = HybridVerificationOracle(
            graph_store=MockGraphStore(),
            vector_store=MockVectorStore(),
        )

        result = await oracle.health_check()
        assert result is True
