"""
Tests for Domain Entities
=========================
"""

from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from open_hallucination_index.domain.entities import (
    Claim,
    ClaimType,
    Evidence,
    EvidenceSource,
)


class TestClaim:
    """Tests for the Claim entity."""

    def test_claim_creation_minimal(self) -> None:
        """Test creating a claim with minimal fields."""
        claim = Claim(
            id=uuid4(),
            text="The sky is blue.",
            claim_type=ClaimType.UNCLASSIFIED,
        )

        assert isinstance(claim.id, UUID)
        assert claim.text == "The sky is blue."
        assert claim.claim_type == ClaimType.UNCLASSIFIED
        assert claim.confidence == 1.0  # Default

    def test_claim_creation_full(self) -> None:
        """Test creating a claim with all fields."""
        claim_id = uuid4()
        claim = Claim(
            id=claim_id,
            text="Paris is the capital of France.",
            claim_type=ClaimType.SUBJECT_PREDICATE_OBJECT,
            subject="Paris",
            predicate="is the capital of",
            object="France",
            source_span=(0, 32),
            confidence=0.95,
            normalized_form="paris is the capital of france",
        )

        assert claim.id == claim_id
        assert claim.subject == "Paris"
        assert claim.predicate == "is the capital of"
        assert claim.object == "France"
        assert claim.source_span == (0, 32)
        assert claim.confidence == 0.95
        assert claim.normalized_form == "paris is the capital of france"

    def test_claim_types(self) -> None:
        """Test all claim types are valid."""
        expected_types = {
            "SUBJECT_PREDICATE_OBJECT",
            "TEMPORAL",
            "QUANTITATIVE",
            "COMPARATIVE",
            "CAUSAL",
            "DEFINITIONAL",
            "EXISTENTIAL",
            "UNCLASSIFIED",
        }

        actual_types = {t.name for t in ClaimType}
        assert actual_types == expected_types

    def test_claim_is_immutable(self) -> None:
        """Test that claims are immutable (frozen)."""
        claim = Claim(
            id=uuid4(),
            text="Test claim",
            claim_type=ClaimType.UNCLASSIFIED,
        )

        with pytest.raises(ValidationError):
            claim.text = "Modified claim"  # type: ignore[misc]


class TestEvidence:
    """Tests for the Evidence entity."""

    def test_evidence_creation(self) -> None:
        """Test creating evidence."""
        evidence = Evidence(
            id=uuid4(),
            source=EvidenceSource.GRAPH_EXACT,
            content="Paris is the capital of France.",
        )

        assert evidence.source == EvidenceSource.GRAPH_EXACT
        assert evidence.content == "Paris is the capital of France."
        assert evidence.similarity_score is None

    def test_evidence_with_similarity(self) -> None:
        """Test evidence with similarity score."""
        evidence = Evidence(
            id=uuid4(),
            source=EvidenceSource.VECTOR_SEMANTIC,
            content="The Eiffel Tower is in Paris.",
            similarity_score=0.87,
        )

        assert evidence.similarity_score == 0.87

    def test_evidence_with_structured_data(self) -> None:
        """Test evidence with structured data."""
        evidence = Evidence(
            id=uuid4(),
            source=EvidenceSource.GRAPH_EXACT,
            content="Paris -> capital_of -> France",
            structured_data={
                "subject": "Paris",
                "predicate": "capital_of",
                "object": "France",
            },
        )

        assert evidence.structured_data["subject"] == "Paris"
        assert evidence.structured_data["predicate"] == "capital_of"
        assert evidence.structured_data["object"] == "France"

    def test_evidence_sources(self) -> None:
        """Test all evidence source types."""
        sources = [s.value for s in EvidenceSource]
        expected = [
            "graph_exact",
            "graph_inferred",
            "vector_semantic",
            "external_api",
            "cached",
            "mcp_wikipedia",
            "mcp_context7",
            "wikipedia",
            "knowledge_graph",
            "academic",
            "pubmed",
            "clinical_trials",
            "news",
            "world_bank",
            "osv",
        ]
        assert sources == expected
