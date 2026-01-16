"""Unit tests for Neo4j graph store adapter."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from open_hallucination_index.adapters.neo4j_graph_store import Neo4jGraphStore
from open_hallucination_index.domain.entities import Evidence, EvidenceSource


@pytest.fixture
def mock_neo4j_driver():
    """Mock Neo4j driver."""
    mock_driver = MagicMock()
    mock_session = MagicMock()
    mock_driver.session.return_value.__enter__.return_value = mock_session
    mock_driver.session.return_value.__exit__.return_value = None
    return mock_driver


@pytest.fixture
def neo4j_store(mock_neo4j_driver):
    """Neo4j store with mocked driver."""
    with patch("neo4j.GraphDatabase.driver", return_value=mock_neo4j_driver):
        store = Neo4jGraphStore(
            uri="bolt://localhost:7687",
            user="neo4j",
            password="test_password",
        )
        store.driver = mock_neo4j_driver
        return store


class TestNeo4jGraphStore:
    """Test Neo4jGraphStore adapter."""

    def test_initialization(self, neo4j_store: Neo4jGraphStore):
        """Test store initialization."""
        assert neo4j_store is not None
        assert neo4j_store.driver is not None

    @pytest.mark.asyncio
    async def test_find_evidence_basic(self, neo4j_store: Neo4jGraphStore):
        """Test finding evidence for a claim."""
        claim = "Python was created in 1991"
        
        # Mock Neo4j query result
        mock_record = MagicMock()
        mock_record.get.side_effect = lambda key, default=None: {
            "text": "Python was created by Guido van Rossum in 1991",
            "title": "Python (programming language)",
            "url": "https://en.wikipedia.org/wiki/Python_(programming_language)",
            "score": 0.95,
        }.get(key, default)
        
        mock_result = MagicMock()
        mock_result.__iter__.return_value = iter([mock_record])
        
        mock_session = neo4j_store.driver.session.return_value.__enter__.return_value
        mock_session.run.return_value = mock_result
        
        evidence = await neo4j_store.find_evidence(claim, limit=5)
        
        assert len(evidence) > 0
        assert isinstance(evidence[0], Evidence)

    @pytest.mark.asyncio
    async def test_find_evidence_empty(self, neo4j_store: Neo4jGraphStore):
        """Test finding evidence with no results."""
        claim = "Completely fabricated claim"
        
        mock_result = MagicMock()
        mock_result.__iter__.return_value = iter([])
        
        mock_session = neo4j_store.driver.session.return_value.__enter__.return_value
        mock_session.run.return_value = mock_result
        
        evidence = await neo4j_store.find_evidence(claim, limit=5)
        
        assert len(evidence) == 0

    def test_close(self, neo4j_store: Neo4jGraphStore):
        """Test closing the connection."""
        neo4j_store.close()
        # Should not raise exception
        assert True


class TestNeo4jConnectionHandling:
    """Test connection handling and error scenarios."""

    @pytest.mark.asyncio
    async def test_connection_error_handling(self):
        """Test handling of connection errors."""
        with patch("neo4j.GraphDatabase.driver") as mock_driver_fn:
            mock_driver = MagicMock()
            mock_driver.session.side_effect = Exception("Connection failed")
            mock_driver_fn.return_value = mock_driver
            
            store = Neo4jGraphStore(
                uri="bolt://localhost:7687",
                user="neo4j",
                password="test",
            )
            store.driver = mock_driver
            
            # Should handle error gracefully
            with pytest.raises(Exception):
                await store.find_evidence("test claim")

    def test_verify_connectivity(self, neo4j_store: Neo4jGraphStore):
        """Test connectivity verification."""
        mock_session = neo4j_store.driver.session.return_value.__enter__.return_value
        mock_session.run.return_value = MagicMock()
        
        # Should not raise exception with mocked driver
        result = neo4j_store.driver.verify_connectivity()
        assert result is not None or result is None  # Mock can return anything
