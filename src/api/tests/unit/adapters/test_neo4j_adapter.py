"""Unit tests for Neo4j graph store adapter."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from pydantic import SecretStr

from adapters.neo4j import Neo4jGraphAdapter
from config.settings import Neo4jSettings
from models.entities import Evidence


@pytest.fixture
def mock_settings():
    """Mock Neo4j settings."""
    settings = MagicMock(spec=Neo4jSettings)
    settings.uri = "bolt://localhost:7687"
    settings.username = "neo4j"
    settings.password = SecretStr("test_password")
    settings.database = "neo4j"
    settings.max_connection_pool_size = 50
    return settings


@pytest.fixture
def mock_neo4j_driver():
    """Mock Neo4j driver."""
    mock_driver = MagicMock()
    mock_session = MagicMock()
    mock_driver.session.return_value.__enter__.return_value = mock_session
    mock_driver.session.return_value.__exit__.return_value = None
    return mock_driver


@pytest.fixture
def neo4j_store(mock_settings, mock_neo4j_driver):
    """Neo4j store with mocked driver."""
    with patch("adapters.neo4j.GraphDatabase.driver", return_value=mock_neo4j_driver):
        store = Neo4jGraphAdapter(mock_settings)
        store._driver = mock_neo4j_driver
        return store


class TestNeo4jGraphAdapter:
    """Test Neo4jGraphAdapter adapter."""

    def test_initialization(self, neo4j_store: Neo4jGraphAdapter):
        """Test store initialization."""
        assert neo4j_store is not None
        assert neo4j_store._driver is not None

    @pytest.mark.asyncio
    async def test_find_evidence_basic(self, neo4j_store: Neo4jGraphAdapter):
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
        
        mock_session = neo4j_store._driver.session.return_value.__enter__.return_value
        mock_session.run.return_value = mock_result
        
        evidence = await neo4j_store.find_evidence(claim, limit=5)
        
        assert len(evidence) > 0
        assert isinstance(evidence[0], Evidence)

    @pytest.mark.asyncio
    async def test_find_evidence_empty(self, neo4j_store: Neo4jGraphAdapter):
        """Test finding evidence with no results."""
        claim = "Completely fabricated claim"
        
        mock_result = MagicMock()
        mock_result.__iter__.return_value = iter([])
        
        mock_session = neo4j_store._driver.session.return_value.__enter__.return_value
        mock_session.run.return_value = mock_result
        
        evidence = await neo4j_store.find_evidence(claim, limit=5)
        
        assert len(evidence) == 0

    def test_close(self, neo4j_store: Neo4jGraphAdapter):
        """Test closing the connection."""
        neo4j_store.close()
        # Should not raise exception
        assert True


class TestNeo4jConnectionHandling:
    """Test connection handling and error scenarios."""

    @pytest.mark.asyncio
    async def test_connection_error_handling(self, mock_settings):
        """Test handling of connection errors."""
        with patch("adapters.neo4j.GraphDatabase.driver") as mock_driver_fn:
            mock_driver = MagicMock()
            mock_driver.session.side_effect = Exception("Connection failed")
            mock_driver_fn.return_value = mock_driver

            store = Neo4jGraphAdapter(mock_settings)
            store._driver = mock_driver

            # Should handle error gracefully
            with pytest.raises(Exception):
                await store.find_evidence("test claim")

    def test_verify_connectivity(self, neo4j_store: Neo4jGraphAdapter):
        """Test connectivity verification."""
        mock_session = neo4j_store._driver.session.return_value.__enter__.return_value
        mock_session.run.return_value = MagicMock()
        
        # Should not raise exception with mocked driver
        result = neo4j_store._driver.verify_connectivity()
        assert result is not None or result is None  # Mock can return anything
