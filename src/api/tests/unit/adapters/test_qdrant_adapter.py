"""Unit tests for Qdrant vector store adapter."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from adapters.qdrant import QdrantVectorAdapter
from config.settings import QdrantSettings
from models.entities import Evidence


@pytest.fixture
def mock_settings():
    """Mock Qdrant settings."""
    settings = MagicMock(spec=QdrantSettings)
    settings.host = "localhost"
    settings.port = 6333
    settings.grpc_port = 6334
    settings.api_key = None
    settings.collection_name = "test_collection"
    settings.vector_size = 384
    settings.use_grpc = False
    return settings


@pytest.fixture
def mock_qdrant_client():
    """Mock Qdrant client."""
    client = MagicMock()
    # Mock query_points instead of search (actual API method)
    client.query_points = MagicMock()
    # Create a mock response with points attribute
    mock_response = MagicMock()
    mock_response.points = []
    client.query_points.return_value = mock_response
    return client


@pytest.fixture
def mock_embedding_func(mock_settings):
    """Mock embedding function."""
    import numpy as np

    async def embed(texts: list[str]) -> list[list[float]]:
        return [
            np.random.rand(mock_settings.vector_size)
            .astype("float32")
            .tolist()
            for _ in texts
        ]

    return embed


@pytest.fixture
def qdrant_store(mock_settings, mock_qdrant_client, mock_embedding_func):
    """Qdrant store with mocked client."""
    with patch("adapters.qdrant.QdrantClient", return_value=mock_qdrant_client):
        store = QdrantVectorAdapter(mock_settings, embedding_func=mock_embedding_func)
        store._client = mock_qdrant_client

        # Ensure the model and its encode method are properly mocked for tests
        store.model = MagicMock()
        store.model.encode = MagicMock()

        return store


class TestQdrantVectorAdapter:
    """Test QdrantVectorAdapter adapter."""

    def test_initialization(self, qdrant_store: QdrantVectorAdapter):
        """Test store initialization."""
        assert qdrant_store is not None
        assert qdrant_store.client is not None
        assert qdrant_store.model is not None

    @pytest.mark.asyncio
    async def test_find_evidence_semantic(self, qdrant_store: QdrantVectorAdapter):
        """Test semantic search for evidence."""
        claim = "Python is a programming language"
        
        # Mock Qdrant search result
        mock_scored_point = MagicMock()
        mock_scored_point.score = 0.92
        mock_scored_point.id = "test_id_1"
        mock_scored_point.payload = {
            "text": "Python is a high-level programming language",
            "title": "Python (programming language)",
            "url": "https://en.wikipedia.org/wiki/Python",
        }
        
        # Mock the response structure
        mock_response = MagicMock()
        mock_response.points = [mock_scored_point]
        qdrant_store.client.query_points.return_value = mock_response
        
        evidence = await qdrant_store.find_evidence(claim, limit=5)
        
        assert len(evidence) > 0
        assert isinstance(evidence[0], Evidence)
        assert evidence[0].similarity_score >= 0.0

    @pytest.mark.asyncio
    async def test_find_evidence_empty(self, qdrant_store: QdrantVectorAdapter):
        """Test search with no results."""
        claim = "Nonexistent information"
        
        # Mock empty response
        mock_response = MagicMock()
        mock_response.points = []
        qdrant_store.client.query_points.return_value = mock_response
        
        evidence = await qdrant_store.find_evidence(claim, limit=5)
        
        assert len(evidence) == 0

    @pytest.mark.asyncio
    async def test_embedding_generation(self, qdrant_store: QdrantVectorAdapter):
        """Test embedding generation via public interface."""
        text = "Test text for embedding"

        # Mock empty response
        mock_response = MagicMock()
        mock_response.points = []
        qdrant_store.client.query_points.return_value = mock_response

        # Mock the embedding function to track calls
        mock_embedding = MagicMock(return_value=[0.1] * 384)
        qdrant_store._embedding_func = mock_embedding

        # Call the public method that should internally generate embeddings
        evidence = await qdrant_store.find_evidence(text, limit=1)

        assert evidence is not None
        assert isinstance(evidence, list)
        # Verify embedding function was called
        mock_embedding.assert_called_once_with(text)


class TestQdrantErrorHandling:
    """Test error handling scenarios."""

    @pytest.mark.asyncio
    async def test_search_error_handling(
        self, qdrant_store: QdrantVectorAdapter
    ):
        """Test handling of search errors."""
        qdrant_store.client.query_points.side_effect = RuntimeError("Search failed")
        
        with pytest.raises(RuntimeError):
            await qdrant_store.find_evidence("test claim")

    @pytest.mark.asyncio
    async def test_embedding_error_handling(
        self, qdrant_store: QdrantVectorAdapter
    ):
        """Test handling of embedding errors."""
        # Mock the embedding function to raise an error
        async def failing_embed(text: str) -> list[float]:
            raise RuntimeError("Embedding failed")
        
        qdrant_store._embedding_func = failing_embed
        
        with pytest.raises(RuntimeError):
            await qdrant_store.find_evidence("test text")
