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
    client.search = MagicMock(return_value=[])
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
        mock_scored_point.payload = {
            "text": "Python is a high-level programming language",
            "title": "Python (programming language)",
            "url": "https://en.wikipedia.org/wiki/Python",
        }
        
        qdrant_store.client.search.return_value = [mock_scored_point]
        
        evidence = await qdrant_store.find_evidence(claim, limit=5)
        
        assert len(evidence) > 0
        assert isinstance(evidence[0], Evidence)
        assert evidence[0].score >= 0.0

    @pytest.mark.asyncio
    async def test_find_evidence_empty(self, qdrant_store: QdrantVectorAdapter):
        """Test search with no results."""
        claim = "Nonexistent information"
        
        qdrant_store.client.search.return_value = []
        
        evidence = await qdrant_store.find_evidence(claim, limit=5)
        
        assert len(evidence) == 0

    @pytest.mark.asyncio
    async def test_embedding_generation(self, qdrant_store: QdrantVectorAdapter):
        """Test embedding generation via public interface."""
        text = "Test text for embedding"

        # Ensure the client search does not interfere with embedding checks
        qdrant_store.client.search.return_value = []

        # Call the public method that should internally generate embeddings
        evidence = await qdrant_store.find_evidence(text, limit=1)

        assert evidence is not None
        assert isinstance(evidence, list)
        qdrant_store.model.encode.assert_called_once()
        qdrant_store.model.encode.assert_called_with([text])


class TestQdrantErrorHandling:
    """Test error handling scenarios."""

    @pytest.mark.asyncio
    async def test_search_error_handling(
        self, qdrant_store: QdrantVectorAdapter
    ):
        """Test handling of search errors."""
        qdrant_store.client.search.side_effect = RuntimeError("Search failed")
        
        with pytest.raises(RuntimeError):
            await qdrant_store.find_evidence("test claim")

    @pytest.mark.asyncio
    async def test_embedding_error_handling(
        self, qdrant_store: QdrantVectorAdapter
    ):
        """Test handling of embedding errors."""
        qdrant_store.model.encode.side_effect = RuntimeError("Embedding failed")
        
        with pytest.raises(RuntimeError):
            await qdrant_store._generate_embedding("test text")
