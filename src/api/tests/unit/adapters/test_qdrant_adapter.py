"""Unit tests for Qdrant vector store adapter."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from open_hallucination_index.adapters.qdrant_vector_store import QdrantVectorStore
from open_hallucination_index.domain.entities import Evidence


@pytest.fixture
def mock_qdrant_client():
    """Mock Qdrant client."""
    return MagicMock()


@pytest.fixture
def mock_embedding_model():
    """Mock sentence transformer."""
    import numpy as np

    mock_model = MagicMock()
    mock_model.encode.return_value = np.random.rand(384).astype("float32")
    return mock_model


@pytest.fixture
def qdrant_store(mock_qdrant_client, mock_embedding_model):
    """Qdrant store with mocked client."""
    with patch(
        "qdrant_client.QdrantClient", return_value=mock_qdrant_client
    ), patch(
        "sentence_transformers.SentenceTransformer", return_value=mock_embedding_model
    ):
        store = QdrantVectorStore(
            url="http://localhost:6333",
            collection_name="test_collection",
            embedding_model="sentence-transformers/all-MiniLM-L12-v2",
        )
        store.client = mock_qdrant_client
        store.model = mock_embedding_model
        return store


class TestQdrantVectorStore:
    """Test QdrantVectorStore adapter."""

    def test_initialization(self, qdrant_store: QdrantVectorStore):
        """Test store initialization."""
        assert qdrant_store is not None
        assert qdrant_store.client is not None
        assert qdrant_store.model is not None

    @pytest.mark.asyncio
    async def test_find_evidence_semantic(self, qdrant_store: QdrantVectorStore):
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
    async def test_find_evidence_empty(self, qdrant_store: QdrantVectorStore):
        """Test search with no results."""
        claim = "Nonexistent information"
        
        qdrant_store.client.search.return_value = []
        
        evidence = await qdrant_store.find_evidence(claim, limit=5)
        
        assert len(evidence) == 0

    @pytest.mark.asyncio
    async def test_embedding_generation(self, qdrant_store: QdrantVectorStore):
        """Test embedding generation."""
        text = "Test text for embedding"
        
        embedding = await qdrant_store._generate_embedding(text)
        
        assert embedding is not None
        assert len(embedding.shape) == 1  # 1D vector
        qdrant_store.model.encode.assert_called_once()


class TestQdrantErrorHandling:
    """Test error handling scenarios."""

    @pytest.mark.asyncio
    async def test_search_error_handling(
        self, qdrant_store: QdrantVectorStore
    ):
        """Test handling of search errors."""
        qdrant_store.client.search.side_effect = Exception("Search failed")
        
        with pytest.raises(Exception):
            await qdrant_store.find_evidence("test claim")

    @pytest.mark.asyncio
    async def test_embedding_error_handling(
        self, qdrant_store: QdrantVectorStore
    ):
        """Test handling of embedding errors."""
        qdrant_store.model.encode.side_effect = Exception("Embedding failed")
        
        with pytest.raises(Exception):
            await qdrant_store._generate_embedding("test text")
