"""Pytest fixtures for ingestion tests."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ingestion.models import (
    IngestionConfig,
    WikiArticle,
    WikiSection,
    WikiInfobox,
)


@pytest.fixture
def sample_wiki_article() -> WikiArticle:
    """Sample Wikipedia article for testing."""
    return WikiArticle(
        id=12345,
        title="Python (programming language)",
        text="""Python is a high-level programming language.
        
        == History ==
        Python was created by Guido van Rossum in 1991.
        
        == Features ==
        Python emphasizes code readability.""",
        url="https://en.wikipedia.org/wiki/Python_(programming_language)",
        revision_id=987654321,
    )


@pytest.fixture
def sample_wiki_section() -> WikiSection:
    """Sample wiki section for testing."""
    return WikiSection(
        title="History",
        level=2,
        content="Python was created by Guido van Rossum in 1991.",
        subsections=[],
    )


@pytest.fixture
def sample_wiki_infobox() -> WikiInfobox:
    """Sample wiki infobox for testing."""
    return WikiInfobox(
        type="programming language",
        data={
            "name": "Python",
            "paradigm": "multi-paradigm",
            "designed_by": "Guido van Rossum",
            "first_appeared": "1991",
        },
    )


@pytest.fixture
def ingestion_config(tmp_path: Path) -> IngestionConfig:
    """Ingestion configuration for testing."""
    return IngestionConfig(
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="test_password",
        qdrant_url="http://localhost:6333",
        qdrant_collection="test_collection",
        qdrant_api_key=None,
        embedding_model="sentence-transformers/all-MiniLM-L12-v2",
        embedding_batch_size=32,
        chunk_size=512,
        chunk_overlap=50,
        download_dir=tmp_path / "downloads",
        checkpoint_file=tmp_path / "checkpoint.json",
        preprocess_workers=2,
        download_workers=2,
        max_articles=100,
    )


@pytest.fixture
def mock_neo4j_driver():
    """Mock Neo4j driver for testing."""
    mock_driver = MagicMock()
    mock_session = MagicMock()
    mock_driver.session.return_value.__enter__.return_value = mock_session
    return mock_driver


@pytest.fixture
def mock_qdrant_client():
    """Mock Qdrant client for testing."""
    return MagicMock()


@pytest.fixture
def mock_sentence_transformer():
    """Mock sentence transformer model for testing."""
    import numpy as np

    mock_model = MagicMock()
    mock_model.encode.return_value = np.random.rand(1, 384).astype("float32")
    return mock_model
