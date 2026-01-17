"""Unit tests for ingestion data models."""
from __future__ import annotations

from pathlib import Path


from ingestion.models import (
    IngestionConfig,
    PipelineStats,
    ProcessedArticle,
    ProcessedChunk,
    WikiArticle,
    WikiInfobox,
    WikiSection,
)


class TestWikiArticle:
    """Test WikiArticle model."""

    def test_wiki_article_creation(self):
        """Test creating a wiki article."""
        article = WikiArticle(
            id=123,
            title="Test Article",
            text="Test content",
            url="https://en.wikipedia.org/wiki/Test_Article",
            revision_id=456,
        )
        assert article.id == 123
        assert article.title == "Test Article"
        assert article.text == "Test content"
        assert article.revision_id == 456


class TestWikiSection:
    """Test WikiSection model."""

    def test_section_creation(self):
        """Test creating a wiki section."""
        section = WikiSection(
            title="Introduction",
            level=2,
            content="This is the introduction.",
            subsections=[],
        )
        assert section.title == "Introduction"
        assert section.level == 2
        assert len(section.subsections) == 0

    def test_nested_sections(self):
        """Test nested section structure."""
        subsection = WikiSection(
            title="Subsection", level=3, content="Sub content", subsections=[]
        )
        parent = WikiSection(
            title="Parent",
            level=2,
            content="Parent content",
            subsections=[subsection],
        )
        assert len(parent.subsections) == 1
        assert parent.subsections[0].title == "Subsection"


class TestWikiInfobox:
    """Test WikiInfobox model."""

    def test_infobox_creation(self):
        """Test creating an infobox."""
        infobox = WikiInfobox(
            type="programming language",
            data={"name": "Python", "paradigm": "multi-paradigm"},
        )
        assert infobox.type == "programming language"
        assert infobox.data["name"] == "Python"


class TestProcessedChunk:
    """Test ProcessedChunk model."""

    def test_chunk_creation(self):
        """Test creating a processed chunk."""
        chunk = ProcessedChunk(
            chunk_id="chunk_123_0",
            text="This is a chunk of text.",
            section_title="Introduction",
            section_level=2,
            bm25_tokens=["chunk", "text"],
            bm25_weights=[0.5, 0.3],
        )
        assert chunk.chunk_id == "chunk_123_0"
        assert "chunk" in chunk.bm25_tokens


class TestProcessedArticle:
    """Test ProcessedArticle model."""

    def test_processed_article_creation(self):
        """Test creating a processed article."""
        chunk = ProcessedChunk(
            chunk_id="chunk_1_0",
            text="Test chunk",
            section_title="Test",
            section_level=2,
            bm25_tokens=["test"],
            bm25_weights=[1.0],
        )
        article = ProcessedArticle(
            article_id=1,
            title="Test",
            url="https://example.com",
            chunks=[chunk],
            sections=[],
            infoboxes=[],
            metadata={},
        )
        assert article.article_id == 1
        assert len(article.chunks) == 1


class TestIngestionConfig:
    """Test IngestionConfig model."""

    def test_config_creation(self, tmp_path: Path):
        """Test creating ingestion config."""
        config = IngestionConfig(
            neo4j_uri="bolt://localhost:7687",
            neo4j_user="neo4j",
            neo4j_password="password",
            qdrant_url="http://localhost:6333",
            qdrant_collection="test",
            download_dir=tmp_path / "downloads",
            checkpoint_file=tmp_path / "checkpoint.json",
        )
        assert config.neo4j_uri == "bolt://localhost:7687"
        assert config.qdrant_collection == "test"

    def test_config_defaults(self, tmp_path: Path):
        """Test config default values."""
        config = IngestionConfig(
            neo4j_uri="bolt://localhost:7687",
            neo4j_user="neo4j",
            neo4j_password="password",
            qdrant_url="http://localhost:6333",
            qdrant_collection="test",
            download_dir=tmp_path / "downloads",
            checkpoint_file=tmp_path / "checkpoint.json",
        )
        # Check defaults
        assert config.embedding_batch_size > 0
        assert config.chunk_size > 0
        assert config.preprocess_workers > 0


class TestPipelineStats:
    """Test PipelineStats model."""

    def test_stats_initialization(self):
        """Test stats initialization."""
        stats = PipelineStats()
        assert stats.articles_processed == 0
        assert stats.chunks_created == 0
        assert stats.articles_uploaded == 0

    def test_stats_update(self):
        """Test updating stats."""
        stats = PipelineStats()
        stats.articles_processed = 10
        stats.chunks_created = 100
        assert stats.articles_processed == 10
        assert stats.chunks_created == 100
