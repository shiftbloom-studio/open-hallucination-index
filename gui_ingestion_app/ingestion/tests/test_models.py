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
        )
        assert article.id == 123
        assert article.title == "Test Article"
        assert article.text == "Test content"



class TestWikiSection:
    """Test WikiSection model."""

    def test_section_creation(self):
        """Test creating a wiki section."""
        section = WikiSection(
            title="Introduction",
            level=2,
            content="This is the introduction.",
            start_pos=0,
            end_pos=24,
        )
        assert section.title == "Introduction"
        assert section.level == 2
        assert section.start_pos == 0


    def test_section_positions(self):
        """Test section position metadata."""
        section = WikiSection(
            title="Body",
            level=2,
            content="Body content",
            start_pos=10,
            end_pos=22,
        )
        assert section.start_pos == 10
        assert section.end_pos == 22



class TestWikiInfobox:
    """Test WikiInfobox model."""

    def test_infobox_creation(self):
        """Test creating an infobox."""
        infobox = WikiInfobox(
            type="programming language",
            properties={"name": "Python", "paradigm": "multi-paradigm"},
        )
        assert infobox.type == "programming language"
        assert infobox.properties["name"] == "Python"



class TestProcessedChunk:
    """Test ProcessedChunk model."""

    def test_chunk_creation(self):
        """Test creating a processed chunk."""
        chunk = ProcessedChunk(
            chunk_id="chunk_123_0",
            text="This is a chunk of text.",
            contextualized_text="Title: This is a chunk of text.",
            section="Introduction",
            start_char=0,
            end_char=27,
            page_id=1,
            title="Test Article",
            url="https://example.com",
            word_count=6,
        )
        assert chunk.chunk_id == "chunk_123_0"
        assert chunk.word_count == 6



class TestProcessedArticle:
    """Test ProcessedArticle model."""

    def test_processed_article_creation(self):
        """Test creating a processed article."""
        chunk = ProcessedChunk(
            chunk_id="chunk_1_0",
            text="Test chunk",
            contextualized_text="Title: Test chunk",
            section="Test",
            start_char=0,
            end_char=10,
            page_id=1,
            title="Test",
            url="https://example.com",
            word_count=2,
        )
        article = WikiArticle(
            id=1,
            title="Test",
            text="Test chunk",
            url="https://example.com",
        )
        processed = ProcessedArticle(
            article=article,
            chunks=[chunk],
            clean_text="Test chunk",
        )
        assert processed.article.id == 1
        assert len(processed.chunks) == 1



class TestIngestionConfig:
    """Test IngestionConfig model."""

    def test_config_creation(self, tmp_path: Path):
        """Test creating ingestion config."""
        config = IngestionConfig(
            neo4j_uri="bolt://localhost:7687",
            neo4j_user="neo4j",
            neo4j_password="password",
            qdrant_host="localhost",
            qdrant_port=6333,
            qdrant_collection="test",
            download_dir=str(tmp_path / "downloads"),
            checkpoint_file=str(tmp_path / "checkpoint.json"),
        )
        assert config.neo4j_uri == "bolt://localhost:7687"
        assert config.qdrant_collection == "test"



    def test_config_defaults(self, tmp_path: Path):
        """Test config default values."""
        config = IngestionConfig(
            neo4j_uri="bolt://localhost:7687",
            neo4j_user="neo4j",
            neo4j_password="password",
            qdrant_host="localhost",
            qdrant_port=6333,
            qdrant_collection="test",
            download_dir=str(tmp_path / "downloads"),
            checkpoint_file=str(tmp_path / "checkpoint.json"),
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
        assert stats.errors == 0


    def test_stats_update(self):
        """Test updating stats."""
        stats = PipelineStats()
        stats.articles_processed = 10
        stats.chunks_created = 100
        assert stats.articles_processed == 10
        assert stats.chunks_created == 100
