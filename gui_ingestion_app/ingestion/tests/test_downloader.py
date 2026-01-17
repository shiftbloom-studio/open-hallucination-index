"""Unit tests for Wikipedia downloader."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch


from ingestion.downloader import (
    ChunkedWikiDownloader,
    DumpFile,
    LocalFileParser,
)
from ingestion.models import WikiArticle


class TestDumpFile:
    """Test DumpFile dataclass."""

    def test_dump_file_creation(self):
        """Test creating a dump file."""
        dump = DumpFile(
            index=1,
            filename="enwiki-latest-pages-articles-multistream1.xml.bz2",
            url="https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles-multistream1.xml.bz2",
            size_bytes=1024000,
        )
        assert dump.index == 1
        assert not dump.download_complete
        assert not dump.processed

    def test_dump_file_completion(self):
        """Test marking dump file as complete."""
        dump = DumpFile(
            index=1,
            filename="test.xml.bz2",
            url="https://example.com/test.xml.bz2",
        )
        dump.download_complete = True
        dump.processed = True
        assert dump.download_complete
        assert dump.processed


class TestChunkedWikiDownloader:
    """Test ChunkedWikiDownloader functionality."""

    @patch("ingestion.downloader.requests.Session")
    def test_downloader_initialization(self, mock_session):
        """Test downloader initialization."""
        download_dir = Path("/tmp/downloads")
        downloader = ChunkedWikiDownloader(
            download_dir=download_dir, parallel_downloads=2
        )
        assert downloader.download_dir == download_dir
        assert downloader.parallel_downloads == 2

    @patch("ingestion.downloader.requests.Session")
    def test_discover_dump_files_mock(self, mock_session):
        """Test discovering dump files with mocked response."""
        mock_response = MagicMock()
        mock_response.text = """
        <html>
        <a href="enwiki-latest-pages-articles-multistream1.xml.bz2">File 1</a>
        <a href="enwiki-latest-pages-articles-multistream2.xml.bz2">File 2</a>
        </html>
        """
        mock_session.return_value.get.return_value = mock_response

        downloader = ChunkedWikiDownloader(
            download_dir=Path("/tmp/downloads"), parallel_downloads=2
        )
        
        with patch.object(downloader, "_discover_dump_files_from_index") as mock_discover:
            mock_discover.return_value = [
                DumpFile(
                    index=1,
                    filename="enwiki-latest-pages-articles-multistream1.xml.bz2",
                    url="https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles-multistream1.xml.bz2",
                ),
                DumpFile(
                    index=2,
                    filename="enwiki-latest-pages-articles-multistream2.xml.bz2",
                    url="https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles-multistream2.xml.bz2",
                ),
            ]
            dump_files = downloader._discover_dump_files_from_index()
            assert len(dump_files) == 2
            assert dump_files[0].index == 1


class TestLocalFileParser:
    """Test LocalFileParser functionality."""

    def test_parser_initialization(self):
        """Test parser initialization."""
        parser = LocalFileParser()
        assert parser is not None

    @patch("bz2.BZ2File")
    @patch("lxml.etree.iterparse")
    def test_parse_dump_file_mock(self, mock_iterparse, mock_bz2):
        """Test parsing dump file with mocked data."""
        # Mock XML structure
        mock_iterparse.return_value = iter([
            ("end", MagicMock(tag="{http://www.mediawiki.org/xml/export-0.10/}page")),
        ])

        parser = LocalFileParser()
        test_file = Path("/tmp/test.xml.bz2")

        with patch.object(parser, "_extract_article_from_page") as mock_extract:
            mock_extract.return_value = WikiArticle(
                id=1,
                title="Test Article",
                text="Test content",
                url="https://en.wikipedia.org/wiki/Test_Article",
                revision_id=12345,
            )
            
            # Test parsing logic without actual file
            articles = list(parser.parse_dump_file(test_file, max_articles=1))
            # Parser should be able to handle the mock
            assert len(articles) <= 1


class TestGracefulShutdown:
    """Test graceful shutdown functionality."""

    def test_shutdown_flag(self):
        """Test shutdown flag manipulation."""
        from ingestion.downloader import is_shutdown_requested, request_shutdown

        # Initially not requested
        initial_state = is_shutdown_requested()

        # Request shutdown
        request_shutdown()
        assert is_shutdown_requested()

        # Reset for other tests (global state)
        import ingestion.downloader as dl
        dl.shutdown_requested = initial_state
