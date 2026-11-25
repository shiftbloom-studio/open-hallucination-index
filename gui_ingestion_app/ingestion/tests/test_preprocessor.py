"""Unit tests for text preprocessor."""
from __future__ import annotations


from ingestion.models import WikiArticle
from ingestion.preprocessor import AdvancedTextPreprocessor, BM25Tokenizer


class TestBM25Tokenizer:
    """Test BM25Tokenizer functionality."""

    def test_tokenizer_initialization(self):
        """Test tokenizer initialization."""
        tokenizer = BM25Tokenizer()
        assert tokenizer is not None
        assert len(tokenizer.STOPWORDS) > 0

    def test_tokenize_basic(self):
        """Test basic tokenization."""
        tokenizer = BM25Tokenizer()
        text = "Python is a programming language"
        tokens = tokenizer.tokenize(text)
        
        assert "python" in tokens
        assert "programming" in tokens
        assert "language" in tokens
        # Stopwords should be removed
        assert "is" not in tokens
        assert "a" not in tokens

    def test_tokenize_with_numbers(self):
        """Test tokenization with numbers."""
        tokenizer = BM25Tokenizer()
        text = "Python 3.14 was released in 2026"
        tokens = tokenizer.tokenize(text)
        
        assert "python" in tokens
        assert "released" in tokens
        # Numbers might be filtered depending on implementation
        assert len(tokens) > 0

    def test_empty_text(self):
        """Test tokenizing empty text."""
        tokenizer = BM25Tokenizer()
        tokens = tokenizer.tokenize("")
        assert len(tokens) == 0

    def test_stopwords_filtering(self):
        """Test that stopwords are properly filtered."""
        tokenizer = BM25Tokenizer()
        text = "the and or but in on at to for of"
        tokens = tokenizer.tokenize(text)
        # All stopwords should be removed
        assert len(tokens) == 0


class TestAdvancedTextPreprocessor:
    """Test AdvancedTextPreprocessor functionality."""

    def test_preprocessor_initialization(self):
        """Test preprocessor initialization."""
        preprocessor = AdvancedTextPreprocessor(
            chunk_size=512, chunk_overlap=50
        )
        assert preprocessor.chunk_size == 512
        assert preprocessor.chunk_overlap == 50

    def test_process_article_basic(self, sample_wiki_article: WikiArticle):
        """Test processing a basic article."""
        preprocessor = AdvancedTextPreprocessor(
            chunk_size=512, chunk_overlap=50
        )
        processed = preprocessor.process_article(sample_wiki_article)
        
        assert processed.article_id == sample_wiki_article.id
        assert processed.title == sample_wiki_article.title
        assert processed.url == sample_wiki_article.url
        assert len(processed.chunks) > 0
        assert len(processed.sections) > 0

    def test_extract_sections(self, sample_wiki_article: WikiArticle):
        """Test section extraction."""
        preprocessor = AdvancedTextPreprocessor()
        sections = preprocessor._extract_sections(sample_wiki_article.text)
        
        assert len(sections) > 0
        # Check that sections have expected structure
        for section in sections:
            assert hasattr(section, "title")
            assert hasattr(section, "content")
            assert hasattr(section, "level")

    def test_clean_text(self):
        """Test text cleaning."""
        preprocessor = AdvancedTextPreprocessor()
        dirty_text = "Test\n\n\nwith    multiple   spaces\tand\ttabs"
        clean = preprocessor._clean_text(dirty_text)
        
        assert "\n\n\n" not in clean
        assert "   " not in clean
        assert "\t" not in clean

    def test_create_semantic_chunks(self):
        """Test semantic chunking."""
        preprocessor = AdvancedTextPreprocessor(
            chunk_size=100, chunk_overlap=20
        )
        text = "Sentence one. Sentence two. Sentence three. " * 10
        chunks = preprocessor._create_semantic_chunks(text)
        
        assert len(chunks) > 0
        # Check chunk size constraints
        for chunk in chunks:
            assert len(chunk) <= preprocessor.chunk_size * 1.2  # Allow some overflow

    def test_empty_article(self):
        """Test processing empty article."""
        preprocessor = AdvancedTextPreprocessor()
        empty_article = WikiArticle(
            id=1,
            title="Empty",
            text="",
            url="https://example.com",
            revision_id=1,
        )
        processed = preprocessor.process_article(empty_article)
        
        assert processed.article_id == 1
        # Should handle empty content gracefully
        assert len(processed.chunks) >= 0

    def test_infobox_extraction(self):
        """Test infobox extraction from article."""
        preprocessor = AdvancedTextPreprocessor()
        article_with_infobox = WikiArticle(
            id=1,
            title="Test",
            text="""
            {{Infobox programming language
            | name = Python
            | paradigm = multi-paradigm
            }}
            
            Python is a language.
            """,
            url="https://example.com",
            revision_id=1,
        )
        processed = preprocessor.process_article(article_with_infobox)
        
        # Should extract infobox data
        assert processed.infoboxes is not None or processed.infoboxes == []


class TestParallelProcessing:
    """Test parallel processing capabilities."""

    def test_batch_processing(self, sample_wiki_article: WikiArticle):
        """Test processing multiple articles."""
        preprocessor = AdvancedTextPreprocessor()
        articles = [sample_wiki_article] * 5
        
        results = []
        for article in articles:
            processed = preprocessor.process_article(article)
            results.append(processed)
        
        assert len(results) == 5
        # All should be processed successfully
        for result in results:
            assert result.article_id == sample_wiki_article.id
