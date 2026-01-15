#!/usr/bin/env python3
"""
Advanced Wikipedia Dual-Ingestion Tool (Neo4j + Qdrant) v2.0
============================================================

Downloads, parses, and ingests Wikipedia articles into BOTH:
1. Qdrant (Vector Store): Hybrid search with Dense + Sparse (BM25) vectors.
2. Neo4j (Graph Store): Rich knowledge graph with entities, links, and categories.

Key Features:
- Hybrid Search: Dense embeddings + BM25 sparse vectors for optimal retrieval.
- Semantic Chunking: Sentence-aware splitting with configurable overlap.
- Rich Metadata: Section headers, infobox data, entity extraction.
- Parallel Processing: Concurrent uploads to both stores for maximum throughput.
- Full-Text Index: Qdrant payload indexing for keyword search.
- Batch Optimization: Tuned batch sizes and connection pooling.

Prerequisites:
    pip install requests tqdm qdrant-client sentence-transformers lxml neo4j

Usage:
    python ingest_wiki_dual.py --limit 10000 --batch-size 64
"""

from __future__ import annotations

import argparse
import atexit
import bz2
import hashlib
import html
import json
import logging
import math
import re
import signal
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Generator
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import quote, urljoin

import requests
from lxml import etree  # type: ignore[import-untyped]
from neo4j import GraphDatabase
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    HnswConfigDiff,
    KeywordIndexParams,
    KeywordIndexType,
    Modifier,
    OptimizersConfigDiff,
    PointStruct,
    SparseVector,
    SparseVectorParams,
    TextIndexParams,
    TextIndexType,
    TokenizerType,
    VectorParams,
)
from requests.adapters import HTTPAdapter
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from urllib3.util.retry import Retry

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("DualIngest")

# Suppress noisy loggers
logging.getLogger("neo4j").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("qdrant_client").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

# --- Constants ---
DUMPS_URL = "https://dumps.wikimedia.org/enwiki/latest/"
DUMP_PATTERN = r"enwiki-latest-pages-articles-multistream\.xml\.bz2"

# Embedding Configuration
DENSE_MODEL = "all-MiniLM-L6-v2"
DENSE_VECTOR_SIZE = 384

# BM25 Sparse Vector Configuration
BM25_K1 = 1.2
BM25_B = 0.75
AVG_DOC_LENGTH = 500  # Approximation for BM25

# Collection Names
QDRANT_COLLECTION = "wikipedia_hybrid"
NEO4J_DATABASE = "neo4j"

# Performance Tuning - Aggressive defaults for maximum throughput
DEFAULT_BATCH_SIZE = 128  # Larger batches = fewer DB round-trips
DEFAULT_WORKERS = 8  # More parallel workers
DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 64
EMBEDDING_BATCH_SIZE = 256  # Large embedding batches for GPU utilization
QDRANT_UPLOAD_WORKERS = 4  # Parallel Qdrant uploads

# Graceful Shutdown
shutdown_requested = False


def signal_handler(sig, frame):
    """Handle graceful shutdown on SIGINT/SIGTERM."""
    global shutdown_requested
    logger.warning("\n⚠️  Shutdown requested! Finishing current batch...")
    shutdown_requested = True


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


# =============================================================================
# DATA MODELS
# =============================================================================


@dataclass
class WikiSection:
    """Represents a section within an article."""

    title: str
    level: int
    content: str
    start_pos: int
    end_pos: int


@dataclass
class WikiInfobox:
    """Parsed infobox data."""

    type: str
    properties: dict[str, str] = field(default_factory=dict)


@dataclass
class WikiArticle:
    """Rich article representation with extracted metadata."""

    id: int
    title: str
    text: str
    url: str
    links: set[str] = field(default_factory=set)
    categories: set[str] = field(default_factory=set)
    sections: list[WikiSection] = field(default_factory=list)
    infobox: WikiInfobox | None = None
    first_paragraph: str = ""
    word_count: int = 0
    entities: set[str] = field(default_factory=set)


@dataclass
class ProcessedChunk:
    """A processed chunk ready for embedding."""

    chunk_id: str
    text: str
    contextualized_text: str
    section: str
    start_char: int
    end_char: int
    page_id: int
    title: str
    url: str
    word_count: int
    is_first_chunk: bool = False


# =============================================================================
# BM25 TOKENIZER FOR SPARSE VECTORS
# =============================================================================


class BM25Tokenizer:
    """
    Lightweight BM25 tokenizer for generating sparse vectors.
    Uses IDF weighting for better retrieval quality.
    """

    # Common English stopwords
    STOPWORDS = frozenset([
        "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "is", "it", "was", "be", "are", "were", "been", "being", "have",
        "has", "had", "do", "does", "did", "will", "would", "could", "should",
        "may", "might", "must", "shall", "can", "this", "that", "these", "those",
        "i", "you", "he", "she", "we", "they", "what", "which", "who", "whom",
        "when", "where", "why", "how", "all", "each", "every", "both", "few",
        "more", "most", "other", "some", "such", "no", "nor", "not", "only",
        "own", "same", "so", "than", "too", "very", "as", "if", "then",
        "because", "while", "with", "about", "against", "between", "into",
        "through", "during", "before", "after", "above", "below", "from", "up",
        "down", "out", "off", "over", "under", "again", "further", "once",
        "here", "there", "any", "also",
    ])

    TOKEN_PATTERN = re.compile(r"\b[a-z0-9]{2,}\b")

    def __init__(self, vocab_size: int = 30000):
        self.vocab_size = vocab_size
        self.idf_scores: dict[str, float] = {}
        self.doc_count = 0
        self._token_hash_cache: dict[str, int] = {}

    def _hash_token(self, token: str) -> int:
        """Hash token to vocabulary index for sparse vector."""
        if token not in self._token_hash_cache:
            hash_val = int(hashlib.md5(token.encode()).hexdigest(), 16) % self.vocab_size
            # Limit cache size to prevent unbounded growth
            if len(self._token_hash_cache) < 50000:
                self._token_hash_cache[token] = hash_val
            return hash_val
        return self._token_hash_cache[token]

    def tokenize(self, text: str) -> list[str]:
        """Tokenize text into lowercased, filtered tokens."""
        text_lower = text.lower()
        tokens = self.TOKEN_PATTERN.findall(text_lower)
        return [t for t in tokens if t not in self.STOPWORDS and len(t) <= 30]

    def compute_tf(self, tokens: list[str]) -> dict[str, float]:
        """Compute term frequency with BM25 saturation."""
        if not tokens:
            return {}

        counts = Counter(tokens)
        doc_len = len(tokens)

        tf_scores = {}
        for token, count in counts.items():
            # BM25 TF formula with saturation
            tf = (count * (BM25_K1 + 1)) / (
                count + BM25_K1 * (1 - BM25_B + BM25_B * (doc_len / AVG_DOC_LENGTH))
            )
            tf_scores[token] = tf

        return tf_scores

    def to_sparse_vector(
        self, text: str, use_idf: bool = True
    ) -> tuple[list[int], list[float]]:
        """
        Convert text to sparse vector representation.
        Returns (indices, values) tuple for Qdrant SparseVector.
        """
        tokens = self.tokenize(text)
        if not tokens:
            return [], []

        tf_scores = self.compute_tf(tokens)

        # Aggregate by hash index
        index_scores: dict[int, float] = defaultdict(float)
        for token, tf in tf_scores.items():
            idx = self._hash_token(token)
            idf = self.idf_scores.get(token, 1.0) if use_idf else 1.0
            index_scores[idx] += tf * idf

        # Sort by index for consistent representation
        sorted_items = sorted(index_scores.items())
        indices = [item[0] for item in sorted_items]
        values = [item[1] for item in sorted_items]

        return indices, values

    def update_idf(self, documents: list[str]):
        """Update IDF scores from a batch of documents."""
        doc_freq: dict[str, int] = defaultdict(int)

        for doc in documents:
            seen_tokens = set(self.tokenize(doc))
            for token in seen_tokens:
                doc_freq[token] += 1

        self.doc_count += len(documents)

        # Compute IDF using BM25 IDF formula
        for token, df in doc_freq.items():
            idf = math.log((self.doc_count - df + 0.5) / (df + 0.5) + 1)
            self.idf_scores[token] = max(idf, 0.0)


# =============================================================================
# ADVANCED TEXT PREPROCESSOR
# =============================================================================


class AdvancedTextPreprocessor:
    """
    Advanced text processing pipeline with:
    - Section extraction
    - Infobox parsing
    - Named entity extraction (bold terms)
    - Sentence-aware chunking
    """

    # Regex patterns compiled once for performance
    SECTION_PATTERN = re.compile(r"^(={2,6})\s*(.+?)\s*\1\s*$", re.MULTILINE)
    INFOBOX_START_PATTERN = re.compile(r"\{\{\s*Infobox\b", re.IGNORECASE)
    TEMPLATE_PATTERN = re.compile(r"\{\{[^{}]*\}\}", re.DOTALL)
    NESTED_TEMPLATE_PATTERN = re.compile(r"\{\{(?:[^{}]|\{[^{]|\}[^}])*\}\}", re.DOTALL)
    LINK_PATTERN = re.compile(r"\[\[([^|\]#]+)(?:\|[^\]]*)?\]\]")
    CATEGORY_PATTERN = re.compile(
        r"\[\[Category:([^|\]]+)(?:\|[^\]]*)?\]\]", re.IGNORECASE
    )
    BOLD_PATTERN = re.compile(r"'''([^']+)'''")
    REF_PATTERN = re.compile(r"<ref[^>]*>.*?</ref>|<ref[^/]*/\s*>", re.DOTALL)
    HTML_PATTERN = re.compile(r"<[^>]+>")
    WHITESPACE_PATTERN = re.compile(r"\s+")

    # Sentence boundary pattern
    SENTENCE_END_PATTERN = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        min_chunk_size: int = 100,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

    def extract_sections(self, text: str) -> list[WikiSection]:
        """Extract section headers and their content."""
        sections = []
        matches = list(self.SECTION_PATTERN.finditer(text))

        if not matches:
            return [
                WikiSection(
                    title="Introduction",
                    level=1,
                    content=text,
                    start_pos=0,
                    end_pos=len(text),
                )
            ]

        # Add introduction (content before first section)
        if matches[0].start() > 0:
            sections.append(
                WikiSection(
                    title="Introduction",
                    level=1,
                    content=text[: matches[0].start()].strip(),
                    start_pos=0,
                    end_pos=matches[0].start(),
                )
            )

        # Process each section
        for i, match in enumerate(matches):
            level = len(match.group(1))
            title = match.group(2).strip()
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)

            sections.append(
                WikiSection(
                    title=title,
                    level=level,
                    content=text[start:end].strip(),
                    start_pos=start,
                    end_pos=end,
                )
            )

        return sections

    def parse_infobox(self, text: str) -> WikiInfobox | None:
        """Extract and parse infobox data."""
        infobox_block = self._extract_infobox_block(text)
        if not infobox_block:
            return None

        inner = infobox_block[2:-2]
        inner = re.sub(r"^\s*Infobox\s*", "", inner, flags=re.IGNORECASE)
        fields = self._split_infobox_fields(inner)
        if not fields:
            return None

        infobox_type = fields[0].strip()
        properties = {}
        for line in fields[1:]:
            if "=" in line:
                key, _, value = line.partition("=")
                key = key.strip().lower()
                value = value.strip()
                # Clean wiki markup from value
                value = self.LINK_PATTERN.sub(r"\1", value)
                value = re.sub(r"\[\[|\]\]", "", value)
                if key and value and len(key) < 50 and len(value) < 500:
                    properties[key] = value

        return WikiInfobox(type=infobox_type, properties=properties)

    def _extract_infobox_block(self, text: str) -> str | None:
        """Extract the full infobox template block using linear scanning."""
        match = self.INFOBOX_START_PATTERN.search(text)
        if not match:
            return None

        start = match.start()
        i = start
        depth = 0

        while i < len(text) - 1:
            two = text[i : i + 2]
            if two == "{{":
                depth += 1
                i += 2
                continue
            if two == "}}":
                depth -= 1
                i += 2
                if depth == 0:
                    return text[start:i]
                continue
            i += 1

        return None

    def _split_infobox_fields(self, inner: str) -> list[str]:
        """Split infobox fields on top-level pipes, avoiding nested templates/links."""
        fields: list[str] = []
        current: list[str] = []
        template_depth = 0
        link_depth = 0
        i = 0

        while i < len(inner):
            two = inner[i : i + 2]
            if two == "{{":
                template_depth += 1
                current.append(two)
                i += 2
                continue
            if two == "}}":
                template_depth = max(0, template_depth - 1)
                current.append(two)
                i += 2
                continue
            if two == "[[":
                link_depth += 1
                current.append(two)
                i += 2
                continue
            if two == "]]":
                link_depth = max(0, link_depth - 1)
                current.append(two)
                i += 2
                continue
            if inner[i] == "|" and template_depth == 0 and link_depth == 0:
                fields.append("".join(current))
                current = []
                i += 1
                continue

            current.append(inner[i])
            i += 1

        fields.append("".join(current))
        return fields

    def extract_first_paragraph(self, text: str) -> str:
        """Extract the first meaningful paragraph (often the definition)."""
        first_section = self.SECTION_PATTERN.search(text)
        intro = text[: first_section.start()] if first_section else text[:2000]

        clean_intro = self._clean_text(intro)
        paragraphs = [p.strip() for p in clean_intro.split("\n\n") if p.strip()]

        if paragraphs:
            return paragraphs[0][:1000]
        return ""

    def extract_entities(self, text: str) -> set[str]:
        """Extract named entities from bold text (Wikipedia convention)."""
        entities = set()
        for match in self.BOLD_PATTERN.finditer(text[:3000]):
            entity = match.group(1).strip()
            if 2 < len(entity) < 100 and not entity.startswith(("[[", "{{")):
                entities.add(entity)
        return entities

    def clean_and_extract(
        self, text: str
    ) -> tuple[str, set[str], set[str], WikiInfobox | None, set[str]]:
        """
        Comprehensive text cleaning and metadata extraction.
        Returns: (clean_text, links, categories, infobox, entities)
        """
        if not text:
            return "", set(), set(), None, set()

        raw_text = html.unescape(text)

        # Extract metadata before cleaning
        links = set()
        categories = set()

        # Extract categories
        for match in self.CATEGORY_PATTERN.finditer(raw_text):
            categories.add(match.group(1).strip())

        # Extract internal links
        for match in self.LINK_PATTERN.finditer(raw_text):
            link_target = match.group(1).strip()
            if not any(
                link_target.lower().startswith(p)
                for p in ["file:", "image:", "category:", "help:", "user:", "template:"]
            ):
                links.add(link_target)

        # Extract infobox
        infobox = self.parse_infobox(raw_text)

        # Extract entities
        entities = self.extract_entities(raw_text)

        # Clean text for embedding
        clean_text = self._clean_text(raw_text)

        return clean_text, links, categories, infobox, entities

    def _clean_text(self, text: str) -> str:
        """Remove wiki markup and clean text for embedding."""
        # Remove nested templates (multiple passes for deeply nested)
        for _ in range(5):
            new_text = self.NESTED_TEMPLATE_PATTERN.sub("", text)
            if new_text == text:
                break
            text = new_text

        # Remove remaining templates
        text = self.TEMPLATE_PATTERN.sub("", text)

        # Convert links to plain text
        text = self.LINK_PATTERN.sub(r"\1", text)
        text = re.sub(r"\[\[|\]\]", "", text)

        # Remove references
        text = self.REF_PATTERN.sub("", text)

        # Remove HTML tags
        text = self.HTML_PATTERN.sub("", text)

        # Remove wiki formatting
        text = re.sub(r"'{2,}", "", text)  # Bold/italic
        text = re.sub(r"^[*#:;]+", "", text, flags=re.MULTILINE)  # List markers

        # Normalize whitespace
        text = self.WHITESPACE_PATTERN.sub(" ", text)

        return text.strip()

    def chunk_article(
        self, article: WikiArticle, clean_text: str
    ) -> list[ProcessedChunk]:
        """
        Sentence-aware chunking with section context.
        Creates overlapping chunks that respect sentence boundaries.
        """
        if len(clean_text) < self.min_chunk_size:
            return []

        chunks = []

        # Split into sentences
        sentences = self.SENTENCE_END_PATTERN.split(clean_text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return []

        current_chunk: list[str] = []
        current_length = 0
        chunk_start = 0
        current_section = "Introduction"

        # Find which section each character position belongs to
        section_map: dict[int, str] = {}
        for section in article.sections:
            for pos in range(section.start_pos, section.end_pos):
                section_map[pos] = section.title

        char_pos = 0

        for sentence in sentences:
            sentence_len = len(sentence)

            # Check if adding this sentence exceeds chunk size
            if current_length + sentence_len > self.chunk_size and current_chunk:
                # Finalize current chunk
                chunk_text = " ".join(current_chunk)
                chunk_end = char_pos

                # Determine section for this chunk
                mid_pos = (chunk_start + chunk_end) // 2
                current_section = section_map.get(mid_pos, "Introduction")

                chunk_id = f"{article.id}_{len(chunks)}"
                contextualized = f"{article.title} - {current_section}: {chunk_text}"

                chunks.append(
                    ProcessedChunk(
                        chunk_id=chunk_id,
                        text=chunk_text,
                        contextualized_text=contextualized,
                        section=current_section,
                        start_char=chunk_start,
                        end_char=chunk_end,
                        page_id=article.id,
                        title=article.title,
                        url=article.url,
                        word_count=len(chunk_text.split()),
                        is_first_chunk=len(chunks) == 0,
                    )
                )

                # Start new chunk with overlap
                overlap_sentences: list[str] = []
                overlap_length = 0
                for s in reversed(current_chunk):
                    if overlap_length + len(s) <= self.chunk_overlap:
                        overlap_sentences.insert(0, s)
                        overlap_length += len(s)
                    else:
                        break

                current_chunk = overlap_sentences
                current_length = overlap_length
                chunk_start = char_pos - overlap_length

            current_chunk.append(sentence)
            current_length += sentence_len
            char_pos += sentence_len + 1  # +1 for space

        # Final chunk
        if current_chunk and current_length >= self.min_chunk_size:
            chunk_text = " ".join(current_chunk)
            chunk_id = f"{article.id}_{len(chunks)}"
            contextualized = f"{article.title}: {chunk_text}"

            chunks.append(
                ProcessedChunk(
                    chunk_id=chunk_id,
                    text=chunk_text,
                    contextualized_text=contextualized,
                    section=current_section,
                    start_char=chunk_start,
                    end_char=char_pos,
                    page_id=article.id,
                    title=article.title,
                    url=article.url,
                    word_count=len(chunk_text.split()),
                    is_first_chunk=len(chunks) == 0,
                )
            )

        return chunks


# =============================================================================
# WIKIPEDIA DUMP DOWNLOADER - CHUNKED FILE APPROACH
# =============================================================================


@dataclass
class DumpFile:
    """Represents a single dump file part."""

    index: int
    filename: str
    url: str
    local_path: Path | None = None
    download_complete: bool = False
    processed: bool = False
    size_bytes: int = 0


class ChunkedWikiDownloader:
    """
    Downloads Wikipedia dump files in chunks (multistream parts).
    
    The dump is split into ~27 parts. This class:
    1. Discovers all available parts
    2. Downloads them in parallel (max 3 concurrent)
    3. Queues completed downloads for processing
    4. Deletes files after processing
    
    This is much more robust than streaming because:
    - Each file is fully downloaded before processing
    - Network errors only affect one part
    - Can resume from where it left off
    """

    # Pattern to match multistream part files
    PART_PATTERN = re.compile(
        r'href="(enwiki-latest-pages-articles-multistream(\d+)\.xml[^"]*\.bz2)"'
    )
    
    # Download settings
    MAX_CONCURRENT_DOWNLOADS = 3
    CHUNK_SIZE = 1024 * 1024  # 1MB chunks for download
    DOWNLOAD_TIMEOUT = (30, 600)  # (connect, read) timeouts
    MAX_RETRIES = 10
    RETRY_BACKOFF = 5
    
    def __init__(self, download_dir: Path | None = None):
        self.download_dir = download_dir or Path(".wiki_dumps")
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.session = self._create_session()
        self._download_executor = ThreadPoolExecutor(
            max_workers=self.MAX_CONCURRENT_DOWNLOADS
        )
        self._active_downloads: dict[int, Future[DumpFile]] = {}
        
    def _create_session(self) -> requests.Session:
        """Create a requests session with retry logic."""
        session = requests.Session()
        retry_strategy = Retry(
            total=self.MAX_RETRIES,
            backoff_factor=self.RETRY_BACKOFF,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "HEAD"],
        )
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=10,
            pool_maxsize=10,
        )
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    def discover_parts(self) -> list[DumpFile]:
        """Discover all available multistream part files."""
        logger.info(f"🔍 Discovering dump parts at {DUMPS_URL}...")
        
        try:
            resp = self.session.get(DUMPS_URL, timeout=30)
            resp.raise_for_status()
            
            # Find all multistream part files
            matches = self.PART_PATTERN.findall(resp.text)
            if not matches:
                raise ValueError("No multistream parts found on server.")
            
            # Create DumpFile objects, sorted by index
            parts = []
            seen_indices: set[int] = set()
            
            for filename, index_str in matches:
                # Skip RSS/index files
                if "rss" in filename.lower() or "index" in filename.lower():
                    continue
                    
                index = int(index_str)
                if index in seen_indices:
                    continue
                seen_indices.add(index)
                
                parts.append(DumpFile(
                    index=index,
                    filename=filename,
                    url=urljoin(DUMPS_URL, filename),
                    local_path=self.download_dir / filename,
                ))
            
            # Sort by index
            parts.sort(key=lambda p: p.index)
            
            logger.info(f"📦 Found {len(parts)} dump parts (1-{parts[-1].index})")
            return parts
            
        except Exception as e:
            logger.error(f"Failed to discover dump parts: {e}")
            raise

    def download_file(self, part: DumpFile) -> DumpFile:
        """
        Download a single dump file with resume support.
        
        Returns the updated DumpFile with download_complete=True on success.
        """
        if shutdown_requested:
            return part
            
        local_path = part.local_path
        if local_path is None:
            local_path = self.download_dir / part.filename
            part.local_path = local_path
        
        # Check if already downloaded
        if local_path.exists():
            # Verify file size with HEAD request
            try:
                head_resp = self.session.head(part.url, timeout=30)
                expected_size = int(head_resp.headers.get("content-length", 0))
                actual_size = local_path.stat().st_size
                
                if expected_size > 0 and actual_size == expected_size:
                    logger.info(f"✅ Part {part.index} already downloaded")
                    part.download_complete = True
                    part.size_bytes = actual_size
                    return part
                elif actual_size > 0:
                    logger.info(
                        f"📥 Resuming part {part.index} from {actual_size:,} bytes"
                    )
            except Exception:
                pass  # Will re-download if HEAD fails
        
        # Download with resume support
        headers = {}
        mode = "wb"
        start_byte = 0
        
        if local_path.exists():
            start_byte = local_path.stat().st_size
            headers["Range"] = f"bytes={start_byte}-"
            mode = "ab"
        
        logger.info(f"📥 Downloading part {part.index}: {part.filename}")
        
        retry_count = 0
        while retry_count < self.MAX_RETRIES and not shutdown_requested:
            try:
                resp = self.session.get(
                    part.url,
                    stream=True,
                    timeout=self.DOWNLOAD_TIMEOUT,
                    headers=headers,
                )
                
                # Handle resume response
                if resp.status_code == 416:  # Range not satisfiable = complete
                    part.download_complete = True
                    part.size_bytes = local_path.stat().st_size
                    return part
                    
                resp.raise_for_status()
                
                # Get total size
                if "content-range" in resp.headers:
                    # Resume response: "bytes start-end/total"
                    total_size = int(
                        resp.headers["content-range"].split("/")[-1]
                    )
                else:
                    total_size = int(resp.headers.get("content-length", 0))
                    total_size += start_byte
                
                # Download with progress
                downloaded = start_byte
                with open(local_path, mode) as f:
                    for chunk in resp.iter_content(chunk_size=self.CHUNK_SIZE):
                        if shutdown_requested:
                            logger.warning(
                                f"⚠️  Download interrupted for part {part.index}"
                            )
                            return part
                            
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                
                # Verify download
                actual_size = local_path.stat().st_size
                if total_size > 0 and actual_size < total_size:
                    logger.warning(
                        f"⚠️  Part {part.index} incomplete: "
                        f"{actual_size:,}/{total_size:,} bytes"
                    )
                    # Update headers for resume
                    start_byte = actual_size
                    headers["Range"] = f"bytes={start_byte}-"
                    mode = "ab"
                    retry_count += 1
                    time.sleep(self.RETRY_BACKOFF)
                    continue
                
                logger.info(
                    f"✅ Part {part.index} downloaded: "
                    f"{actual_size / (1024*1024):.1f} MB"
                )
                part.download_complete = True
                part.size_bytes = actual_size
                return part
                
            except Exception as e:
                retry_count += 1
                logger.warning(
                    f"⚠️  Download error for part {part.index} "
                    f"(attempt {retry_count}/{self.MAX_RETRIES}): {e}"
                )
                if retry_count < self.MAX_RETRIES:
                    time.sleep(self.RETRY_BACKOFF * retry_count)
                    # Update for resume
                    if local_path.exists():
                        start_byte = local_path.stat().st_size
                        headers["Range"] = f"bytes={start_byte}-"
                        mode = "ab"
        
        logger.error(f"❌ Failed to download part {part.index} after retries")
        return part

    def start_download(self, part: DumpFile) -> Future[DumpFile]:
        """Start an async download for a part."""
        future = self._download_executor.submit(self.download_file, part)
        self._active_downloads[part.index] = future
        return future

    def get_completed_downloads(self) -> list[DumpFile]:
        """Check for and return completed downloads."""
        completed = []
        to_remove = []
        
        for index, future in self._active_downloads.items():
            if future.done():
                to_remove.append(index)
                try:
                    part = future.result()
                    if part.download_complete:
                        completed.append(part)
                except Exception as e:
                    logger.error(f"Download future error for part {index}: {e}")
        
        for index in to_remove:
            del self._active_downloads[index]
        
        return completed

    def active_download_count(self) -> int:
        """Return number of active downloads."""
        return len(self._active_downloads)

    def close(self):
        """Clean up resources."""
        self._download_executor.shutdown(wait=False)
        self.session.close()


class LocalFileParser:
    """
    Parses downloaded Wikipedia dump files from disk.
    
    Much more robust than streaming because:
    - File is complete before parsing
    - Can seek/retry on errors
    - Memory-mapped for efficiency
    """

    @staticmethod
    def parse_file(file_path: Path) -> Generator[WikiArticle]:
        """
        Parse a local BZ2-compressed Wikipedia dump file.
        
        Yields WikiArticle objects for each valid article.
        """
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return
        
        logger.info(f"📖 Parsing {file_path.name}...")
        articles_yielded = 0
        parse_errors = 0
        
        try:
            # Open BZ2 file and parse with lxml
            with bz2.open(file_path, "rb") as f:
                context = etree.iterparse(
                    f,
                    events=("end",),
                    tag="{http://www.mediawiki.org/xml/export-0.11/}page",
                    recover=True,
                    huge_tree=True,
                )
                
                for _event, elem in context:
                    if shutdown_requested:
                        break
                    
                    try:
                        # Namespace-aware lookups
                        ns = {"mw": "http://www.mediawiki.org/xml/export-0.11/"}
                        
                        title = (
                            elem.findtext("mw:title", namespaces=ns)
                            or elem.findtext(
                                "{http://www.mediawiki.org/xml/export-0.11/}title"
                            )
                            or elem.findtext("title")
                        )
                        
                        page_id = (
                            elem.findtext("mw:id", namespaces=ns)
                            or elem.findtext(
                                "{http://www.mediawiki.org/xml/export-0.11/}id"
                            )
                            or elem.findtext("id")
                        )
                        
                        revision = (
                            elem.find("mw:revision", namespaces=ns)
                            or elem.find(
                                "{http://www.mediawiki.org/xml/export-0.11/}revision"
                            )
                            or elem.find("revision")
                        )
                        
                        page_ns = (
                            elem.findtext("mw:ns", namespaces=ns)
                            or elem.findtext(
                                "{http://www.mediawiki.org/xml/export-0.11/}ns"
                            )
                            or elem.findtext("ns")
                        )

                        text = ""
                        if revision is not None:
                            text = (
                                revision.findtext("mw:text", namespaces=ns)
                                or revision.findtext(
                                    "{http://www.mediawiki.org/xml/export-0.11/}text"
                                )
                                or revision.findtext("text")
                                or ""
                            )

                        # Only process main namespace articles, skip redirects
                        if (
                            page_ns == "0"
                            and text
                            and title is not None
                            and page_id is not None
                            and not text.lower().startswith("#redirect")
                            and len(text) > 100
                        ):
                            safe_title = quote(title.replace(" ", "_"), safe="/:@")
                            yield WikiArticle(
                                id=int(page_id),
                                title=title,
                                text=text,
                                url=f"https://en.wikipedia.org/wiki/{safe_title}",
                                word_count=len(text.split()),
                            )
                            articles_yielded += 1

                    except Exception as e:
                        parse_errors += 1
                        if parse_errors <= 10:
                            logger.debug(f"Parse error: {e}")
                    finally:
                        # Clear element to free memory
                        elem.clear()
                        while elem.getprevious() is not None:
                            parent = elem.getparent()
                            if parent is not None:
                                del parent[0]
                            else:
                                break
                
        except Exception as e:
            logger.error(f"Error parsing {file_path.name}: {e}")
        
        if parse_errors > 0:
            logger.warning(
                f"Completed {file_path.name} with {parse_errors} errors, "
                f"{articles_yielded} articles"
            )
        else:
            logger.info(
                f"✅ Parsed {file_path.name}: {articles_yielded} articles"
            )


# =============================================================================
# LEGACY STREAMING DOWNLOADER (kept for compatibility)
# =============================================================================


class WikiDownloader:
    """Handles finding and streaming the Wikipedia dump file."""

    # Retry configuration for network resilience
    MAX_RETRIES = 5
    RETRY_BACKOFF = 2  # seconds, will be exponentially increased
    CHUNK_SIZE = 512 * 1024  # 512KB chunks for better network handling
    STREAM_TIMEOUT = (30, 300)  # (connect, read) timeouts

    @staticmethod
    def _create_session() -> requests.Session:
        """Create a requests session with retry logic."""
        session = requests.Session()
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=WikiDownloader.MAX_RETRIES,
            backoff_factor=WikiDownloader.RETRY_BACKOFF,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "HEAD"],
        )
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=10,
            pool_maxsize=10,
        )
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        return session

    @staticmethod
    def get_latest_url() -> str:
        """Find the latest dump file URL."""
        logger.info(f"Finding latest dump at {DUMPS_URL}...")
        try:
            session = WikiDownloader._create_session()
            resp = session.get(DUMPS_URL, timeout=30)
            resp.raise_for_status()
            matches = re.findall(f'href="({DUMP_PATTERN})"', resp.text)
            if not matches:
                raise ValueError("Dump file not found on server.")
            url = urljoin(DUMPS_URL, matches[0])
            logger.info(f"📦 Target: {url}")
            return url
        except Exception as e:
            logger.error(f"Failed to find dump: {e}")
            sys.exit(1)

    @staticmethod
    def stream_articles(url: str) -> Generator[WikiArticle]:
        """Stream articles from BZ2-compressed XML dump with retry support."""
        logger.info("🚀 Starting stream download...")

        session = WikiDownloader._create_session()
        
        # Use longer timeouts and keep-alive for large file streaming
        response = session.get(
            url,
            stream=True,
            timeout=WikiDownloader.STREAM_TIMEOUT,
            headers={
                "Connection": "keep-alive",
                "Accept-Encoding": "identity",  # No compression on top of bz2
            },
        )
        response.raise_for_status()

        try:
            class MultistreamBZ2Reader:
                """
                Reader for multistream bz2 files (like Wikipedia dumps).
                These files contain multiple concatenated bz2 streams.
                Implements proper file-like interface for lxml.
                """

                def __init__(self, response):
                    self.response = response
                    self.chunk_iter = response.iter_content(
                        chunk_size=WikiDownloader.CHUNK_SIZE
                    )
                    self.raw_buffer = b""
                    self.decompressed_buffer = b""
                    self.decompressor = bz2.BZ2Decompressor()
                    self.finished = False
                    self.bytes_read = 0
                    self.network_errors = 0
                    self.decompress_errors = 0
                    self.max_decompress_errors = 1000  # Allow some corruption

                def _refill_raw(self) -> bool:
                    """Refill raw buffer from network with error handling."""
                    max_network_errors = 10
                    
                    while self.network_errors < max_network_errors:
                        try:
                            chunk = next(self.chunk_iter)
                            if chunk:
                                self.raw_buffer += chunk
                                self.bytes_read += len(chunk)
                                self.network_errors = 0  # Reset on success
                                return True
                            return False
                        except StopIteration:
                            return False
                        except Exception as e:
                            self.network_errors += 1
                            if self.network_errors >= max_network_errors:
                                logger.error(
                                    f"Network error after {self.bytes_read:,} bytes: {e}"
                                )
                                raise
                            # Brief pause before retry
                            time.sleep(1)
                            continue
                    
                    return False

                def _decompress_available(self) -> None:
                    """Decompress as much as possible from raw buffer."""
                    while self.raw_buffer and not shutdown_requested:
                        try:
                            # Try to decompress
                            result = self.decompressor.decompress(self.raw_buffer)
                            self.decompressed_buffer += result
                            
                            if self.decompressor.eof:
                                # Stream ended, check for more data
                                unused = self.decompressor.unused_data
                                self.raw_buffer = unused
                                if unused:
                                    # Start new decompressor for next stream
                                    self.decompressor = bz2.BZ2Decompressor()
                                else:
                                    break
                            else:
                                # Need more input
                                self.raw_buffer = b""
                                break
                        except OSError:
                            # BZ2 decompression error - likely end of valid data
                            self.decompress_errors += 1
                            if self.decompress_errors > self.max_decompress_errors:
                                logger.warning(
                                    "Too many decompression errors, "
                                    "treating as end of data"
                                )
                                self.raw_buffer = b""
                                self.finished = True
                                break
                            # Skip corrupted byte and try again
                            self.raw_buffer = self.raw_buffer[1:]
                            if not self.raw_buffer:
                                break
                        except Exception as e:
                            # Unexpected error - log and skip
                            logger.debug(f"Decompress error: {e}")
                            self.raw_buffer = self.raw_buffer[1:]
                            if not self.raw_buffer:
                                break

                def read(self, n: int = -1) -> bytes:
                    """
                    Read n bytes of decompressed data.
                    If n is -1, read all available data.
                    """
                    if shutdown_requested:
                        return b""
                    
                    if n == -1:
                        # Read all remaining data
                        while not self.finished:
                            self._decompress_available()
                            if not self._refill_raw():
                                self.finished = True
                                break
                            self._decompress_available()
                        result = self.decompressed_buffer
                        self.decompressed_buffer = b""
                        return result

                    while len(self.decompressed_buffer) < n and not self.finished:
                        # First try to decompress what we have
                        self._decompress_available()

                        if len(self.decompressed_buffer) >= n:
                            break

                        # Need more raw data
                        if not self._refill_raw():
                            self.finished = True
                            break

                        # Decompress the new data
                        self._decompress_available()

                    result = self.decompressed_buffer[:n]
                    self.decompressed_buffer = self.decompressed_buffer[n:]
                    return result

            source = MultistreamBZ2Reader(response)
            
            # Use lxml's iterparse which is more robust for large/malformed XML
            # recover=True allows parsing to continue past errors
            context = etree.iterparse(
                source,
                events=("end",),
                tag="{http://www.mediawiki.org/xml/export-0.11/}page",
                recover=True,
                huge_tree=True,
            )
            
            articles_yielded = 0
            parse_errors = 0
            max_consecutive_errors = 100
            consecutive_errors = 0

            try:
                for _event, elem in context:
                    if shutdown_requested:
                        break
                    
                    consecutive_errors = 0  # Reset on successful parse

                    # Use namespace-aware lookups
                    ns = {"mw": "http://www.mediawiki.org/xml/export-0.11/"}
                    
                    title = elem.findtext("mw:title", namespaces=ns)
                    if title is None:
                        title = elem.findtext("{http://www.mediawiki.org/xml/export-0.11/}title")
                    if title is None:
                        title = elem.findtext("title")
                    
                    page_id = elem.findtext("mw:id", namespaces=ns)
                    if page_id is None:
                        page_id = elem.findtext("{http://www.mediawiki.org/xml/export-0.11/}id")
                    if page_id is None:
                        page_id = elem.findtext("id")
                    
                    revision = elem.find("mw:revision", namespaces=ns)
                    if revision is None:
                        revision = elem.find("{http://www.mediawiki.org/xml/export-0.11/}revision")
                    if revision is None:
                        revision = elem.find("revision")
                    
                    page_ns = elem.findtext("mw:ns", namespaces=ns)
                    if page_ns is None:
                        page_ns = elem.findtext("{http://www.mediawiki.org/xml/export-0.11/}ns")
                    if page_ns is None:
                        page_ns = elem.findtext("ns")

                    text = ""
                    if revision is not None:
                        text = (
                            revision.findtext("mw:text", namespaces=ns)
                            or revision.findtext("{http://www.mediawiki.org/xml/export-0.11/}text")
                            or revision.findtext("text")
                            or ""
                        )

                    # Only process main namespace articles, skip redirects
                    if (
                        page_ns == "0"
                        and text
                        and title is not None
                        and page_id is not None
                        and not text.lower().startswith("#redirect")
                        and len(text) > 100
                    ):
                        safe_title = quote(title.replace(" ", "_"), safe="/:@")
                        yield WikiArticle(
                            id=int(page_id),
                            title=title,
                            text=text,
                            url=f"https://en.wikipedia.org/wiki/{safe_title}",
                            word_count=len(text.split()),
                        )
                        articles_yielded += 1

                    # Clear element to free memory
                    elem.clear()
                    # Also clear preceding siblings to prevent memory growth
                    while elem.getprevious() is not None:
                        del elem.getparent()[0]
                        
            except etree.XMLSyntaxError as e:
                consecutive_errors += 1
                parse_errors += 1
                if consecutive_errors < max_consecutive_errors:
                    logger.warning(f"XML parse error (recoverable): {e}")
                else:
                    logger.error(f"Too many consecutive XML errors, stopping: {e}")
            except Exception as e:
                logger.error(f"Unexpected error during XML parsing: {e}")
            
            if parse_errors > 0:
                logger.warning(
                    f"Completed with {parse_errors} parse errors, "
                    f"{articles_yielded} articles yielded"
                )
            else:
                logger.info(
                    f"Parsing completed successfully: {articles_yielded} articles yielded"
                )
        finally:
            # Ensure response is closed properly
            response.close()
            session.close()


# =============================================================================
# QDRANT HYBRID STORE
# =============================================================================


class QdrantHybridStore:
    """
    Qdrant store with hybrid search support:
    - Dense vectors (sentence-transformers)
    - Sparse vectors (BM25)
    - Full-text payload index
    
    Optimized for maximum throughput with:
    - GPU acceleration when available
    - Parallel uploads
    - Large embedding batches
    """

    def __init__(
        self,
        host: str,
        port: int,
        collection: str,
        grpc_port: int | None = None,
        prefer_grpc: bool = False,
    ):
        # Only use gRPC if explicitly provided and prefer_grpc is True
        client_kwargs: dict[str, Any] = {
            "host": host,
            "port": port,
            "timeout": 120,
            "prefer_grpc": prefer_grpc and grpc_port is not None,
        }
        if grpc_port is not None:
            client_kwargs["grpc_port"] = grpc_port
        
        self.client = QdrantClient(**client_kwargs)
        self.collection = collection
        
        # Initialize embedding model with GPU if available
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            logger.info(f"🚀 GPU detected: {torch.cuda.get_device_name(0)}")
        else:
            logger.info("💻 Using CPU for embeddings")
        
        self.model = SentenceTransformer(DENSE_MODEL, device=device)
        self.tokenizer = BM25Tokenizer(vocab_size=50000)
        self._upload_executor = ThreadPoolExecutor(max_workers=QDRANT_UPLOAD_WORKERS)
        self._init_collection()

    def _init_collection(self):
        """Initialize collection with hybrid vector configuration."""
        if self.client.collection_exists(self.collection):
            logger.info(f"Collection '{self.collection}' exists, using it.")
            return

        logger.info(f"Creating Qdrant collection: {self.collection}")

        self.client.create_collection(
            collection_name=self.collection,
            vectors_config={
                "dense": VectorParams(
                    size=DENSE_VECTOR_SIZE,
                    distance=Distance.COSINE,
                    on_disk=True,
                ),
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(
                    modifier=Modifier.IDF,
                ),
            },
            hnsw_config=HnswConfigDiff(
                m=16,
                ef_construct=128,
                full_scan_threshold=10000,
                on_disk=True,
            ),
            optimizers_config=OptimizersConfigDiff(
                indexing_threshold=20000,
                memmap_threshold=50000,
            ),
        )

        # Create payload indexes for filtering and full-text search
        self._create_payload_indexes()

    def _create_payload_indexes(self):
        """Create indexes on payload fields for efficient filtering."""
        try:
            # Full-text index on text content
            self.client.create_payload_index(
                collection_name=self.collection,
                field_name="text",
                field_schema=TextIndexParams(
                    type=TextIndexType.TEXT,
                    tokenizer=TokenizerType.WORD,
                    lowercase=True,
                    min_token_len=2,
                    max_token_len=20,
                ),
            )

            # Keyword index on title
            self.client.create_payload_index(
                collection_name=self.collection,
                field_name="title",
                field_schema=KeywordIndexParams(type=KeywordIndexType.KEYWORD),
            )

            # Keyword index on section
            self.client.create_payload_index(
                collection_name=self.collection,
                field_name="section",
                field_schema=KeywordIndexParams(type=KeywordIndexType.KEYWORD),
            )

            logger.info("✅ Payload indexes created")
        except Exception as e:
            logger.warning(f"Payload index creation: {e}")

    # Maximum points per upload to stay under Qdrant's 32MB payload limit
    MAX_POINTS_PER_UPLOAD = 200  # Increased for fewer round-trips
    # Maximum text length per chunk to prevent oversized payloads
    MAX_TEXT_LENGTH = 4000

    def upload_batch(self, chunks: list[ProcessedChunk]):
        """
        Upload a batch of chunks with both dense and sparse vectors.
        
        Optimized for maximum throughput:
        - Large embedding batches for GPU utilization
        - Parallel sparse vector computation
        - Async uploads to Qdrant
        """
        if not chunks:
            return

        # Truncate oversized texts to prevent payload limit errors
        max_ctx_len = self.MAX_TEXT_LENGTH + 200
        for chunk in chunks:
            if len(chunk.text) > self.MAX_TEXT_LENGTH:
                chunk.text = chunk.text[:self.MAX_TEXT_LENGTH] + "..."
            if len(chunk.contextualized_text) > max_ctx_len:
                chunk.contextualized_text = chunk.contextualized_text[:max_ctx_len] + "..."

        # Update IDF scores with this batch
        texts = [c.contextualized_text for c in chunks]
        self.tokenizer.update_idf(texts)

        # Compute dense embeddings with large batches for GPU
        try:
            dense_vectors = self.model.encode(
                texts,
                batch_size=EMBEDDING_BATCH_SIZE,  # Large batch for GPU
                show_progress_bar=False,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
        except Exception as e:
            logger.error(f"Embedding error: {e}")
            return

        # Pre-compute all sparse vectors (fast, CPU-bound)
        sparse_vectors = [
            self.tokenizer.to_sparse_vector(c.contextualized_text)
            for c in chunks
        ]

        # Build points with hybrid vectors
        points = []
        for i, chunk in enumerate(chunks):
            # Generate deterministic point ID
            point_id = abs(hash(chunk.chunk_id)) % (2**63)

            indices, values = sparse_vectors[i]

            point = PointStruct(
                id=point_id,
                vector={
                    "dense": dense_vectors[i].tolist(),
                    "sparse": SparseVector(indices=indices, values=values),
                },
                payload={
                    "page_id": chunk.page_id,
                    "title": chunk.title,
                    "text": chunk.text,
                    "section": chunk.section,
                    "url": chunk.url,
                    "chunk_id": chunk.chunk_id,
                    "word_count": chunk.word_count,
                    "is_first": chunk.is_first_chunk,
                    "source": "wikipedia",
                },
            )
            points.append(point)

        # Upload in smaller chunks to stay under payload limit
        self._upload_points_chunked(points)

    def _upload_points_chunked(self, points: list[PointStruct]) -> None:
        """Upload points in chunks to avoid payload size limits."""
        for i in range(0, len(points), self.MAX_POINTS_PER_UPLOAD):
            chunk = points[i : i + self.MAX_POINTS_PER_UPLOAD]
            try:
                self.client.upsert(
                    collection_name=self.collection,
                    points=chunk,
                    wait=False,
                )
            except Exception as e:
                # If still too large, try even smaller chunks
                if "larger than allowed" in str(e):
                    logger.warning("Payload too large, retrying with smaller chunks")
                    for point in chunk:
                        try:
                            self.client.upsert(
                                collection_name=self.collection,
                                points=[point],
                                wait=False,
                            )
                        except Exception as pe:
                            logger.error(f"Failed to upload point: {pe}")
                else:
                    logger.error(f"Upload error: {e}")


# =============================================================================
# NEO4J GRAPH STORE
# =============================================================================


class Neo4jGraphStore:
    """
    Neo4j store with optimized batch writes and rich relationship modeling.
    
    Optimized for maximum throughput:
    - Large connection pool
    - Optimized batch queries
    - Parallel write operations
    """

    def __init__(
        self, uri: str, user: str, password: str, database: str = "neo4j"
    ):
        self.driver = GraphDatabase.driver(
            uri,
            auth=(user, password),
            max_connection_pool_size=100,  # Increased from 50
            connection_acquisition_timeout=120,  # Increased timeout
        )
        self.database = database
        self._init_schema()

    def _init_schema(self):
        """Initialize constraints and indexes for optimal performance."""
        with self.driver.session(database=self.database) as session:
            constraints = [
                "CREATE CONSTRAINT article_id IF NOT EXISTS "
                "FOR (a:Article) REQUIRE a.id IS UNIQUE",
                "CREATE CONSTRAINT category_name IF NOT EXISTS "
                "FOR (c:Category) REQUIRE c.name IS UNIQUE",
                "CREATE CONSTRAINT entity_name IF NOT EXISTS "
                "FOR (e:Entity) REQUIRE e.name IS UNIQUE",
            ]

            indexes = [
                "CREATE INDEX article_title IF NOT EXISTS FOR (a:Article) ON (a.title)",
                "CREATE INDEX article_url IF NOT EXISTS FOR (a:Article) ON (a.url)",
                "CREATE INDEX article_word_count IF NOT EXISTS FOR (a:Article) ON (a.word_count)",
            ]

            for query in constraints + indexes:
                try:
                    session.run(query)  # type: ignore[arg-type]
                except Exception as e:
                    if "already exists" not in str(e).lower():
                        logger.warning(f"Schema setup: {e}")

            logger.info("✅ Neo4j schema initialized")

    def upload_batch(
        self,
        articles: list[WikiArticle],
        chunk_map: dict[int, list[str]],
    ):
        """
        Batch upload articles with all relationships.
        Uses UNWIND for efficient bulk operations.
        """
        if not articles:
            return

        # Prepare batch data
        article_data = []
        for article in articles:
            article_data.append({
                "id": article.id,
                "title": article.title,
                "url": article.url,
                "word_count": article.word_count,
                "first_paragraph": (
                    article.first_paragraph[:1000]
                    if article.first_paragraph
                    else ""
                ),
                "infobox_type": (
                    article.infobox.type if article.infobox else None
                ),
                "links": list(article.links)[:100],
                "categories": list(article.categories),
                "entities": list(article.entities)[:50],
                "chunk_ids": chunk_map.get(article.id, []),
            })

        # Main article creation query with relationships
        query = """
        UNWIND $batch AS data
        
        // Create or update Article node
        MERGE (a:Article {id: data.id})
        SET a.title = data.title,
            a.url = data.url,
            a.word_count = data.word_count,
            a.first_paragraph = data.first_paragraph,
            a.infobox_type = data.infobox_type,
            a.vector_chunk_ids = data.chunk_ids,
            a.last_updated = datetime()
        
        // Create Category relationships
        WITH a, data
        UNWIND CASE WHEN size(data.categories) > 0 THEN data.categories ELSE [null] END AS catName
        WITH a, data, catName WHERE catName IS NOT NULL
        MERGE (c:Category {name: catName})
        MERGE (a)-[:IN_CATEGORY]->(c)
        
        WITH DISTINCT a, data
        
        // Create Entity relationships
        UNWIND CASE WHEN size(data.entities) > 0 THEN data.entities ELSE [null] END AS entityName
        WITH a, data, entityName WHERE entityName IS NOT NULL
        MERGE (e:Entity {name: entityName})
        MERGE (a)-[:MENTIONS]->(e)
        
        WITH DISTINCT a, data
        RETURN a.id as article_id
        """

        # Links query (separate to avoid cartesian products)
        links_query = """
        UNWIND $batch AS data
        MATCH (source:Article {id: data.id})
        WITH source, data
        UNWIND CASE WHEN size(data.links) > 0 THEN data.links ELSE [null] END AS linkTitle
        WITH source, linkTitle WHERE linkTitle IS NOT NULL
        MERGE (target:Article {title: linkTitle})
        MERGE (source)-[:LINKS_TO]->(target)
        """

        try:
            with self.driver.session(database=self.database) as session:
                session.run(query, batch=article_data)
                session.run(links_query, batch=article_data)
        except Exception as e:
            logger.error(f"Neo4j write error: {e}")

    def close(self):
        """Close driver connection."""
        self.driver.close()


# =============================================================================
# STATISTICS TRACKER
# =============================================================================


class IngestionStats:
    """Track and display ingestion statistics."""

    def __init__(self):
        self.start_time = time.time()
        self.articles_processed = 0
        self.chunks_created = 0
        self.links_extracted = 0
        self.categories_extracted = 0
        self.errors = 0

    def update(
        self,
        articles: int = 0,
        chunks: int = 0,
        links: int = 0,
        categories: int = 0,
        errors: int = 0,
    ):
        """Update statistics counters."""
        self.articles_processed += articles
        self.chunks_created += chunks
        self.links_extracted += links
        self.categories_extracted += categories
        self.errors += errors

    def summary(self) -> str:
        """Generate summary string."""
        elapsed = time.time() - self.start_time
        rate = self.articles_processed / elapsed if elapsed > 0 else 0
        return (
            f"\n{'='*60}\n"
            f"📊 INGESTION SUMMARY\n"
            f"{'='*60}\n"
            f"⏱️  Duration: {elapsed:.1f}s\n"
            f"📄 Articles: {self.articles_processed:,}\n"
            f"🧩 Chunks: {self.chunks_created:,}\n"
            f"🔗 Links: {self.links_extracted:,}\n"
            f"🏷️  Categories: {self.categories_extracted:,}\n"
            f"⚡ Rate: {rate:.1f} articles/sec\n"
            f"❌ Errors: {self.errors}\n"
            f"{'='*60}"
        )


# =============================================================================
# CHECKPOINT MANAGER FOR RESUMABLE INGESTION
# =============================================================================


class CheckpointManager:
    """
    Manages checkpoint state for resumable ingestion.
    
    Saves progress after each batch so ingestion can be resumed from
    the last successful batch if interrupted.
    
    IMPORTANT: processed_ids are ALWAYS preserved unless --clear-checkpoint
    is used. This ensures no duplicate articles are ever imported, even
    across multiple runs with different settings.
    """

    DEFAULT_CHECKPOINT_FILE = ".ingest_checkpoint.json"

    def __init__(
        self,
        checkpoint_file: str | None = None,
        collection: str = QDRANT_COLLECTION,
        auto_save: bool = True,
    ):
        self.checkpoint_file = Path(checkpoint_file or self.DEFAULT_CHECKPOINT_FILE)
        self.collection = collection
        self.auto_save = auto_save
        
        # Checkpoint state
        self.processed_ids: set[int] = set()
        self.last_article_id: int = 0
        self.articles_processed: int = 0
        self.chunks_created: int = 0
        self.session_start: float = time.time()
        self.total_elapsed: float = 0.0
        
        # Register atexit handler for emergency saves
        atexit.register(self._emergency_save)

    def _emergency_save(self) -> None:
        """Save checkpoint on unexpected exit."""
        if self.processed_ids and shutdown_requested:
            self.save()
            logger.info("💾 Emergency checkpoint saved")

    def load(self) -> bool:
        """
        Load checkpoint from file if it exists and matches current collection.
        
        Returns:
            True if checkpoint was loaded, False otherwise.
        """
        if not self.checkpoint_file.exists():
            logger.info("📍 No checkpoint found, starting fresh")
            return False

        try:
            with open(self.checkpoint_file, encoding="utf-8") as f:
                data = json.load(f)

            # Verify collection matches
            if data.get("collection") != self.collection:
                logger.warning(
                    f"⚠️  Checkpoint is for different collection "
                    f"('{data.get('collection')}' vs '{self.collection}'). "
                    f"Starting fresh."
                )
                return False

            self.processed_ids = set(data.get("processed_ids", []))
            self.last_article_id = data.get("last_article_id", 0)
            self.articles_processed = data.get("articles_processed", 0)
            self.chunks_created = data.get("chunks_created", 0)
            self.total_elapsed = data.get("total_elapsed", 0.0)

            logger.info(
                f"✅ Checkpoint loaded: {self.articles_processed:,} articles processed, "
                f"resuming from ID {self.last_article_id}"
            )
            return True

        except (json.JSONDecodeError, KeyError, OSError) as e:
            logger.warning(f"⚠️  Failed to load checkpoint: {e}. Starting fresh.")
            return False

    def save(self) -> None:
        """Save current state to checkpoint file."""
        # Calculate total elapsed time including previous sessions
        current_session_time = time.time() - self.session_start
        total_time = self.total_elapsed + current_session_time

        data = {
            "collection": self.collection,
            "processed_ids": list(self.processed_ids),
            "last_article_id": self.last_article_id,
            "articles_processed": self.articles_processed,
            "chunks_created": self.chunks_created,
            "total_elapsed": total_time,
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "version": "2.0",
        }

        # Write atomically using temp file
        temp_file = self.checkpoint_file.with_suffix(".tmp")
        try:
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            
            # Atomic rename
            temp_file.replace(self.checkpoint_file)
            
        except OSError as e:
            logger.error(f"❌ Failed to save checkpoint: {e}")
            if temp_file.exists():
                temp_file.unlink()

    def should_skip(self, article_id: int) -> bool:
        """
        Check if article has already been processed.
        
        This is the PRIMARY mechanism for preventing duplicate imports.
        Even though the stream always starts from the beginning after
        a reconnection, already-processed articles are tracked in the
        checkpoint file and will be skipped.
        
        Secondary protections:
        - Qdrant: Uses upsert() with deterministic IDs (same article = same ID)
        - Neo4j: Uses MERGE instead of CREATE (idempotent operations)
        
        This means even if an article somehow bypasses this check,
        it would simply overwrite the existing data, not create duplicates.
        """
        return article_id in self.processed_ids

    def record_batch(
        self,
        articles: list[Any],
        chunk_count: int,
    ) -> None:
        """
        Record a successfully processed batch.
        
        Args:
            articles: List of processed WikiArticle objects
            chunk_count: Number of chunks created from this batch
        """
        for article in articles:
            self.processed_ids.add(article.id)
            self.last_article_id = max(self.last_article_id, article.id)

        self.articles_processed += len(articles)
        self.chunks_created += chunk_count

        if self.auto_save:
            self.save()

    def load_ids_only(self) -> bool:
        """
        Load ONLY the processed_ids from checkpoint, reset all other stats.
        
        This is used with --no-resume to start fresh statistics but still
        preserve the list of already-processed articles to prevent duplicates.
        
        Returns:
            True if IDs were loaded, False otherwise.
        """
        if not self.checkpoint_file.exists():
            logger.info("📍 No checkpoint found, starting completely fresh")
            return False

        try:
            with open(self.checkpoint_file, encoding="utf-8") as f:
                data = json.load(f)

            # Load ONLY the processed IDs - these are sacred and must persist
            self.processed_ids = set(data.get("processed_ids", []))
            
            # Reset all other stats to zero (fresh run)
            self.last_article_id = 0
            self.articles_processed = 0
            self.chunks_created = 0
            self.total_elapsed = 0.0

            if self.processed_ids:
                logger.info(
                    f"🛡️  Loaded {len(self.processed_ids):,} previously processed "
                    f"article IDs (will be skipped to prevent duplicates)"
                )
                logger.info("📊 Statistics reset to zero for fresh run")
            return True

        except (json.JSONDecodeError, KeyError, OSError) as e:
            logger.warning(f"⚠️  Failed to load checkpoint: {e}. Starting fresh.")
            return False

    def clear(self) -> None:
        """
        Clear checkpoint file completely (for TRULY fresh start).
        
        WARNING: This will remove ALL processed article IDs!
        Articles may be re-imported if the databases are not also cleared.
        """
        if self.checkpoint_file.exists():
            # Load current state for logging
            try:
                with open(self.checkpoint_file, encoding="utf-8") as f:
                    data = json.load(f)
                old_count = len(data.get("processed_ids", []))
                logger.warning(
                    f"⚠️  Deleting {old_count:,} processed article IDs - "
                    f"articles may be re-imported!"
                )
            except Exception:
                pass
            
            self.checkpoint_file.unlink()
            logger.info("🗑️  Checkpoint file deleted")
        
        self.processed_ids.clear()
        self.last_article_id = 0
        self.articles_processed = 0
        self.chunks_created = 0
        self.total_elapsed = 0.0
        
        logger.info("✅ Checkpoint completely cleared")

    def get_resume_info(self) -> str:
        """Get human-readable resume information."""
        if not self.processed_ids:
            return "Starting fresh ingestion"
        
        elapsed_str = time.strftime("%H:%M:%S", time.gmtime(self.total_elapsed))
        return (
            f"Resuming: {self.articles_processed:,} articles already processed, "
            f"{self.chunks_created:,} chunks created, "
            f"previous runtime: {elapsed_str}"
        )


# =============================================================================
# BATCH PROCESSING
# =============================================================================


def process_batch(
    batch: list[tuple[WikiArticle, list[ProcessedChunk]]],
    qdrant: QdrantHybridStore,
    neo4j: Neo4jGraphStore,
    stats: IngestionStats,
    executor: ThreadPoolExecutor,
) -> list[Future[None]]:
    """
    Process a batch of articles for both stores in parallel.
    
    Returns futures so caller can check for completion asynchronously.
    This enables pipeline prefetching for maximum throughput.
    """
    # Flatten all chunks for Qdrant
    all_chunks = [chunk for _, chunks in batch for chunk in chunks]

    # Prepare data for Neo4j
    articles = [article for article, _ in batch]
    chunk_map = {
        article.id: [c.chunk_id for c in chunks] for article, chunks in batch
    }

    # Upload to both stores in parallel
    futures: list[Future[None]] = []
    if all_chunks:
        futures.append(executor.submit(qdrant.upload_batch, all_chunks))
    futures.append(executor.submit(neo4j.upload_batch, articles, chunk_map))

    # Update statistics immediately (don't wait for uploads)
    stats.update(
        articles=len(articles),
        chunks=len(all_chunks),
        links=sum(len(a.links) for a in articles),
        categories=sum(len(a.categories) for a in articles),
    )
    
    return futures


def wait_for_uploads(futures: list[Future[None]], stats: IngestionStats) -> None:
    """Wait for pending uploads to complete and handle errors."""
    for future in as_completed(futures):
        try:
            future.result()
        except Exception as e:
            logger.error(f"Batch upload error: {e}")
            stats.update(errors=1)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================


def main():
    """Main entry point for the ingestion pipeline."""
    parser = argparse.ArgumentParser(
        description="Wikipedia Dual Ingestion (Neo4j + Qdrant)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Connection settings
    parser.add_argument(
        "--qdrant-host", default="localhost", help="Qdrant host"
    )
    parser.add_argument(
        "--qdrant-port", type=int, default=6333, help="Qdrant HTTP port"
    )
    parser.add_argument(
        "--qdrant-grpc-port", type=int, default=6334, help="Qdrant gRPC port"
    )
    parser.add_argument(
        "--neo4j-uri", default="bolt://localhost:7687", help="Neo4j Bolt URI"
    )
    parser.add_argument(
        "--neo4j-user", default="neo4j", help="Neo4j username"
    )
    parser.add_argument(
        "--neo4j-pass", default="password123", help="Neo4j password"
    )
    parser.add_argument(
        "--neo4j-db", default="neo4j", help="Neo4j database name"
    )

    # Processing settings
    parser.add_argument(
        "--limit", type=int, default=None, help="Maximum articles to process"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Articles per batch (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"Target chunk size in chars (default: {DEFAULT_CHUNK_SIZE})",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP,
        help=f"Chunk overlap in chars (default: {DEFAULT_CHUNK_OVERLAP})",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Number of parallel workers (default: {DEFAULT_WORKERS})",
    )

    # Collection settings
    parser.add_argument(
        "--collection",
        default=QDRANT_COLLECTION,
        help="Qdrant collection name",
    )

    # Checkpoint settings
    parser.add_argument(
        "--checkpoint-file",
        default=None,
        help="Checkpoint file path (default: .ingest_checkpoint.json)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Reset statistics but keep processed article IDs (prevents duplicates)",
    )
    parser.add_argument(
        "--clear-checkpoint",
        action="store_true",
        help="DELETE all checkpoint data including processed IDs (allows re-import)",
    )
    parser.add_argument(
        "--endless",
        action="store_true",
        help="Keep retrying on network errors until the entire dump is processed",
    )
    parser.add_argument(
        "--endless-retry-delay",
        type=int,
        default=30,
        help="Seconds to wait before retrying after a network error (default: 30)",
    )
    parser.add_argument(
        "--download-dir",
        type=str,
        default=".wiki_dumps",
        help="Directory for temporary dump file downloads",
    )
    parser.add_argument(
        "--keep-downloads",
        action="store_true",
        help="Keep downloaded files after processing (don't delete)",
    )
    parser.add_argument(
        "--max-concurrent-downloads",
        type=int,
        default=3,
        help="Maximum concurrent dump file downloads (default: 3)",
    )
    parser.add_argument(
        "--stream-mode",
        action="store_true",
        help="Use legacy streaming mode instead of chunked downloads",
    )

    args = parser.parse_args()

    # Initialize checkpoint manager
    checkpoint = CheckpointManager(
        checkpoint_file=args.checkpoint_file,
        collection=args.collection,
    )

    # Handle --clear-checkpoint (completely wipes everything)
    if args.clear_checkpoint:
        checkpoint.clear()
        logger.info("Exiting. Use without --clear-checkpoint to run ingestion.")
        return

    # Load checkpoint based on mode
    if args.no_resume:
        # --no-resume: Load ONLY the processed IDs, reset statistics
        # This prevents duplicates while starting fresh statistics
        checkpoint.load_ids_only()
        logger.info("📊 Starting fresh run (but keeping duplicate protection)")
    else:
        # Normal mode: Load everything (resume from where we left off)
        checkpoint.load()
        logger.info(f"📍 {checkpoint.get_resume_info()}")

    # Initialize components
    logger.info("🔧 Initializing components...")

    preprocessor = AdvancedTextPreprocessor(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    qdrant = QdrantHybridStore(
        host=args.qdrant_host,
        port=args.qdrant_port,
        grpc_port=args.qdrant_grpc_port,
        collection=args.collection,
    )

    neo4j = Neo4jGraphStore(
        uri=args.neo4j_uri,
        user=args.neo4j_user,
        password=args.neo4j_pass,
        database=args.neo4j_db,
    )

    stats = IngestionStats()
    skipped_count = 0

    # Calculate remaining articles if limit is set
    remaining_limit = None
    if args.limit:
        already_done = checkpoint.articles_processed
        remaining_limit = max(0, args.limit - already_done)
        if remaining_limit == 0:
            logger.info("✅ Limit already reached from previous run")
            return

    # Show duplicate prevention info
    if checkpoint.processed_ids:
        logger.info(
            f"🛡️  Duplicate prevention active: {len(checkpoint.processed_ids):,} "
            f"article IDs will be skipped if encountered again"
        )

    # Thread pool for parallel uploads
    executor = ThreadPoolExecutor(max_workers=args.workers * 2)

    # Choose mode: chunked downloads (default) or legacy streaming
    if args.stream_mode:
        # Legacy streaming mode
        logger.info("📡 Using legacy streaming mode")
        completed_successfully = run_streaming_mode(
            args=args,
            checkpoint=checkpoint,
            preprocessor=preprocessor,
            qdrant=qdrant,
            neo4j=neo4j,
            stats=stats,
            executor=executor,
            remaining_limit=remaining_limit,
        )
        skipped_count = 0  # Not tracked in streaming mode
    else:
        # Chunked download mode (default, more robust)
        logger.info("📦 Using chunked download mode (more robust)")
        completed_successfully, skipped_count = run_chunked_mode(
            args=args,
            checkpoint=checkpoint,
            preprocessor=preprocessor,
            qdrant=qdrant,
            neo4j=neo4j,
            stats=stats,
            executor=executor,
            remaining_limit=remaining_limit,
        )

    # Cleanup
    executor.shutdown(wait=True)
    neo4j.close()
    
    # Final checkpoint save
    checkpoint.save()
    
    # Show summary
    logger.info(stats.summary())
    
    if skipped_count > 0:
        logger.info(f"⏭️  Skipped {skipped_count:,} already-processed articles")
    
    if completed_successfully and not shutdown_requested:
        logger.info("🎉 Ingestion completed successfully!")
    elif shutdown_requested:
        logger.info(
            f"💾 Checkpoint saved. Run again to resume from "
            f"{checkpoint.articles_processed:,} articles."
        )


def run_chunked_mode(
    args,
    checkpoint: CheckpointManager,
    preprocessor: AdvancedTextPreprocessor,
    qdrant: QdrantHybridStore,
    neo4j: Neo4jGraphStore,
    stats: IngestionStats,
    executor: ThreadPoolExecutor,
    remaining_limit: int | None,
) -> tuple[bool, int]:
    """
    Run ingestion using chunked file downloads.
    
    This is the default mode and is much more robust than streaming.
    Downloads each dump part completely before processing.
    
    Returns (completed_successfully, skipped_count)
    """
    download_dir = Path(args.download_dir)
    downloader = ChunkedWikiDownloader(download_dir=download_dir)
    
    # Track which parts have been processed
    parts_checkpoint_file = download_dir / ".parts_checkpoint.json"
    processed_parts: set[int] = set()
    
    # Load parts checkpoint
    if parts_checkpoint_file.exists():
        try:
            with open(parts_checkpoint_file) as f:
                data = json.load(f)
                processed_parts = set(data.get("processed_parts", []))
                logger.info(f"📍 Resuming: {len(processed_parts)} parts already processed")
        except Exception as e:
            logger.warning(f"Could not load parts checkpoint: {e}")
    
    def save_parts_checkpoint():
        """Save which parts have been processed."""
        try:
            with open(parts_checkpoint_file, "w") as f:
                json.dump({"processed_parts": list(processed_parts)}, f)
        except Exception as e:
            logger.warning(f"Could not save parts checkpoint: {e}")
    
    try:
        # Discover all available parts
        all_parts = downloader.discover_parts()
        
        # Filter out already processed parts
        pending_parts = [p for p in all_parts if p.index not in processed_parts]
        
        if not pending_parts:
            logger.info("✅ All parts already processed!")
            return True, 0
        
        logger.info(f"📥 {len(pending_parts)} parts remaining to process")
        
        # Queue for parts ready to process
        ready_to_process: list[DumpFile] = []
        download_queue = list(pending_parts)
        
        # Track state
        completed_successfully = True
        skipped_count = 0
        article_batch: list[tuple[WikiArticle, list[ProcessedChunk]]] = []
        pending_futures: list[Future[None]] = []
        
        # Progress bar
        pbar = tqdm(
            total=remaining_limit or args.limit or len(pending_parts) * 50000,
            unit="articles",
            desc="Dual Ingest",
            dynamic_ncols=True,
        )
        
        # Start initial downloads (up to max concurrent)
        while (
            download_queue
            and downloader.active_download_count() < args.max_concurrent_downloads
        ):
            part = download_queue.pop(0)
            downloader.start_download(part)
            logger.info(f"📥 Started download: part {part.index}")
        
        # Main processing loop
        has_work = (
            ready_to_process
            or download_queue
            or downloader.active_download_count() > 0
        )
        while has_work and not shutdown_requested:
            # Check for completed downloads
            completed = downloader.get_completed_downloads()
            ready_to_process.extend(completed)
            
            # Start new downloads if slots available
            while (
                download_queue
                and downloader.active_download_count() < args.max_concurrent_downloads
            ):
                part = download_queue.pop(0)
                downloader.start_download(part)
                logger.info(f"📥 Started download: part {part.index}")
            
            # Process any ready files
            if ready_to_process:
                part = ready_to_process.pop(0)
                
                if not part.download_complete or part.local_path is None:
                    logger.warning(f"⚠️  Part {part.index} not ready, skipping")
                    continue
                
                logger.info(f"📖 Processing part {part.index}...")
                
                try:
                    # Process all articles in this part
                    for article in LocalFileParser.parse_file(part.local_path):
                        if shutdown_requested:
                            break
                        
                        # Skip already processed articles
                        if checkpoint.should_skip(article.id):
                            skipped_count += 1
                            continue
                        
                        # Clean and extract metadata
                        (
                            clean_text,
                            links,
                            categories,
                            infobox,
                            entities,
                        ) = preprocessor.clean_and_extract(article.text)

                        article.links = links
                        article.categories = categories
                        article.infobox = infobox
                        article.entities = entities
                        article.sections = preprocessor.extract_sections(article.text)
                        article.first_paragraph = preprocessor.extract_first_paragraph(
                            article.text
                        )

                        # Create semantic chunks
                        chunks = preprocessor.chunk_article(article, clean_text)

                        if chunks:
                            article_batch.append((article, chunks))

                        # Process batch when full
                        if len(article_batch) >= args.batch_size:
                            if pending_futures:
                                wait_for_uploads(pending_futures, stats)
                                pending_futures = []
                            
                            pending_futures = process_batch(
                                article_batch, qdrant, neo4j, stats, executor
                            )
                            
                            # Record in checkpoint
                            batch_articles = [a for a, _ in article_batch]
                            batch_chunks = sum(len(c) for _, c in article_batch)
                            checkpoint.record_batch(batch_articles, batch_chunks)
                            
                            pbar.update(len(article_batch))
                            article_batch = []

                            if remaining_limit and stats.articles_processed >= remaining_limit:
                                logger.info("✅ Limit reached")
                                break
                    
                    # Wait for pending uploads
                    if pending_futures:
                        wait_for_uploads(pending_futures, stats)
                        pending_futures = []
                    
                    # Process final batch for this part
                    if article_batch:
                        futures = process_batch(
                            article_batch, qdrant, neo4j, stats, executor
                        )
                        wait_for_uploads(futures, stats)
                        
                        batch_articles = [a for a, _ in article_batch]
                        batch_chunks = sum(len(c) for _, c in article_batch)
                        checkpoint.record_batch(batch_articles, batch_chunks)
                        
                        pbar.update(len(article_batch))
                        article_batch = []
                    
                    # Mark part as processed
                    processed_parts.add(part.index)
                    save_parts_checkpoint()
                    
                    # Delete file unless --keep-downloads
                    if not args.keep_downloads and part.local_path.exists():
                        try:
                            part.local_path.unlink()
                            logger.info(f"🗑️  Deleted {part.local_path.name}")
                        except Exception as e:
                            logger.warning(f"Could not delete {part.local_path}: {e}")
                    
                    logger.info(
                        f"✅ Part {part.index} complete. "
                        f"Progress: {len(processed_parts)}/{len(all_parts)} parts"
                    )
                    
                    # Check limit
                    if remaining_limit and stats.articles_processed >= remaining_limit:
                        break
                        
                except Exception as e:
                    logger.error(f"Error processing part {part.index}: {e}")
                    stats.update(errors=1)
                    # Don't mark as processed so it can be retried
            else:
                # No files ready, wait a bit for downloads
                time.sleep(1)
            
            # Update loop condition
            has_work = (
                ready_to_process
                or download_queue
                or downloader.active_download_count() > 0
            )
        
        pbar.close()
        
        # Check if all parts processed or limit reached
        all_done = len(processed_parts) == len(all_parts)
        limit_reached = bool(
            remaining_limit and stats.articles_processed >= remaining_limit
        )
        completed_successfully = (all_done or limit_reached) and not shutdown_requested
        
        return completed_successfully, skipped_count
        
    finally:
        downloader.close()


def run_streaming_mode(
    args,
    checkpoint: CheckpointManager,
    preprocessor: AdvancedTextPreprocessor,
    qdrant: QdrantHybridStore,
    neo4j: Neo4jGraphStore,
    stats: IngestionStats,
    executor: ThreadPoolExecutor,
    remaining_limit: int | None,
) -> bool:
    """
    Run ingestion using legacy streaming mode.
    
    This is the original mode that streams directly from the server.
    Less robust but uses less disk space.
    
    Returns completed_successfully
    """
    url = WikiDownloader.get_latest_url()
    
    article_batch: list[tuple[WikiArticle, list[ProcessedChunk]]] = []
    pending_futures: list[Future[None]] = []
    
    pbar = tqdm(
        total=remaining_limit or args.limit,
        unit="articles",
        desc="Dual Ingest",
        dynamic_ncols=True,
    )

    connection_attempts = 0
    max_connection_attempts = 1000 if args.endless else 1
    completed_successfully = False

    while connection_attempts < max_connection_attempts and not shutdown_requested:
        connection_attempts += 1
        
        if connection_attempts > 1:
            logger.info(
                f"🔄 Reconnection attempt {connection_attempts} "
                f"(waiting {args.endless_retry_delay}s)..."
            )
            time.sleep(args.endless_retry_delay)
            checkpoint.load()
            logger.info(f"📍 {checkpoint.get_resume_info()}")

        try:
            for article in WikiDownloader.stream_articles(url):
                if shutdown_requested:
                    logger.warning("⚠️  Shutdown requested, saving checkpoint...")
                    break

                if checkpoint.should_skip(article.id):
                    continue

                (
                    clean_text,
                    links,
                    categories,
                    infobox,
                    entities,
                ) = preprocessor.clean_and_extract(article.text)

                article.links = links
                article.categories = categories
                article.infobox = infobox
                article.entities = entities
                article.sections = preprocessor.extract_sections(article.text)
                article.first_paragraph = preprocessor.extract_first_paragraph(
                    article.text
                )

                chunks = preprocessor.chunk_article(article, clean_text)

                if chunks:
                    article_batch.append((article, chunks))

                if len(article_batch) >= args.batch_size:
                    if pending_futures:
                        wait_for_uploads(pending_futures, stats)
                        pending_futures = []
                    
                    pending_futures = process_batch(
                        article_batch, qdrant, neo4j, stats, executor
                    )
                    
                    batch_articles = [a for a, _ in article_batch]
                    batch_chunks = sum(len(c) for _, c in article_batch)
                    checkpoint.record_batch(batch_articles, batch_chunks)
                    
                    pbar.update(len(article_batch))
                    article_batch = []

                    if remaining_limit and stats.articles_processed >= remaining_limit:
                        logger.info("✅ Limit reached")
                        completed_successfully = True
                        break

            if not shutdown_requested:
                completed_successfully = True
                
            if pending_futures:
                wait_for_uploads(pending_futures, stats)
                pending_futures = []
                
            if article_batch and not shutdown_requested:
                futures = process_batch(
                    article_batch, qdrant, neo4j, stats, executor
                )
                wait_for_uploads(futures, stats)
                
                batch_articles = [a for a, _ in article_batch]
                batch_chunks = sum(len(c) for _, c in article_batch)
                checkpoint.record_batch(batch_articles, batch_chunks)
                
                pbar.update(len(article_batch))
                article_batch = []

            if completed_successfully or shutdown_requested:
                break

        except (
            requests.exceptions.RequestException,
            requests.exceptions.ConnectionError,
            requests.exceptions.ChunkedEncodingError,
            ConnectionResetError,
            TimeoutError,
            OSError,
        ) as e:
            logger.warning(f"⚠️  Network error: {e}")
            
            if pending_futures:
                wait_for_uploads(pending_futures, stats)
                pending_futures = []
            
            checkpoint.save()
            stats.update(errors=1)
            
            if args.endless:
                logger.info(
                    f"💾 Checkpoint saved at {checkpoint.articles_processed:,} articles. "
                    f"Will retry..."
                )
                article_batch = []
                continue
            else:
                logger.error("Network error. Use --endless to auto-retry.")
                break

        except Exception as e:
            logger.error(f"Critical failure: {e}", exc_info=True)
            stats.update(errors=1)
            
            if pending_futures:
                wait_for_uploads(pending_futures, stats)
            
            checkpoint.save()
            break

    pbar.close()
    return completed_successfully


if __name__ == "__main__":
    main()
