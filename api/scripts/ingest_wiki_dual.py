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
import bz2
import hashlib
import html
import logging
import math
import re
import signal
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from urllib.parse import urljoin, quote
from xml.etree.ElementTree import iterparse

import requests
from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    HnswConfigDiff,
    SparseVectorParams,
    SparseVector,
    Modifier,
    TextIndexParams,
    TextIndexType,
    TokenizerType,
    OptimizersConfigDiff,
    KeywordIndexParams,
    KeywordIndexType,
)
from sentence_transformers import SentenceTransformer
from neo4j import GraphDatabase

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
    INFOBOX_PATTERN = re.compile(
        r"\{\{Infobox\s+([^|}\n]+)((?:\|[^}]+)+)\}\}", re.IGNORECASE | re.DOTALL
    )
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
        match = self.INFOBOX_PATTERN.search(text)
        if not match:
            return None

        infobox_type = match.group(1).strip()
        properties_text = match.group(2)

        properties = {}
        for line in properties_text.split("|"):
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
# WIKIPEDIA DUMP DOWNLOADER
# =============================================================================


class WikiDownloader:
    """Handles finding and streaming the Wikipedia dump file."""

    @staticmethod
    def get_latest_url() -> str:
        """Find the latest dump file URL."""
        logger.info(f"Finding latest dump at {DUMPS_URL}...")
        try:
            resp = requests.get(DUMPS_URL, timeout=30)
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
        """Stream articles from BZ2-compressed XML dump."""
        logger.info("🚀 Starting stream download...")

        with requests.get(url, stream=True, timeout=60) as response:
            response.raise_for_status()

            decompressor = bz2.BZ2Decompressor()

            def stream_reader():
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if shutdown_requested:
                        break
                    try:
                        yield decompressor.decompress(chunk)
                    except EOFError:
                        break

            class StreamWrapper:
                """Wrapper to make generator look like file object."""

                def __init__(self, gen):
                    self.gen = gen
                    self.buffer = b""

                def read(self, n: int) -> bytes:
                    while len(self.buffer) < n:
                        try:
                            chunk = next(self.gen)
                            if not chunk:
                                break
                            self.buffer += chunk
                        except StopIteration:
                            break
                    data, self.buffer = self.buffer[:n], self.buffer[n:]
                    return data

            source = StreamWrapper(stream_reader())
            context = iterparse(source, events=("end",))

            for _event, elem in context:
                if shutdown_requested:
                    break

                tag_name = elem.tag.split("}")[-1]
                if tag_name == "page":
                    title = elem.findtext("{*}title") or elem.findtext("title")
                    page_id = elem.findtext("{*}id") or elem.findtext("id")
                    revision = elem.find("{*}revision") or elem.find("revision")
                    ns = elem.findtext("{*}ns") or elem.findtext("ns")

                    text = ""
                    if revision is not None:
                        text = (
                            revision.findtext("{*}text")
                            or revision.findtext("text")
                            or ""
                        )

                    # Only process main namespace articles, skip redirects
                    if (
                        ns == "0"
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

                    elem.clear()


# =============================================================================
# QDRANT HYBRID STORE
# =============================================================================


class QdrantHybridStore:
    """
    Qdrant store with hybrid search support:
    - Dense vectors (sentence-transformers)
    - Sparse vectors (BM25)
    - Full-text payload index
    """

    def __init__(
        self,
        host: str,
        port: int,
        collection: str,
        grpc_port: int = 6334,
        prefer_grpc: bool = True,
    ):
        self.client = QdrantClient(
            host=host,
            port=port,
            grpc_port=grpc_port,
            prefer_grpc=prefer_grpc,
            timeout=120,
        )
        self.collection = collection
        self.model = SentenceTransformer(DENSE_MODEL)
        self.tokenizer = BM25Tokenizer(vocab_size=50000)
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

    def upload_batch(self, chunks: list[ProcessedChunk]):
        """Upload a batch of chunks with both dense and sparse vectors."""
        if not chunks:
            return

        # Update IDF scores with this batch
        texts = [c.contextualized_text for c in chunks]
        self.tokenizer.update_idf(texts)

        # Compute dense embeddings
        dense_vectors = self.model.encode(
            texts,
            batch_size=64,
            show_progress_bar=False,
            normalize_embeddings=True,
        )

        # Build points with hybrid vectors
        points = []
        for i, chunk in enumerate(chunks):
            # Generate deterministic point ID
            point_id = abs(hash(chunk.chunk_id)) % (2**63)

            # Compute sparse vector
            indices, values = self.tokenizer.to_sparse_vector(
                chunk.contextualized_text
            )

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

        # Batch upsert (async for speed)
        self.client.upsert(
            collection_name=self.collection,
            points=points,
            wait=False,
        )


# =============================================================================
# NEO4J GRAPH STORE
# =============================================================================


class Neo4jGraphStore:
    """Neo4j store with optimized batch writes and rich relationship modeling."""

    def __init__(
        self, uri: str, user: str, password: str, database: str = "neo4j"
    ):
        self.driver = GraphDatabase.driver(
            uri,
            auth=(user, password),
            max_connection_pool_size=50,
            connection_acquisition_timeout=60,
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
                    session.run(query)
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
# BATCH PROCESSING
# =============================================================================


def process_batch(
    batch: list[tuple[WikiArticle, list[ProcessedChunk]]],
    qdrant: QdrantHybridStore,
    neo4j: Neo4jGraphStore,
    stats: IngestionStats,
    executor: ThreadPoolExecutor,
):
    """Process a batch of articles for both stores in parallel."""
    # Flatten all chunks for Qdrant
    all_chunks = [chunk for _, chunks in batch for chunk in chunks]

    # Prepare data for Neo4j
    articles = [article for article, _ in batch]
    chunk_map = {
        article.id: [c.chunk_id for c in chunks] for article, chunks in batch
    }

    # Upload to both stores in parallel
    futures = []
    if all_chunks:
        futures.append(executor.submit(qdrant.upload_batch, all_chunks))
    futures.append(executor.submit(neo4j.upload_batch, articles, chunk_map))

    # Wait for all uploads to complete
    for future in as_completed(futures):
        try:
            future.result()
        except Exception as e:
            logger.error(f"Batch upload error: {e}")
            stats.update(errors=1)

    # Update statistics
    stats.update(
        articles=len(articles),
        chunks=len(all_chunks),
        links=sum(len(a.links) for a in articles),
        categories=sum(len(a.categories) for a in articles),
    )


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
        "--batch-size", type=int, default=32, help="Articles per batch"
    )
    parser.add_argument(
        "--chunk-size", type=int, default=512, help="Target chunk size in chars"
    )
    parser.add_argument(
        "--chunk-overlap", type=int, default=64, help="Chunk overlap in chars"
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="Number of parallel workers"
    )

    # Collection settings
    parser.add_argument(
        "--collection",
        default=QDRANT_COLLECTION,
        help="Qdrant collection name",
    )

    args = parser.parse_args()

    # Initialize components
    logger.info("🔧 Initializing components...")

    url = WikiDownloader.get_latest_url()
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
    article_batch: list[tuple[WikiArticle, list[ProcessedChunk]]] = []

    pbar = tqdm(
        total=args.limit,
        unit="articles",
        desc="Dual Ingest",
        dynamic_ncols=True,
    )

    # Thread pool for parallel uploads
    executor = ThreadPoolExecutor(max_workers=args.workers)

    try:
        for article in WikiDownloader.stream_articles(url):
            if shutdown_requested:
                break

            # 1. Clean and extract all metadata
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

            # 2. Create semantic chunks
            chunks = preprocessor.chunk_article(article, clean_text)

            if chunks:
                article_batch.append((article, chunks))

            # 3. Process batch when full
            if len(article_batch) >= args.batch_size:
                process_batch(article_batch, qdrant, neo4j, stats, executor)
                pbar.update(len(article_batch))
                article_batch = []

                if args.limit and stats.articles_processed >= args.limit:
                    logger.info("✅ Limit reached")
                    break

        # Process final batch
        if article_batch and not shutdown_requested:
            process_batch(article_batch, qdrant, neo4j, stats, executor)
            pbar.update(len(article_batch))

    except Exception as e:
        logger.error(f"Critical failure: {e}", exc_info=True)
        stats.update(errors=1)
    finally:
        pbar.close()
        executor.shutdown(wait=True)
        neo4j.close()
        logger.info(stats.summary())


if __name__ == "__main__":
    main()
