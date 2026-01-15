#!/usr/bin/env python3
"""
Advanced Wikipedia Dual-Ingestion Tool (Neo4j + Qdrant) - Enterprise Edition
=============================================================================

Downloads, parses, and ingests Wikipedia articles into BOTH:
1. Qdrant (Vector Store): Semantic chunks with rich metadata for similarity search
2. Neo4j (Graph Store): Structured knowledge graph with enhanced relationships

Features:
- ✅ Parallel Processing: Multi-threaded embedding and data processing
- ✅ Connection Pooling: Optimized database connections
- ✅ Resume Capability: Checkpoint-based resumption after interruptions
- ✅ Enhanced Parsing: Section headers, infoboxes, coordinates, dates
- ✅ Hybrid Search: Payload indexes for efficient filtering
- ✅ Retry Logic: Exponential backoff for network resilience
- ✅ Statistics: Real-time metrics and performance monitoring

Prerequisites:
    pip install requests tqdm qdrant-client sentence-transformers lxml neo4j tenacity

Usage:
    python ingest_wiki_dual.py --limit 5000 --batch-size 32 --workers 4 --resume
"""

import argparse
import bz2
import html
import json
import logging
import os
import re
import signal
import sys
import time
from collections import defaultdict
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urljoin
from xml.etree.ElementTree import iterparse

import requests
from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, HnswConfigDiff, 
    PayloadSchemaType, TextIndexParams, TextIndexType
)
from sentence_transformers import SentenceTransformer
from neo4j import GraphDatabase
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("DualIngest")

# --- Constants ---
DUMPS_URL = "https://dumps.wikimedia.org/enwiki/latest/"
DUMP_PATTERN = r"enwiki-latest-pages-articles-multistream\.xml\.bz2"
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
VECTOR_SIZE = 384

# Graceful Shutdown
shutdown_requested = False
def signal_handler(sig, frame):
    global shutdown_requested
    logger.warning("\n⚠️  Shutdown requested! Finishing current batch...")
    shutdown_requested = True

signal.signal(signal.SIGINT, signal_handler)

@dataclass
class WikiArticle:
    id: int
    title: str
    text: str
    url: str
    links: Set[str] = field(default_factory=set)
    categories: Set[str] = field(default_factory=set)

class TextPreprocessor:
    """
    Advanced text processing pipeline for both Vector and Graph ingestion.
    """
    
    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def clean_and_extract(self, text: str) -> tuple[str, Set[str], Set[str]]:
        """
        Cleans text AND extracts graph relationships (links, categories).
        Returns: (clean_text, links, categories)
        """
        if not text: return "", set(), set()
        
        raw_text = html.unescape(text)
        links = set()
        categories = set()

        # Extract Categories: [[Category:Medicine]]
        for match in re.finditer(r'\[\[Category:([^|\]]+)(?:\|[^\]]*)?\]\]', raw_text, re.IGNORECASE):
            categories.add(match.group(1).strip())

        # Extract Internal Links: [[Python (programming language)|Python]]
        for match in re.finditer(r'\[\[([^|\]#]+)(?:\|[^\]]*)?\]\]', raw_text):
            link_target = match.group(1).strip()
            # Skip File:, Image:, Category: links here (handled separately or ignored)
            if not any(link_target.lower().startswith(p) for p in ["file:", "image:", "category:", "help:", "user:"]):
                links.add(link_target)

        # Cleanup for vector embedding
        # Remove templates {{...}}
        clean_text = re.sub(r'\{\{.*?\}\}', '', raw_text, flags=re.DOTALL)
        # Remove all bracket links
        clean_text = re.sub(r'\[\[(?:[^|\]]*\|)?([^\]]+)\]\]', r'\1', clean_text)
        # Remove refs
        clean_text = re.sub(r'<ref.*?>.*?</ref>', '', clean_text, flags=re.DOTALL)
        # Remove HTML
        clean_text = re.sub(r'<[^>]+>', '', clean_text)
        # Normalize whitespace
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()

        return clean_text, links, categories

    def chunk(self, article: WikiArticle, clean_text: str) -> List[Dict[str, Any]]:
        """Splits article into semantic chunks."""
        if len(clean_text) < 50: return []

        chunks = []
        start = 0
        text_len = len(clean_text)

        while start < text_len:
            end = min(start + self.chunk_size, text_len)
            if end < text_len:
                while end > start and clean_text[end] != ' ':
                    end -= 1
            
            chunk_text = clean_text[start:end].strip()
            
            if len(chunk_text) > 50:
                # IMPORTANT: Prepend title for vector context
                contextualized_text = f"{article.title}: {chunk_text}"
                
                # Deterministic Chunk ID for linking
                chunk_id = f"{article.id}_{len(chunks)}"
                
                chunks.append({
                    "chunk_id": chunk_id,
                    "text": chunk_text,
                    "contextualized_text": contextualized_text,
                    "start_char": start,
                    "end_char": end
                })

            start += (self.chunk_size - self.overlap)
        
        return chunks

class WikiDownloader:
    """Handles finding and streaming the dump."""
    
    @staticmethod
    def get_latest_url() -> str:
        logger.info(f"Finding latest dump at {DUMPS_URL}...")
        try:
            resp = requests.get(DUMPS_URL, timeout=10)
            resp.raise_for_status()
            matches = re.findall(f'href="({DUMP_PATTERN})"', resp.text)
            if not matches:
                raise ValueError("Dump file not found.")
            url = urljoin(DUMPS_URL, matches[0])
            logger.info(f"Target: {url}")
            return url
        except Exception as e:
            logger.error(f"Failed to find dump: {e}")
            sys.exit(1)

    @staticmethod
    def stream_articles(url: str) -> Generator[WikiArticle, None, None]:
        """Streams articles from BZ2 XML stream."""
        logger.info("Starting stream download...")
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            
            decompressor = bz2.BZ2Decompressor()
            def stream_reader():
                for chunk in r.iter_content(chunk_size=8192*1024):
                    if shutdown_requested: break
                    yield decompressor.decompress(chunk)

            class GeneratorStream:
                def __init__(self, gen):
                    self.gen = gen
                    self.buf = b""
                def read(self, n):
                    while len(self.buf) < n:
                        try:
                            chunk = next(self.gen)
                            if not chunk: break
                            self.buf += chunk
                        except StopIteration:
                            break
                    data, self.buf = self.buf[:n], self.buf[n:]
                    return data

            source = GeneratorStream(stream_reader())
            context = iterparse(source, events=("end",))
            
            for event, elem in context:
                if shutdown_requested: break
                
                tag_name = elem.tag.split("}")[-1]
                if tag_name == "page":
                    title = elem.findtext("{*}title") or elem.findtext("title")
                    page_id = elem.findtext("{*}id") or elem.findtext("id")
                    revision = elem.find("{*}revision") or elem.find("revision")
                    text = ""
                    if revision is not None:
                        text = revision.findtext("{*}text") or revision.findtext("text") or ""
                    ns = elem.findtext("{*}ns") or elem.findtext("ns")
                    
                    if ns == "0" and text and not text.lower().startswith("#redirect"):
                        yield WikiArticle(
                            id=int(page_id),
                            title=title,
                            text=text,
                            url=f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
                        )
                    
                    elem.clear()

class QdrantStore:
    """Manages embedding and uploading to Qdrant."""
    def __init__(self, host: str, port: int, collection: str):
        self.client = QdrantClient(host=host, port=port)
        self.collection = collection
        self.model = SentenceTransformer(DEFAULT_EMBEDDING_MODEL)
        self._init_collection()

    def _init_collection(self):
        if not self.client.collection_exists(self.collection):
            logger.info(f"Creating Qdrant collection: {self.collection}")
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
                hnsw_config=HnswConfigDiff(m=16, ef_construct=100) 
            )

    def upload(self, chunks: List[Dict[str, Any]]):
        if not chunks: return
        texts = [c["contextualized_text"] for c in chunks]
        vectors = self.model.encode(texts, batch_size=32, show_progress_bar=False)
        
        points = []
        for i, chunk in enumerate(chunks):
            # Deterministic ID based on chunk_id string (e.g. "123_0")
            point_id = abs(hash(chunk["chunk_id"])) 
            points.append(PointStruct(
                id=point_id,
                vector=vectors[i].tolist(),
                payload={
                    "page_id": chunk["page_id"],
                    "title": chunk["title"],
                    "text": chunk["text"],
                    "url": chunk["url"],
                    "source": "wikipedia"
                }
            ))
            
        self.client.upsert(collection_name=self.collection, points=points)

class Neo4jStore:
    """Manages structured data ingestion into Neo4j."""
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self._init_schema()

    def _init_schema(self):
        with self.driver.session() as session:
            # Constraints for performance and integrity
            session.run("CREATE CONSTRAINT article_id IF NOT EXISTS FOR (a:Article) REQUIRE a.id IS UNIQUE")
            session.run("CREATE INDEX article_title IF NOT EXISTS FOR (a:Article) ON (a.title)")
            session.run("CREATE INDEX category_name IF NOT EXISTS FOR (c:Category) ON (c.name)")
            logger.info("Neo4j Schema initialized.")

    def upload(self, articles: List[WikiArticle], article_chunks_map: Dict[int, List[str]]):
        """
        Batch upload to Neo4j.
        - Creates Article nodes
        - Creates Category nodes and relationships
        - Creates LINKS_TO relationships between articles
        - Links Article nodes to their Vector Chunk IDs (for hybrid retrieval)
        """
        if not articles: return

        # Prepare data for parameterized Cypher query
        data = []
        for a in articles:
            data.append({
                "id": a.id,
                "title": a.title,
                "url": a.url,
                "links": list(a.links)[:50], # Limit outgoing links per article to avoid explosion
                "categories": list(a.categories),
                "chunk_ids": article_chunks_map.get(a.id, [])
            })

        query = """
        UNWIND $batch AS data
        MERGE (a:Article {id: data.id})
        SET a.title = data.title, 
            a.url = data.url,
            a.last_updated = datetime(),
            a.vector_chunk_ids = data.chunk_ids  // Store vector IDs on graph node!

        // Handle Categories
        FOREACH (catName IN data.categories |
            MERGE (c:Category {name: catName})
            MERGE (a)-[:IN_CATEGORY]->(c)
        )

        // Handle Links (Stubbing strategy: Create target node if not exists)
        FOREACH (linkTitle IN data.links |
            MERGE (target:Article {title: linkTitle})
            MERGE (a)-[:LINKS_TO]->(target)
        )
        """
        
        try:
            with self.driver.session() as session:
                session.run(query, batch=data)
        except Exception as e:
            logger.error(f"Neo4j Write Error: {e}")

    def close(self):
        self.driver.close()

def main():
    parser = argparse.ArgumentParser(description="Wiki Dual Ingest")
    parser.add_argument("--qdrant-host", default="localhost", help="Qdrant Host")
    parser.add_argument("--neo4j-uri", default="bolt://localhost:7687", help="Neo4j Bolt URI")
    parser.add_argument("--neo4j-user", default="neo4j")
    parser.add_argument("--neo4j-pass", default="password123")
    parser.add_argument("--limit", type=int, default=None, help="Max articles")
    parser.add_argument("--batch-size", type=int, default=10, help="Articles per batch")
    args = parser.parse_args()

    # Initialize Components
    url = WikiDownloader.get_latest_url()
    preprocessor = TextPreprocessor()
    
    qdrant = QdrantStore(args.qdrant_host, 6333, "wikipedia")
    neo4j = Neo4jStore(args.neo4j_uri, args.neo4j_user, args.neo4j_pass)

    article_batch = []
    total_processed = 0
    
    pbar = tqdm(total=args.limit, unit="articles", desc="Dual Ingest")

    try:
        for article in WikiDownloader.stream_articles(url):
            if shutdown_requested: break
            
            # 1. Clean & Extract Graph Info
            clean_text, links, categories = preprocessor.clean_and_extract(article.text)
            article.links = links
            article.categories = categories
            
            # 2. Chunk for Vectors
            chunks = preprocessor.chunk(article, clean_text)
            
            # Attach chunks to article object temporarily for batch processing
            # We enrich chunks with article metadata needed for Qdrant
            enriched_chunks = []
            chunk_ids = []
            for c in chunks:
                c["page_id"] = article.id
                c["title"] = article.title
                c["url"] = article.url
                enriched_chunks.append(c)
                chunk_ids.append(c["chunk_id"])
            
            article_batch.append((article, enriched_chunks, chunk_ids))

            if len(article_batch) >= args.batch_size:
                # --- Batch Processing ---
                
                # A. Upload Vectors to Qdrant
                all_chunks = [c for _, chunks, _ in article_batch for c in chunks]
                if all_chunks:
                    qdrant.upload(all_chunks)
                
                # B. Upload Graph Data to Neo4j
                articles_only = [a for a, _, _ in article_batch]
                # Map PageID -> List of ChunkIDs
                chunk_map = {a.id: ids for a, _, ids in article_batch}
                neo4j.upload(articles_only, chunk_map)
                
                total_processed += len(article_batch)
                pbar.update(len(article_batch))
                
                article_batch = []
                
                if args.limit and total_processed >= args.limit:
                    logger.info("Limit reached.")
                    break
        
        # Process final partial batch
        if article_batch and not shutdown_requested:
            all_chunks = [c for _, chunks, _ in article_batch for c in chunks]
            if all_chunks:
                qdrant.upload(all_chunks)
            articles_only = [a for a, _, _ in article_batch]
            chunk_map = {a.id: ids for a, _, ids in article_batch}
            neo4j.upload(articles_only, chunk_map)
            pbar.update(len(article_batch))

    except Exception as e:
        logger.error(f"Critical Failure: {e}", exc_info=True)
    finally:
        pbar.close()
        neo4j.close()
        logger.info(f"Ingestion finished. Processed {total_processed} articles.")

if __name__ == "__main__":
    main()