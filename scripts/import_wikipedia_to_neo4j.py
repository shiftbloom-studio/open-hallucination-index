#!/usr/bin/env python3
"""
Wikipedia XML Dump to Neo4j Importer
=====================================

Imports Wikipedia XML dumps (English) into Neo4j graph database.
Handles large multi-gigabyte files by streaming and processing in batches.

Features:
- Streaming XML parsing (memory efficient)
- Batch processing with configurable sizes
- Resume capability via progress tracking
- Text cleaning (wiki markup, HTML, special chars)
- Parallel processing for text extraction

Usage:
    python scripts/import_wikipedia_to_neo4j.py
    python scripts/import_wikipedia_to_neo4j.py --resume
    python scripts/import_wikipedia_to_neo4j.py --batch-size 500 --workers 4
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import signal
import sys
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from html import unescape
from pathlib import Path
from threading import Event
from typing import Iterator, Any
from xml.etree.ElementTree import iterparse

# Global shutdown event for graceful termination
shutdown_event = Event()

# Third-party imports
try:
    import wikitextparser as wtp
except ImportError:
    print("Please install wikitextparser: pip install wikitextparser")
    sys.exit(1)

from neo4j import GraphDatabase

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("wikipedia_import.log"),
    ],
)
logger = logging.getLogger(__name__)


def signal_handler(signum, frame):
    """Handle interrupt signals gracefully."""
    logger.info("\n⚠️  Shutdown signal received. Finishing current batch...")
    shutdown_event.set()


# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


# Constants
PROGRESS_FILE = "wikipedia_import_progress.json"


def strip_namespace(tag: str) -> str:
    """
    Remove XML namespace from tag name.
    
    Handles tags like '{http://www.mediawiki.org/xml/export-0.11/}page' -> 'page'
    """
    if tag.startswith("{"):
        return tag.split("}", 1)[1] if "}" in tag else tag
    return tag

# Neo4j connection settings (match docker-compose.yml)
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password123")


@dataclass
class WikiArticle:
    """Represents a parsed Wikipedia article."""

    title: str
    page_id: int
    revision_id: int
    text: str
    categories: list[str] = field(default_factory=list)
    links: list[str] = field(default_factory=list)
    redirect_to: str | None = None


@dataclass
class ImportProgress:
    """Tracks import progress for resume capability."""

    processed_files: list[str] = field(default_factory=list)
    current_file: str | None = None
    last_page_id: int = 0
    total_articles: int = 0
    total_relationships: int = 0
    started_at: str = ""
    updated_at: str = ""

    def save(self, filepath: str = PROGRESS_FILE) -> None:
        """Save progress to JSON file."""
        self.updated_at = datetime.now().isoformat()
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.__dict__, f, indent=2)

    @classmethod
    def load(cls, filepath: str = PROGRESS_FILE) -> "ImportProgress":
        """Load progress from JSON file."""
        if not os.path.exists(filepath):
            return cls(started_at=datetime.now().isoformat())
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)


def remove_accents(input_str: str) -> str:
    """
    Remove accents from string, keeping ASCII characters only.

    Args:
        input_str: Input string with potential accents.

    Returns:
        ASCII-only string.
    """
    nfkd_form = unicodedata.normalize("NFKD", input_str)
    return nfkd_form.encode("ASCII", "ignore").decode("ASCII")


def html_to_text(text: str) -> str:
    """
    Remove HTML tags and entities from text.

    Args:
        text: Text potentially containing HTML.

    Returns:
        Clean text without HTML.
    """
    # Unescape HTML entities
    text = unescape(text)
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", "", text)
    # Remove HTML comments
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    return text


def clean_wiki_text(text: str, keep_accents: bool = True) -> str:
    """
    Clean Wikipedia markup text to plain text.

    Based on the dewiki function but adapted for English Wikipedia.

    Args:
        text: Raw Wikipedia markup text.
        keep_accents: Whether to keep accented characters (default True for English).

    Returns:
        Cleaned plain text.
    """
    if not text:
        return ""

    try:
        # Parse wiki markup to plain text
        parsed = wtp.parse(text)
        text = parsed.plain_text()
    except Exception as e:
        logger.debug(f"Wiki parsing failed, using regex fallback: {e}")
        # Fallback: basic wiki markup removal
        text = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", r"\1", text)  # [[link|text]] -> text
        text = re.sub(r"\{\{[^}]*\}\}", "", text)  # Remove templates
        text = re.sub(r"'''?", "", text)  # Remove bold/italic markers

    # Remove HTML
    text = html_to_text(text)

    # Remove content in curly braces (templates, infoboxes)
    text = re.sub(r"\{[^}]*\}", "", text)

    # Remove references like [1], [2], etc.
    text = re.sub(r"\[\d+\]", "", text)

    # Remove file/image references
    text = re.sub(r"\[\[File:[^\]]*\]\]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\[\[Image:[^\]]*\]\]", "", text, flags=re.IGNORECASE)

    # Replace newlines with spaces
    text = text.replace("\n", " ")

    # Replace multiple whitespace with single space
    text = re.sub(r"\s+", " ", text)

    # Remove non-word characters except basic punctuation
    # Keep letters, numbers, spaces, and common punctuation
    text = re.sub(r"[^\w\s.,;:!?'\"-]", "", text)

    # Optionally remove accents
    if not keep_accents:
        text = remove_accents(text)

    return text.strip()


def extract_categories(text: str) -> list[str]:
    """
    Extract category names from Wikipedia markup.

    Args:
        text: Raw Wikipedia markup text.

    Returns:
        List of category names.
    """
    categories = []
    # Match [[Category:Name]] or [[Category:Name|Sort key]]
    pattern = r"\[\[Category:([^|\]]+)(?:\|[^\]]*)?\]\]"
    for match in re.finditer(pattern, text, re.IGNORECASE):
        category = match.group(1).strip()
        if category:
            categories.append(category)
    return categories


def extract_links(text: str) -> list[str]:
    """
    Extract internal Wikipedia links from markup.

    Args:
        text: Raw Wikipedia markup text.

    Returns:
        List of linked article titles.
    """
    links = []
    # Match [[Title]] or [[Title|Display text]]
    pattern = r"\[\[([^|\]#]+)(?:[#|][^\]]*)?\]\]"
    for match in re.finditer(pattern, text):
        link = match.group(1).strip()
        # Skip special namespaces
        if link and ":" not in link:
            links.append(link)
    return links[:100]  # Limit to prevent excessive relationships


def extract_redirect(text: str) -> str | None:
    """
    Check if page is a redirect and extract target.

    Args:
        text: Raw Wikipedia markup text.

    Returns:
        Redirect target title or None.
    """
    match = re.match(r"#REDIRECT\s*\[\[([^\]]+)\]\]", text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


def process_article_text(article_data: tuple[str, int, int, str]) -> WikiArticle | None:
    """
    Process a single article's text (used for parallel processing).

    Args:
        article_data: Tuple of (title, page_id, revision_id, raw_text).

    Returns:
        WikiArticle or None if processing fails.
    """
    title, page_id, revision_id, raw_text = article_data

    try:
        # Check for redirect
        redirect_to = extract_redirect(raw_text)
        if redirect_to:
            return WikiArticle(
                title=title,
                page_id=page_id,
                revision_id=revision_id,
                text="",
                redirect_to=redirect_to,
            )

        # Extract metadata before cleaning
        categories = extract_categories(raw_text)
        links = extract_links(raw_text)

        # Clean the text
        clean_text = clean_wiki_text(raw_text)

        # Skip very short articles (likely stubs or disambiguation)
        if len(clean_text) < 100:
            return None

        return WikiArticle(
            title=title,
            page_id=page_id,
            revision_id=revision_id,
            text=clean_text[:50000],  # Limit text length for Neo4j
            categories=categories,
            links=links,
        )
    except Exception as e:
        logger.warning(f"Failed to process article '{title}': {e}")
        return None


def find_wikipedia_dumps(directory: str) -> list[Path]:
    """
    Find all Wikipedia XML dump files in directory.

    Matches files like: enwiki-*-pages-articles*.xml*

    Args:
        directory: Path to search for dump files.

    Returns:
        List of paths to dump files, sorted.
    """
    dump_dir = Path(directory)
    if not dump_dir.exists():
        logger.error(f"Directory not found: {directory}")
        return []

    # Match various Wikipedia dump file patterns
    patterns = [
        "*.xml",
        "*.xml-*",
        "*pages-articles*.xml*",
    ]

    files = []
    for pattern in patterns:
        files.extend(dump_dir.glob(pattern))

    # Remove duplicates and sort
    files = sorted(set(files))

    # Filter out non-files
    files = [f for f in files if f.is_file()]

    logger.info(f"Found {len(files)} Wikipedia dump files")
    for f in files:
        size_mb = f.stat().st_size / (1024 * 1024)
        logger.info(f"  - {f.name} ({size_mb:.1f} MB)")

    return files


def iter_wiki_articles(
    filepath: Path,
    start_after_page_id: int = 0,
) -> Iterator[tuple[str, int, int, str]]:
    """
    Stream Wikipedia articles from XML dump file.

    Uses iterparse for memory-efficient processing.

    Args:
        filepath: Path to XML dump file.
        start_after_page_id: Skip articles until this page ID (for resume).

    Yields:
        Tuples of (title, page_id, revision_id, raw_text).
    """
    logger.info(f"Parsing: {filepath}")

    # Handle potential compressed files
    if str(filepath).endswith(".bz2"):
        import bz2

        open_func = bz2.open
    elif str(filepath).endswith(".gz"):
        import gzip

        open_func = gzip.open
    else:
        open_func = open

    context = None
    try:
        file_handle = open_func(filepath, "rb")
        context = iterparse(file_handle, events=("start", "end"))

        # Variables to track current page
        current_page: dict[str, Any] = {}
        in_page = False
        skip_namespaces = {"Talk", "User", "Wikipedia", "File", "MediaWiki", "Template", "Help", "Category", "Portal", "Draft", "Module"}

        for event, elem in context:
            tag = strip_namespace(elem.tag)

            if event == "start":
                if tag == "page":
                    in_page = True
                    current_page = {}

            elif event == "end":
                if not in_page:
                    elem.clear()
                    continue

                if tag == "title":
                    current_page["title"] = elem.text or ""
                elif tag == "id" and "page_id" not in current_page:
                    # First id is page id, second is revision id
                    current_page["page_id"] = int(elem.text or 0)
                elif tag == "id" and "revision_id" not in current_page:
                    current_page["revision_id"] = int(elem.text or 0)
                elif tag == "ns":
                    current_page["namespace"] = int(elem.text or 0)
                elif tag == "text":
                    current_page["text"] = elem.text or ""
                elif tag == "page":
                    in_page = False

                    # Skip non-article namespaces (ns=0 is main namespace)
                    if current_page.get("namespace", 0) != 0:
                        current_page = {}
                        elem.clear()
                        continue

                    # Skip if title starts with special namespace
                    title = current_page.get("title", "")
                    if any(title.startswith(f"{ns}:") for ns in skip_namespaces):
                        current_page = {}
                        elem.clear()
                        continue

                    # Skip if we're resuming and haven't reached the checkpoint
                    page_id = current_page.get("page_id", 0)
                    if page_id <= start_after_page_id:
                        current_page = {}
                        elem.clear()
                        continue

                    # Yield the article data
                    if current_page.get("text"):
                        yield (
                            current_page.get("title", ""),
                            current_page.get("page_id", 0),
                            current_page.get("revision_id", 0),
                            current_page.get("text", ""),
                        )

                    current_page = {}

                # Clear element to free memory
                elem.clear()

    except Exception as e:
        logger.error(f"Error parsing {filepath}: {e}")
        raise
    finally:
        if context is not None:
            del context


class Neo4jImporter:
    """Handles batch imports to Neo4j."""

    def __init__(self, uri: str, user: str, password: str):
        """Initialize Neo4j connection."""
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self._setup_schema()

    def _setup_schema(self) -> None:
        """Create indexes and constraints for better performance."""
        with self.driver.session() as session:
            # Drop old title constraint if it exists (we only need page_id unique)
            try:
                session.run("DROP CONSTRAINT article_title IF EXISTS")
            except Exception:
                pass
            
            # Constraints - only page_id needs to be unique
            # Title is not unique because we create stub articles for links
            constraints = [
                "CREATE CONSTRAINT article_page_id IF NOT EXISTS FOR (a:Article) REQUIRE a.page_id IS UNIQUE",
                "CREATE CONSTRAINT category_name IF NOT EXISTS FOR (c:Category) REQUIRE c.name IS UNIQUE",
            ]
            for constraint in constraints:
                try:
                    session.run(constraint)
                except Exception as e:
                    logger.debug(f"Constraint creation: {e}")

            # Indexes for faster lookups
            indexes = [
                "CREATE INDEX article_title_index IF NOT EXISTS FOR (a:Article) ON (a.title)",
                "CREATE INDEX category_name_index IF NOT EXISTS FOR (c:Category) ON (c.name)",
            ]
            for index in indexes:
                try:
                    session.run(index)
                except Exception as e:
                    logger.debug(f"Index creation: {e}")

        logger.info("Neo4j schema setup complete")

    def import_batch(self, articles: list[WikiArticle]) -> tuple[int, int]:
        """
        Import a batch of articles to Neo4j.

        Args:
            articles: List of WikiArticle objects.

        Returns:
            Tuple of (articles_created, relationships_created).
        """
        if not articles:
            return 0, 0

        articles_created = 0
        relationships_created = 0

        with self.driver.session() as session:
            # First pass: Create/update article nodes
            article_data = [
                {
                    "title": a.title,
                    "page_id": a.page_id,
                    "revision_id": a.revision_id,
                    "text": a.text[:10000],  # Truncate for storage
                    "text_length": len(a.text),
                    "is_redirect": a.redirect_to is not None,
                    "redirect_to": a.redirect_to,
                }
                for a in articles
            ]

            result = session.run(
                """
                UNWIND $articles AS article
                // First try to find an existing stub by title and update it
                OPTIONAL MATCH (stub:Article {title: article.title, stub: true})
                // If stub exists, update it with the real data
                FOREACH (s IN CASE WHEN stub IS NOT NULL THEN [stub] ELSE [] END |
                    SET s.page_id = article.page_id,
                        s.revision_id = article.revision_id,
                        s.text = article.text,
                        s.text_length = article.text_length,
                        s.is_redirect = article.is_redirect,
                        s.redirect_to = article.redirect_to,
                        s.stub = false,
                        s.imported_at = datetime()
                )
                // If no stub, merge by page_id
                WITH article, stub
                WHERE stub IS NULL
                MERGE (a:Article {page_id: article.page_id})
                ON CREATE SET
                    a.title = article.title,
                    a.revision_id = article.revision_id,
                    a.text = article.text,
                    a.text_length = article.text_length,
                    a.is_redirect = article.is_redirect,
                    a.redirect_to = article.redirect_to,
                    a.stub = false,
                    a.imported_at = datetime()
                ON MATCH SET
                    a.title = article.title,
                    a.revision_id = article.revision_id,
                    a.text = article.text,
                    a.text_length = article.text_length,
                    a.is_redirect = article.is_redirect,
                    a.redirect_to = article.redirect_to,
                    a.stub = false,
                    a.updated_at = datetime()
                RETURN count(*) as count
                """,
                articles=article_data,
            )
            articles_created = result.single()["count"]

            # Second pass: Create category relationships
            for article in articles:
                if article.categories:
                    cat_result = session.run(
                        """
                        MATCH (a:Article {page_id: $page_id})
                        UNWIND $categories AS cat_name
                        MERGE (c:Category {name: cat_name})
                        MERGE (a)-[r:IN_CATEGORY]->(c)
                        RETURN count(r) as count
                        """,
                        page_id=article.page_id,
                        categories=article.categories[:20],  # Limit categories
                    )
                    relationships_created += cat_result.single()["count"]

            # Third pass: Create article links (limited for performance)
            # Use MERGE on title but allow updates when real article comes
            for article in articles:
                if article.links and not article.redirect_to:
                    link_result = session.run(
                        """
                        MATCH (source:Article {page_id: $page_id})
                        UNWIND $links AS target_title
                        MERGE (target:Article {title: target_title})
                        ON CREATE SET target.stub = true, target.page_id = null
                        MERGE (source)-[r:LINKS_TO]->(target)
                        RETURN count(r) as count
                        """,
                        page_id=article.page_id,
                        links=article.links[:30],  # Limit links per article
                    )
                    relationships_created += link_result.single()["count"]

            # Fourth pass: Create redirect relationships
            redirects = [a for a in articles if a.redirect_to]
            if redirects:
                redirect_data = [
                    {"page_id": a.page_id, "target": a.redirect_to}
                    for a in redirects
                ]
                redirect_result = session.run(
                    """
                    UNWIND $redirects AS redir
                    MATCH (source:Article {page_id: redir.page_id})
                    MERGE (target:Article {title: redir.target})
                    ON CREATE SET target.stub = true
                    MERGE (source)-[r:REDIRECTS_TO]->(target)
                    RETURN count(r) as count
                    """,
                    redirects=redirect_data,
                )
                relationships_created += redirect_result.single()["count"]

        return articles_created, relationships_created

    def get_stats(self) -> dict[str, int]:
        """Get current database statistics."""
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (a:Article)
                WITH count(a) as articles
                MATCH (c:Category)
                WITH articles, count(c) as categories
                MATCH ()-[r]->()
                RETURN articles, categories, count(r) as relationships
                """
            )
            record = result.single()
            return {
                "articles": record["articles"],
                "categories": record["categories"],
                "relationships": record["relationships"],
            }

    def close(self) -> None:
        """Close the driver connection."""
        self.driver.close()


def main():
    """Main entry point for Wikipedia import."""
    parser = argparse.ArgumentParser(
        description="Import Wikipedia XML dumps to Neo4j"
    )
    parser.add_argument(
        "--directory",
        "-d",
        default="wikipedia-knowledge",
        help="Directory containing Wikipedia XML dump files",
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=100,
        help="Number of articles to process before writing to Neo4j",
    )
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=4,
        help="Number of parallel workers for text processing",
    )
    parser.add_argument(
        "--resume",
        "-r",
        action="store_true",
        help="Resume from last checkpoint (auto-detected if progress file exists)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start fresh, ignoring any existing progress file",
    )
    parser.add_argument(
        "--max-articles",
        "-m",
        type=int,
        default=None,
        help="Maximum number of articles to import (for testing)",
    )
    parser.add_argument(
        "--neo4j-uri",
        default=NEO4J_URI,
        help="Neo4j connection URI",
    )
    parser.add_argument(
        "--neo4j-user",
        default=NEO4J_USER,
        help="Neo4j username",
    )
    parser.add_argument(
        "--neo4j-password",
        default=NEO4J_PASSWORD,
        help="Neo4j password",
    )

    args = parser.parse_args()

    # Load or create progress tracker
    # Auto-detect if we should resume: if progress file exists and has data
    progress_file_exists = os.path.exists(PROGRESS_FILE)

    if args.no_resume:
        # User explicitly wants to start fresh
        logger.info("Starting fresh (--no-resume specified)")
        if progress_file_exists:
            os.remove(PROGRESS_FILE)
            logger.info(f"Removed existing progress file: {PROGRESS_FILE}")
        progress = ImportProgress(started_at=datetime.now().isoformat())
    elif args.resume or progress_file_exists:
        progress = ImportProgress.load()
        if progress.total_articles > 0:
            logger.info(f"Resuming from checkpoint: {progress.total_articles:,} articles processed")
            logger.info(f"  Last file: {progress.current_file}")
            logger.info(f"  Last page ID: {progress.last_page_id}")
            if not args.resume and progress_file_exists:
                logger.info("  (Auto-detected progress file. Use --no-resume to start fresh)")
        else:
            progress = ImportProgress(started_at=datetime.now().isoformat())
    else:
        progress = ImportProgress(started_at=datetime.now().isoformat())

    # Find dump files
    dump_files = find_wikipedia_dumps(args.directory)
    if not dump_files:
        logger.error(f"No Wikipedia dump files found in {args.directory}")
        sys.exit(1)

    # Filter out already processed files
    if args.resume:
        dump_files = [f for f in dump_files if str(f) not in progress.processed_files]

    # Connect to Neo4j
    logger.info(f"Connecting to Neo4j at {args.neo4j_uri}")
    try:
        importer = Neo4jImporter(args.neo4j_uri, args.neo4j_user, args.neo4j_password)
    except Exception as e:
        logger.error(f"Failed to connect to Neo4j: {e}")
        sys.exit(1)

    total_imported = progress.total_articles
    total_relationships = progress.total_relationships

    try:
        for dump_file in dump_files:
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing: {dump_file.name}")
            logger.info(f"{'='*60}")

            # Check if we're resuming in the middle of this file
            # IMPORTANT: Check BEFORE updating current_file
            if progress.current_file == str(dump_file) and progress.last_page_id > 0:
                start_after = progress.last_page_id
                logger.info(f"Resuming from page ID {start_after:,}")
            else:
                start_after = 0
                # Reset page ID when starting a new file
                progress.last_page_id = 0

            progress.current_file = str(dump_file)
            progress.save()  # Save immediately so we know which file we're on

            batch: list[tuple[str, int, int, str]] = []
            file_articles = 0
            should_break = False

            # Process articles in batches (using ThreadPoolExecutor for stability)
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                for article_data in iter_wiki_articles(dump_file, start_after):
                    # Check for shutdown signal
                    if shutdown_event.is_set():
                        logger.info("Shutdown requested, saving progress...")
                        should_break = True
                        break
                    
                    batch.append(article_data)

                    if len(batch) >= args.batch_size:
                        # Process batch in parallel
                        futures = [
                            executor.submit(process_article_text, data)
                            for data in batch
                        ]

                        processed_articles = []
                        for future in as_completed(futures):
                            try:
                                result = future.result(timeout=60)
                                if result is not None:
                                    processed_articles.append(result)
                            except Exception as e:
                                logger.warning(f"Article processing failed: {e}")

                        # Import to Neo4j
                        if processed_articles:
                            try:
                                articles_created, rels_created = importer.import_batch(
                                    processed_articles
                                )
                                total_imported += articles_created
                                total_relationships += rels_created
                                file_articles += articles_created

                                # Update progress
                                progress.last_page_id = max(a.page_id for a in processed_articles)
                                progress.total_articles = total_imported
                                progress.total_relationships = total_relationships

                                # Log progress
                                logger.info(
                                    f"Imported: {total_imported:,} articles, "
                                    f"{total_relationships:,} relationships "
                                    f"(batch: +{articles_created})"
                                )
                            except Exception as e:
                                logger.error(f"Batch import failed: {e}")
                                # Save progress and continue with next batch
                                progress.save()

                        # Save checkpoint after each batch for reliable resume
                        progress.save()

                        batch = []

                        # Check max articles limit
                        if args.max_articles and total_imported >= args.max_articles:
                            logger.info(f"Reached max articles limit: {args.max_articles}")
                            should_break = True
                            break

                # Process remaining batch (only if not shutting down)
                if batch and not shutdown_event.is_set():
                    futures = [
                        executor.submit(process_article_text, data)
                        for data in batch
                    ]
                    processed_articles = []
                    for future in as_completed(futures):
                        try:
                            result = future.result(timeout=60)
                            if result is not None:
                                processed_articles.append(result)
                        except Exception as e:
                            logger.warning(f"Article processing failed: {e}")

                    if processed_articles:
                        try:
                            articles_created, rels_created = importer.import_batch(
                                processed_articles
                            )
                            total_imported += articles_created
                            total_relationships += rels_created
                            file_articles += articles_created
                            
                            # Update progress for remaining batch
                            progress.last_page_id = max(a.page_id for a in processed_articles)
                            progress.total_articles = total_imported
                            progress.total_relationships = total_relationships
                            progress.save()
                        except Exception as e:
                            logger.error(f"Final batch import failed: {e}")
                            progress.save()

            # Only mark file as fully processed if we completed naturally
            file_fully_processed = not should_break and not shutdown_event.is_set()
            
            if file_fully_processed:
                # Mark file as processed only if we went through the entire file
                progress.processed_files.append(str(dump_file))
                progress.last_page_id = 0  # Reset for next file
                progress.save()
                logger.info(f"Completed {dump_file.name}: {file_articles:,} articles (file fully processed)")
            else:
                # Keep the current file and page_id for resume
                progress.save()
                logger.info(f"Paused {dump_file.name}: {file_articles:,} articles (will resume from page_id {progress.last_page_id})")

            if should_break or shutdown_event.is_set():
                break

    except KeyboardInterrupt:
        logger.info("\n⚠️  Import interrupted. Progress saved.")
        progress.save()
    except Exception as e:
        logger.error(f"Import failed: {e}")
        progress.save()
    finally:
        try:
            importer.close()
        except Exception:
            pass  # Ignore errors during cleanup

    # Final statistics
    logger.info(f"\n{'='*60}")
    logger.info("Import Complete!")
    logger.info(f"{'='*60}")
    logger.info(f"Total articles imported: {total_imported:,}")
    logger.info(f"Total relationships created: {total_relationships:,}")

    # Get final Neo4j stats
    try:
        importer = Neo4jImporter(args.neo4j_uri, args.neo4j_user, args.neo4j_password)
        stats = importer.get_stats()
        logger.info(f"\nNeo4j Database Statistics:")
        logger.info(f"  Articles: {stats['articles']:,}")
        logger.info(f"  Categories: {stats['categories']:,}")
        logger.info(f"  Relationships: {stats['relationships']:,}")
        importer.close()
    except Exception as e:
        logger.warning(f"Could not retrieve final stats: {e}")

    progress.save()
    logger.info(f"\nProgress saved to {PROGRESS_FILE}")


if __name__ == "__main__":
    main()
