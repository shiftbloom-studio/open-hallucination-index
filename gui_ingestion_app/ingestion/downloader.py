"""
Wikipedia dump file downloader with async parallelism.

Features:
- Parallel downloads of dump parts
- Resume support for interrupted downloads
- Producer pattern - fills download queue for processing
- Non-blocking design for pipeline integration
"""

from __future__ import annotations

import bz2
import logging
import re
import signal
import time
from collections.abc import Generator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import quote, urljoin

import requests
from lxml import etree  # type: ignore[import-untyped]
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ingestion.models import WikiArticle

logger = logging.getLogger("ingestion.downloader")

# --- Constants ---
DUMPS_URL = "https://dumps.wikimedia.org/enwiki/latest/"
DUMP_PATTERN = r"enwiki-latest-pages-articles-multistream\.xml\.bz2"

# Graceful shutdown flag
shutdown_requested = False


def _signal_handler(sig, frame):
    """Handle graceful shutdown on SIGINT/SIGTERM."""
    global shutdown_requested
    logger.warning("\n⚠️  Shutdown requested! Finishing current batch...")
    shutdown_requested = True


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


def is_shutdown_requested() -> bool:
    """Check if shutdown has been requested."""
    return shutdown_requested


def request_shutdown() -> None:
    """Request a graceful shutdown."""
    global shutdown_requested
    shutdown_requested = True


# =============================================================================
# DUMP FILE MODEL
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


# =============================================================================
# CHUNKED WIKI DOWNLOADER
# =============================================================================


@dataclass
class DownloadProgress:
    """Real-time download progress tracking."""

    active_downloads: dict[int, float] = field(
        default_factory=dict
    )  # part -> progress %
    completed_parts: int = 0
    total_parts: int = 0
    bytes_downloaded: int = 0
    download_speed_mbps: float = 0.0


class ChunkedWikiDownloader:
    """
    Downloads Wikipedia dump files in parallel chunks.

    The dump is split into ~27 parts. This class:
    1. Discovers all available parts
    2. Downloads them in parallel (configurable concurrency)
    3. Queues completed downloads for processing
    4. Deletes files after processing (optional)

    Designed as a producer in a producer-consumer pipeline:
    - Download thread pool fills the ready queue
    - Main pipeline consumes from the queue
    - Non-blocking status updates
    """

    # Pattern to match multistream part files
    PART_PATTERN = re.compile(
        r'href="(enwiki-latest-pages-articles-multistream(\d+)\.xml[^"]*\.bz2)"'
    )

    # Download settings
    CHUNK_SIZE = 2 * 1024 * 1024  # 2MB chunks for better throughput
    DOWNLOAD_TIMEOUT = (30, 600)  # (connect, read) timeouts
    MAX_RETRIES = 10
    RETRY_BACKOFF = 5

    def __init__(
        self,
        download_dir: Path | None = None,
        max_concurrent: int = 4,
    ):
        self.download_dir = download_dir or Path(".wiki_dumps")
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.max_concurrent = max_concurrent
        self.session = self._create_session()
        self._download_executor = ThreadPoolExecutor(
            max_workers=max_concurrent,
            thread_name_prefix="downloader",
        )
        self._active_downloads: dict[int, Future[DumpFile]] = {}
        self._all_parts: list[DumpFile] = []
        self._progress = DownloadProgress()

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
            pool_connections=self.max_concurrent * 2,
            pool_maxsize=self.max_concurrent * 2,
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

                parts.append(
                    DumpFile(
                        index=index,
                        filename=filename,
                        url=urljoin(DUMPS_URL, filename),
                        local_path=self.download_dir / filename,
                    )
                )

            # Sort by index
            parts.sort(key=lambda p: p.index)
            self._all_parts = parts
            self._progress.total_parts = len(parts)

            logger.info(f"📦 Found {len(parts)} dump parts (1-{parts[-1].index})")
            return parts

        except Exception as e:
            logger.error(f"Failed to discover dump parts: {e}")
            raise

    def check_existing_downloads(
        self, parts: list[DumpFile], verify_sizes: bool = True
    ) -> tuple[list[DumpFile], list[DumpFile]]:
        """
        Check which parts are already fully downloaded locally.

        Args:
            parts: List of dump file parts to check
            verify_sizes: If True, verify file sizes match server (slower but accurate)

        Returns:
            Tuple of (already_downloaded, need_download) parts
        """
        already_downloaded: list[DumpFile] = []
        need_download: list[DumpFile] = []

        logger.info(f"🔍 Checking for existing downloads in {self.download_dir}...")

        # First pass: check local files exist with non-zero size
        for part in parts:
            local_path = part.local_path or self.download_dir / part.filename
            part.local_path = local_path

            if local_path.exists() and local_path.stat().st_size > 0:
                part.size_bytes = local_path.stat().st_size
                already_downloaded.append(part)
            else:
                need_download.append(part)

        if not already_downloaded:
            logger.info("📥 No existing downloads found")
            return [], parts

        # Second pass: optionally verify sizes match server
        if verify_sizes and already_downloaded:
            logger.info(
                f"🔍 Verifying {len(already_downloaded)} existing files against server..."
            )
            verified: list[DumpFile] = []

            for part in already_downloaded:
                try:
                    head_resp = self.session.head(part.url, timeout=30)
                    expected_size = int(head_resp.headers.get("content-length", 0))

                    if expected_size > 0 and part.size_bytes == expected_size:
                        part.download_complete = True
                        verified.append(part)
                        logger.debug(
                            f"✅ Part {part.index} verified: {part.size_bytes:,} bytes"
                        )
                    elif part.size_bytes >= expected_size * 0.99:
                        # Allow 1% tolerance for potential compression differences
                        part.download_complete = True
                        verified.append(part)
                        logger.debug(
                            f"✅ Part {part.index} verified (within 1%): "
                            f"{part.size_bytes:,}/{expected_size:,} bytes"
                        )
                    else:
                        logger.info(
                            f"📥 Part {part.index} incomplete: "
                            f"{part.size_bytes:,}/{expected_size:,} bytes - will resume"
                        )
                        need_download.append(part)
                except Exception as e:
                    logger.warning(
                        f"⚠️  Could not verify part {part.index}, will re-check: {e}"
                    )
                    # Still mark as potentially complete - download_file will verify
                    part.download_complete = True
                    verified.append(part)

            already_downloaded = verified
            self._progress.completed_parts = len(already_downloaded)

        else:
            # Trust local file sizes without verification
            for part in already_downloaded:
                part.download_complete = True
            self._progress.completed_parts = len(already_downloaded)

        if already_downloaded:
            total_size = sum(p.size_bytes for p in already_downloaded)
            logger.info(
                f"✅ Found {len(already_downloaded)} existing downloads "
                f"({total_size / (1024**3):.2f} GB)"
            )

        if need_download:
            logger.info(f"📥 {len(need_download)} parts need downloading")

        return already_downloaded, need_download

    def download_file(self, part: DumpFile) -> DumpFile:
        """
        Download a single dump file with resume support.

        Returns the updated DumpFile with download_complete=True on success.
        Thread-safe for parallel execution.
        """
        if shutdown_requested:
            return part

        local_path = part.local_path
        if local_path is None:
            local_path = self.download_dir / part.filename
            part.local_path = local_path

        # Check if already downloaded
        if local_path.exists():
            try:
                head_resp = self.session.head(part.url, timeout=30)
                expected_size = int(head_resp.headers.get("content-length", 0))
                actual_size = local_path.stat().st_size

                if expected_size > 0 and actual_size == expected_size:
                    logger.info(f"✅ Part {part.index} already downloaded")
                    part.download_complete = True
                    part.size_bytes = actual_size
                    self._progress.completed_parts += 1
                    return part
                elif actual_size > 0:
                    logger.info(
                        f"📥 Resuming part {part.index} from {actual_size:,} bytes"
                    )
            except Exception:
                pass

        # Download with resume support
        headers = {}
        mode = "wb"
        start_byte = 0

        if local_path.exists():
            start_byte = local_path.stat().st_size
            headers["Range"] = f"bytes={start_byte}-"
            mode = "ab"

        logger.info(f"📥 Downloading part {part.index}: {part.filename}")
        start_time = time.time()

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
                    self._progress.completed_parts += 1
                    return part

                resp.raise_for_status()

                # Get total size
                if "content-range" in resp.headers:
                    total_size = int(resp.headers["content-range"].split("/")[-1])
                else:
                    total_size = int(resp.headers.get("content-length", 0))
                    total_size += start_byte

                # Download with progress tracking
                downloaded = start_byte
                last_update = time.time()

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
                            self._progress.bytes_downloaded += len(chunk)

                            # Update progress every second
                            now = time.time()
                            if now - last_update > 1.0:
                                if total_size > 0:
                                    progress = downloaded / total_size * 100
                                    self._progress.active_downloads[part.index] = (
                                        progress
                                    )
                                elapsed = now - start_time
                                if elapsed > 0:
                                    speed = (
                                        (downloaded - start_byte)
                                        / elapsed
                                        / 1024
                                        / 1024
                                    )
                                    self._progress.download_speed_mbps = speed
                                last_update = now

                # Verify download
                actual_size = local_path.stat().st_size
                if total_size > 0 and actual_size < total_size:
                    logger.warning(
                        f"⚠️  Part {part.index} incomplete: "
                        f"{actual_size:,}/{total_size:,} bytes"
                    )
                    start_byte = actual_size
                    headers["Range"] = f"bytes={start_byte}-"
                    mode = "ab"
                    retry_count += 1
                    time.sleep(self.RETRY_BACKOFF)
                    continue

                elapsed = time.time() - start_time
                speed = actual_size / elapsed / 1024 / 1024
                logger.info(
                    f"✅ Part {part.index} downloaded: "
                    f"{actual_size / (1024*1024):.1f} MB @ {speed:.1f} MB/s"
                )
                part.download_complete = True
                part.size_bytes = actual_size
                self._progress.completed_parts += 1
                self._progress.active_downloads.pop(part.index, None)
                return part

            except Exception as e:
                retry_count += 1
                logger.warning(
                    f"⚠️  Download error for part {part.index} "
                    f"(attempt {retry_count}/{self.MAX_RETRIES}): {e}"
                )
                if retry_count < self.MAX_RETRIES:
                    time.sleep(self.RETRY_BACKOFF * retry_count)
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
        self._progress.active_downloads[part.index] = 0.0
        return future

    def get_completed_downloads(self) -> list[DumpFile]:
        """Check for and return completed downloads (non-blocking)."""
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

    def get_progress(self) -> DownloadProgress:
        """Get current download progress."""
        return self._progress

    def close(self):
        """Clean up resources."""
        self._download_executor.shutdown(wait=False)
        self.session.close()


# =============================================================================
# LOCAL FILE PARSER - OPTIMIZED
# =============================================================================


class LocalFileParser:
    """
    Parses downloaded Wikipedia dump files from disk.

    Optimized for throughput:
    - Large read buffers
    - Efficient XML parsing with lxml
    - Generator pattern for streaming
    """

    # XML namespace for MediaWiki
    NS = "{http://www.mediawiki.org/xml/export-0.11/}"

    @classmethod
    def parse_file(
        cls,
        file_path: Path,
        skip_ids: set[int] | None = None,
    ) -> Generator[WikiArticle]:
        """
        Parse a local BZ2-compressed Wikipedia dump file.

        Args:
            file_path: Path to the .bz2 dump file
            skip_ids: Set of article IDs to skip (already processed)

        Yields:
            WikiArticle objects for each valid article
        """
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return

        skip_ids = skip_ids or set()
        logger.info(f"📖 Parsing {file_path.name}...")
        articles_yielded = 0
        articles_skipped = 0
        parse_errors = 0

        try:
            # Open with larger buffer for better I/O
            with bz2.open(file_path, "rb") as f:
                context = etree.iterparse(
                    f,
                    events=("end",),
                    tag=f"{cls.NS}page",
                    recover=True,
                    huge_tree=True,
                )

                for _event, elem in context:
                    if shutdown_requested:
                        break

                    try:
                        # Fast ID check first
                        page_id_elem = elem.find(f"{cls.NS}id")
                        if page_id_elem is None or page_id_elem.text is None:
                            continue

                        page_id = int(page_id_elem.text)

                        # Skip if already processed
                        if page_id in skip_ids:
                            articles_skipped += 1
                            continue

                        # Check namespace (only main articles)
                        ns_elem = elem.find(f"{cls.NS}ns")
                        if ns_elem is None or ns_elem.text != "0":
                            continue

                        # Get title
                        title_elem = elem.find(f"{cls.NS}title")
                        if title_elem is None or not title_elem.text:
                            continue
                        title = title_elem.text

                        # Get revision text
                        revision = elem.find(f"{cls.NS}revision")
                        if revision is None:
                            continue

                        text_elem = revision.find(f"{cls.NS}text")
                        if text_elem is None or not text_elem.text:
                            continue
                        text = text_elem.text

                        # Skip redirects and stubs
                        if text.lower().startswith("#redirect") or len(text) < 100:
                            continue

                        safe_title = quote(title.replace(" ", "_"), safe="/:@")
                        yield WikiArticle(
                            id=page_id,
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
                        # Critical: Clear element to free memory
                        elem.clear()
                        # Remove preceding siblings to prevent memory buildup
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
                f"Completed {file_path.name}: {articles_yielded} articles, "
                f"{articles_skipped} skipped, {parse_errors} errors"
            )
        else:
            logger.info(
                f"✅ Parsed {file_path.name}: {articles_yielded} articles, "
                f"{articles_skipped} skipped"
            )
