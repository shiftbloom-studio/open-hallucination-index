"""
SQL dump parsers for Wikipedia metadata files.

Wikipedia provides pre-computed relational data that is much more complete
and accurate than extracting from article wikitext. These files include:

- pagelinks.sql.gz: Complete link graph between pages
- categorylinks.sql.gz: All category assignments
- geo_tags.sql.gz: Geographic coordinates for pages
- page_props.sql.gz: Page properties including Wikidata IDs
- page.sql.gz: Page metadata (namespace, title, redirects)
- redirect.sql.gz: Redirect mappings
- category.sql.gz: Category metadata

This module provides streaming parsers that can handle files of any size
with constant memory usage.
"""

from __future__ import annotations

import bz2
import gzip
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

logger = logging.getLogger("ingestion.sql_parsers")


# =============================================================================
# DATA CLASSES FOR PARSED RECORDS
# =============================================================================


@dataclass
class PageLink:
    """A link from one page to another."""

    from_page_id: int
    to_namespace: int  # 0 = main namespace (articles)
    to_title: str

    @property
    def is_article_link(self) -> bool:
        return self.to_namespace == 0


@dataclass
class CategoryLink:
    """A page's membership in a category."""

    page_id: int
    category_name: str
    sort_key: str = ""
    timestamp: str = ""
    sort_key_prefix: str = ""
    collation: str = ""
    link_type: str = "page"  # 'page', 'subcat', or 'file'


@dataclass
class GeoTag:
    """Geographic coordinates for a page."""

    page_id: int
    globe: str  # Usually 'earth'
    primary: bool  # Is this the primary location?
    latitude: float
    longitude: float
    dim: int | None = None  # Dimension/size in meters
    type: str | None = None  # 'city', 'country', 'landmark', etc.
    name: str | None = None  # Optional name
    country: str | None = None  # Country code
    region: str | None = None  # Region code


@dataclass
class PageProperty:
    """A property of a page (includes Wikidata ID, disambiguation flag, etc.)."""

    page_id: int
    property_name: str
    property_value: str
    sort_key: float | None = None


@dataclass
class PageMetadata:
    """Metadata about a page."""

    page_id: int
    namespace: int
    title: str
    is_redirect: bool
    is_new: bool = False
    random: float = 0.0
    touched: str = ""
    links_updated: str | None = None
    latest_revision: int = 0
    length: int = 0
    content_model: str = "wikitext"
    lang: str | None = None

    @property
    def is_article(self) -> bool:
        return self.namespace == 0


@dataclass
class Redirect:
    """A redirect from one page to another."""

    from_page_id: int
    to_namespace: int
    to_title: str
    interwiki: str | None = None
    fragment: str | None = None


@dataclass
class CategoryMetadata:
    """Metadata about a category."""

    category_id: int
    name: str
    pages_count: int = 0
    subcats_count: int = 0
    files_count: int = 0


# =============================================================================
# PRELOADED LOOKUP TABLES
# =============================================================================


@dataclass
class WikipediaLookupTables:
    """
    Pre-loaded lookup tables from SQL dumps for enriching article data.

    These tables are loaded once and used to enrich articles during ingestion,
    providing much more complete data than regex extraction from wikitext.
    """

    # Page ID -> Wikidata Q-ID mapping
    wikidata_ids: dict[int, str] = field(default_factory=dict)

    # Page ID -> (latitude, longitude, type, country)
    geo_coordinates: dict[int, tuple[float, float, str | None, str | None]] = field(
        default_factory=dict
    )

    # Page ID -> set of category names
    categories: dict[int, set[str]] = field(default_factory=dict)

    # Page ID -> set of linked page titles
    links: dict[int, set[str]] = field(default_factory=dict)

    # Redirect: from_title -> to_title (normalized)
    redirects: dict[str, str] = field(default_factory=dict)

    # Page title -> page ID mapping
    title_to_id: dict[str, int] = field(default_factory=dict)

    # Page ID -> page metadata
    page_metadata: dict[int, PageMetadata] = field(default_factory=dict)

    # Category name -> (pages_count, subcats_count)
    category_stats: dict[str, tuple[int, int]] = field(default_factory=dict)

    # Page ID -> is disambiguation page
    disambiguation_pages: set[int] = field(default_factory=set)

    # Statistics
    total_links_loaded: int = 0
    total_categories_loaded: int = 0
    total_geo_loaded: int = 0
    total_wikidata_loaded: int = 0


# =============================================================================
# SQL VALUE PARSER
# =============================================================================


class SQLValueParser:
    """
    Parser for MySQL INSERT statement values.

    Handles the complex escaping and quoting in MySQL dumps.
    """

    # Regex to find INSERT statements
    INSERT_PATTERN = re.compile(
        r"INSERT\s+INTO\s+`?(\w+)`?\s+VALUES\s*", re.IGNORECASE
    )

    # Regex to split tuples (handles nested parentheses and quoted strings)
    TUPLE_PATTERN = re.compile(r"\((?:[^()']|'(?:[^'\\]|\\.)*')*\)")

    @staticmethod
    def parse_value(value: str) -> str | int | float | None:
        """Parse a single MySQL value, handling escaping."""
        value = value.strip()

        if value == "NULL":
            return None
        if value == "''":
            return ""

        # Integer
        if value.isdigit() or (value.startswith("-") and value[1:].isdigit()):
            return int(value)

        # Float
        try:
            if "." in value or "e" in value.lower():
                return float(value)
        except ValueError:
            pass

        # Quoted string
        if value.startswith("'") and value.endswith("'"):
            # Unescape MySQL string
            inner = value[1:-1]
            # Handle escape sequences
            inner = inner.replace("\\'", "'")
            inner = inner.replace("\\\\", "\\")
            inner = inner.replace("\\n", "\n")
            inner = inner.replace("\\r", "\r")
            inner = inner.replace("\\t", "\t")
            inner = inner.replace('\\"', '"')
            return inner

        return value

    @classmethod
    def parse_tuple(cls, tuple_str: str) -> list:
        """Parse a single MySQL tuple like (1,'foo',2.5,NULL)."""
        # Remove outer parentheses
        inner = tuple_str[1:-1].strip()
        if not inner:
            return []

        values = []
        current: list[str] = []
        in_string = False
        escape_next = False
        i = 0

        while i < len(inner):
            char = inner[i]

            if escape_next:
                current.append(char)
                escape_next = False
                i += 1
                continue

            if char == "\\":
                current.append(char)
                escape_next = True
                i += 1
                continue

            if char == "'" and not escape_next:
                in_string = not in_string
                current.append(char)
                i += 1
                continue

            if char == "," and not in_string:
                values.append(cls.parse_value("".join(current)))
                current = []
                i += 1
                continue

            current.append(char)
            i += 1

        # Don't forget the last value
        if current:
            values.append(cls.parse_value("".join(current)))

        return values


# =============================================================================
# STREAMING PARSERS
# =============================================================================


def _open_compressed(path: Path) -> Iterator[str]:
    """Open a compressed file and yield lines."""
    suffix = path.suffix.lower()

    if suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
            yield from f
    elif suffix == ".bz2":
        with bz2.open(path, "rt", encoding="utf-8", errors="replace") as f:
            yield from f
    else:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            yield from f


def _stream_tuples(path: Path) -> Iterator[list]:
    """Stream parsed tuples from a SQL dump file."""
    parser = SQLValueParser()
    buffer = ""

    for line in _open_compressed(path):
        # Skip non-INSERT lines
        if not line.startswith("INSERT INTO"):
            # But check if we're continuing a previous INSERT
            if buffer and line.strip():
                buffer += line
            continue

        # Start of INSERT statement
        buffer = line

        # Find all tuples in the line
        for match in parser.TUPLE_PATTERN.finditer(buffer):
            try:
                values = parser.parse_tuple(match.group(0))
                if values:
                    yield values
            except Exception as e:
                logger.debug(f"Failed to parse tuple: {e}")
                continue

        buffer = ""


def parse_pagelinks(path: Path) -> Iterator[PageLink]:
    """
    Parse enwiki-latest-pagelinks.sql.gz

    Schema: (pl_from, pl_target_id, pl_from_namespace)
    Note: Newer dumps use pl_target_id which references linktarget table
    """
    logger.info(f"Parsing pagelinks from {path}")
    count = 0

    for values in _stream_tuples(path):
        try:
            # Handle both old and new schema
            if len(values) >= 3:
                from_page = int(values[0])
                # New schema uses target_id, old uses namespace + title
                if isinstance(values[1], int):
                    # New schema - need linktarget lookup
                    # For now, skip these as they require join
                    continue
                else:
                    namespace = int(values[1])
                    title = str(values[2])
                    yield PageLink(
                        from_page_id=from_page,
                        to_namespace=namespace,
                        to_title=title.replace("_", " "),
                    )
                    count += 1
                    if count % 5_000_000 == 0:
                        logger.info(f"  Parsed {count:,} pagelinks...")
        except (ValueError, IndexError) as e:
            logger.debug(f"Skipping invalid pagelink: {e}")
            continue

    logger.info(f"Parsed {count:,} pagelinks total")


def parse_categorylinks(path: Path) -> Iterator[CategoryLink]:
    """
    Parse enwiki-latest-categorylinks.sql.gz

    Schema: (cl_from, cl_to, cl_sortkey, cl_timestamp, cl_sortkey_prefix,
             cl_collation, cl_type)
    """
    logger.info(f"Parsing categorylinks from {path}")
    count = 0

    for values in _stream_tuples(path):
        try:
            if len(values) >= 7:
                yield CategoryLink(
                    page_id=int(values[0]),
                    category_name=str(values[1]).replace("_", " "),
                    sort_key=str(values[2]) if values[2] else "",
                    timestamp=str(values[3]) if values[3] else "",
                    sort_key_prefix=str(values[4]) if values[4] else "",
                    collation=str(values[5]) if values[5] else "",
                    link_type=str(values[6]) if values[6] else "page",
                )
                count += 1
                if count % 5_000_000 == 0:
                    logger.info(f"  Parsed {count:,} categorylinks...")
        except (ValueError, IndexError) as e:
            logger.debug(f"Skipping invalid categorylink: {e}")
            continue

    logger.info(f"Parsed {count:,} categorylinks total")


def parse_geo_tags(path: Path) -> Iterator[GeoTag]:
    """
    Parse enwiki-latest-geo_tags.sql.gz

    Schema: (gt_id, gt_page_id, gt_globe, gt_primary, gt_lat, gt_lon,
             gt_dim, gt_type, gt_name, gt_country, gt_region)
    """
    logger.info(f"Parsing geo_tags from {path}")
    count = 0

    for values in _stream_tuples(path):
        try:
            if len(values) >= 6:
                yield GeoTag(
                    page_id=int(values[1]),
                    globe=str(values[2]) if values[2] else "earth",
                    primary=bool(values[3]),
                    latitude=float(values[4]),
                    longitude=float(values[5]),
                    dim=int(values[6]) if len(values) > 6 and values[6] else None,
                    type=str(values[7]) if len(values) > 7 and values[7] else None,
                    name=str(values[8]) if len(values) > 8 and values[8] else None,
                    country=str(values[9]) if len(values) > 9 and values[9] else None,
                    region=str(values[10]) if len(values) > 10 and values[10] else None,
                )
                count += 1
                if count % 500_000 == 0:
                    logger.info(f"  Parsed {count:,} geo_tags...")
        except (ValueError, IndexError) as e:
            logger.debug(f"Skipping invalid geo_tag: {e}")
            continue

    logger.info(f"Parsed {count:,} geo_tags total")


def parse_page_props(path: Path) -> Iterator[PageProperty]:
    """
    Parse enwiki-latest-page_props.sql.gz

    Schema: (pp_page, pp_propname, pp_value, pp_sortkey)

    Key properties:
    - 'wikibase_item': Wikidata Q-ID (e.g., 'Q42')
    - 'disambiguation': Flag for disambiguation pages
    - 'displaytitle': Display title
    - 'page_image_free': Main image
    """
    logger.info(f"Parsing page_props from {path}")
    count = 0

    for values in _stream_tuples(path):
        try:
            if len(values) >= 3:
                yield PageProperty(
                    page_id=int(values[0]),
                    property_name=str(values[1]),
                    property_value=str(values[2]) if values[2] else "",
                    sort_key=float(values[3]) if len(values) > 3 and values[3] else None,
                )
                count += 1
                if count % 5_000_000 == 0:
                    logger.info(f"  Parsed {count:,} page_props...")
        except (ValueError, IndexError) as e:
            logger.debug(f"Skipping invalid page_prop: {e}")
            continue

    logger.info(f"Parsed {count:,} page_props total")


def parse_page(path: Path) -> Iterator[PageMetadata]:
    """
    Parse enwiki-latest-page.sql.gz

    Schema: (page_id, page_namespace, page_title, page_is_redirect, page_is_new,
             page_random, page_touched, page_links_updated, page_latest,
             page_len, page_content_model, page_lang)
    """
    logger.info(f"Parsing page metadata from {path}")
    count = 0

    for values in _stream_tuples(path):
        try:
            if len(values) >= 10:
                yield PageMetadata(
                    page_id=int(values[0]),
                    namespace=int(values[1]),
                    title=str(values[2]).replace("_", " "),
                    is_redirect=bool(values[3]),
                    is_new=bool(values[4]) if len(values) > 4 else False,
                    random=float(values[5]) if len(values) > 5 and values[5] else 0.0,
                    touched=str(values[6]) if len(values) > 6 and values[6] else "",
                    links_updated=str(values[7]) if len(values) > 7 and values[7] else None,
                    latest_revision=int(values[8]) if len(values) > 8 and values[8] else 0,
                    length=int(values[9]) if len(values) > 9 and values[9] else 0,
                    content_model=str(values[10]) if len(values) > 10 and values[10] else "wikitext",
                    lang=str(values[11]) if len(values) > 11 and values[11] else None,
                )
                count += 1
                if count % 2_000_000 == 0:
                    logger.info(f"  Parsed {count:,} page records...")
        except (ValueError, IndexError) as e:
            logger.debug(f"Skipping invalid page record: {e}")
            continue

    logger.info(f"Parsed {count:,} page records total")


def parse_redirect(path: Path) -> Iterator[Redirect]:
    """
    Parse enwiki-latest-redirect.sql.gz

    Schema: (rd_from, rd_namespace, rd_title, rd_interwiki, rd_fragment)
    """
    logger.info(f"Parsing redirects from {path}")
    count = 0

    for values in _stream_tuples(path):
        try:
            if len(values) >= 3:
                yield Redirect(
                    from_page_id=int(values[0]),
                    to_namespace=int(values[1]),
                    to_title=str(values[2]).replace("_", " "),
                    interwiki=str(values[3]) if len(values) > 3 and values[3] else None,
                    fragment=str(values[4]) if len(values) > 4 and values[4] else None,
                )
                count += 1
                if count % 1_000_000 == 0:
                    logger.info(f"  Parsed {count:,} redirects...")
        except (ValueError, IndexError) as e:
            logger.debug(f"Skipping invalid redirect: {e}")
            continue

    logger.info(f"Parsed {count:,} redirects total")


def parse_category(path: Path) -> Iterator[CategoryMetadata]:
    """
    Parse enwiki-latest-category.sql.gz

    Schema: (cat_id, cat_title, cat_pages, cat_subcats, cat_files)
    """
    logger.info(f"Parsing category metadata from {path}")
    count = 0

    for values in _stream_tuples(path):
        try:
            if len(values) >= 5:
                yield CategoryMetadata(
                    category_id=int(values[0]),
                    name=str(values[1]).replace("_", " "),
                    pages_count=int(values[2]) if values[2] else 0,
                    subcats_count=int(values[3]) if values[3] else 0,
                    files_count=int(values[4]) if values[4] else 0,
                )
                count += 1
                if count % 500_000 == 0:
                    logger.info(f"  Parsed {count:,} categories...")
        except (ValueError, IndexError) as e:
            logger.debug(f"Skipping invalid category: {e}")
            continue

    logger.info(f"Parsed {count:,} categories total")


# =============================================================================
# LOOKUP TABLE LOADERS
# =============================================================================


def load_wikidata_ids(page_props_path: Path) -> dict[int, str]:
    """Load Wikidata IDs from page_props.sql.gz."""
    logger.info("Loading Wikidata IDs...")
    wikidata_ids: dict[int, str] = {}

    for prop in parse_page_props(page_props_path):
        if prop.property_name == "wikibase_item" and prop.property_value:
            wikidata_ids[prop.page_id] = prop.property_value

    logger.info(f"Loaded {len(wikidata_ids):,} Wikidata IDs")
    return wikidata_ids


def load_disambiguation_pages(page_props_path: Path) -> set[int]:
    """Load disambiguation page IDs from page_props.sql.gz."""
    logger.info("Loading disambiguation pages...")
    disambig_pages: set[int] = set()

    for prop in parse_page_props(page_props_path):
        if prop.property_name == "disambiguation":
            disambig_pages.add(prop.page_id)

    logger.info(f"Loaded {len(disambig_pages):,} disambiguation pages")
    return disambig_pages


def load_geo_coordinates(
    geo_tags_path: Path, primary_only: bool = True
) -> dict[int, tuple[float, float, str | None, str | None]]:
    """Load geographic coordinates from geo_tags.sql.gz."""
    logger.info("Loading geographic coordinates...")
    geo_coords: dict[int, tuple[float, float, str | None, str | None]] = {}

    for geo in parse_geo_tags(geo_tags_path):
        # Only load primary coordinates (or all if not filtering)
        if primary_only and not geo.primary:
            continue
        # Skip non-Earth coordinates
        if geo.globe != "earth":
            continue
        geo_coords[geo.page_id] = (geo.latitude, geo.longitude, geo.type, geo.country)

    logger.info(f"Loaded {len(geo_coords):,} geographic coordinates")
    return geo_coords


def load_category_assignments(
    categorylinks_path: Path, articles_only: bool = True
) -> dict[int, set[str]]:
    """Load category assignments from categorylinks.sql.gz."""
    logger.info("Loading category assignments...")
    categories: dict[int, set[str]] = {}

    for catlink in parse_categorylinks(categorylinks_path):
        # Only load article->category links (not subcat or file)
        if articles_only and catlink.link_type != "page":
            continue
        if catlink.page_id not in categories:
            categories[catlink.page_id] = set()
        categories[catlink.page_id].add(catlink.category_name)

    logger.info(f"Loaded categories for {len(categories):,} pages")
    return categories


def load_page_metadata(
    page_path: Path, articles_only: bool = True
) -> tuple[dict[int, PageMetadata], dict[str, int]]:
    """Load page metadata and create title->ID mapping."""
    logger.info("Loading page metadata...")
    metadata: dict[int, PageMetadata] = {}
    title_to_id: dict[str, int] = {}

    for page in parse_page(page_path):
        if articles_only and not page.is_article:
            continue
        metadata[page.page_id] = page
        title_to_id[page.title] = page.page_id

    logger.info(f"Loaded metadata for {len(metadata):,} pages")
    return metadata, title_to_id


def load_redirects(
    redirect_path: Path, title_to_id: dict[str, int] | None = None
) -> dict[str, str]:
    """
    Load redirect mappings.

    If title_to_id is provided, will resolve source page IDs to titles.
    Otherwise, returns page_id -> target_title mapping.
    """
    logger.info("Loading redirects...")
    redirects: dict[str, str] = {}

    for redirect in parse_redirect(redirect_path):
        if redirect.to_namespace != 0:  # Only article namespace
            continue

        target_title = redirect.to_title

        # Try to resolve source title if we have the mapping
        if title_to_id:
            # Find source title by page_id (reverse lookup - slow but accurate)
            # For now, just store by page_id
            source_key = str(redirect.from_page_id)
        else:
            source_key = str(redirect.from_page_id)

        redirects[source_key] = target_title

    logger.info(f"Loaded {len(redirects):,} redirects")
    return redirects


def load_category_stats(category_path: Path) -> dict[str, tuple[int, int]]:
    """Load category statistics (page count, subcat count)."""
    logger.info("Loading category statistics...")
    stats: dict[str, tuple[int, int]] = {}

    for cat in parse_category(category_path):
        stats[cat.name] = (cat.pages_count, cat.subcats_count)

    logger.info(f"Loaded stats for {len(stats):,} categories")
    return stats


def load_all_lookup_tables(
    dump_dir: Path,
    load_links: bool = False,  # Very large, optional
) -> WikipediaLookupTables:
    """
    Load all available lookup tables from SQL dumps.

    Args:
        dump_dir: Directory containing SQL dump files
        load_links: Whether to load pagelinks (requires ~10GB+ RAM)

    Returns:
        WikipediaLookupTables with all available data
    """
    tables = WikipediaLookupTables()

    # Check which files exist
    page_path = dump_dir / "enwiki-latest-page.sql.gz"
    page_props_path = dump_dir / "enwiki-latest-page_props.sql.gz"
    geo_tags_path = dump_dir / "enwiki-latest-geo_tags.sql.gz"
    categorylinks_path = dump_dir / "enwiki-latest-categorylinks.sql.gz"
    redirect_path = dump_dir / "enwiki-latest-redirect.sql.gz"
    category_path = dump_dir / "enwiki-latest-category.sql.gz"

    # Load page metadata first (needed for title resolution)
    if page_path.exists():
        tables.page_metadata, tables.title_to_id = load_page_metadata(page_path)

    # Load Wikidata IDs and disambiguation flags
    if page_props_path.exists():
        tables.wikidata_ids = load_wikidata_ids(page_props_path)
        tables.disambiguation_pages = load_disambiguation_pages(page_props_path)
        tables.total_wikidata_loaded = len(tables.wikidata_ids)

    # Load geographic coordinates
    if geo_tags_path.exists():
        tables.geo_coordinates = load_geo_coordinates(geo_tags_path)
        tables.total_geo_loaded = len(tables.geo_coordinates)

    # Load category assignments
    if categorylinks_path.exists():
        tables.categories = load_category_assignments(categorylinks_path)
        tables.total_categories_loaded = sum(len(cats) for cats in tables.categories.values())

    # Load redirects
    if redirect_path.exists():
        tables.redirects = load_redirects(redirect_path, tables.title_to_id)

    # Load category stats
    if category_path.exists():
        tables.category_stats = load_category_stats(category_path)

    logger.info(
        f"Loaded lookup tables: "
        f"{len(tables.wikidata_ids):,} Wikidata IDs, "
        f"{len(tables.geo_coordinates):,} geo coords, "
        f"{len(tables.categories):,} pages with categories"
    )

    return tables
