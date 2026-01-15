"""
Advanced text preprocessing for Wikipedia articles.

Features:
- Section extraction with hierarchy
- Infobox parsing for structured data
- Named entity extraction
- BM25 sparse vector tokenization
- Sentence-aware semantic chunking
- Rich metadata extraction for graph relationships
"""

from __future__ import annotations

import hashlib
import html
import logging
import math
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from ingestion.models import (
    ProcessedArticle,
    ProcessedChunk,
    WikiArticle,
    WikiInfobox,
    WikiSection,
)

logger = logging.getLogger("ingestion.preprocessor")


# =============================================================================
# BM25 TOKENIZER FOR SPARSE VECTORS
# =============================================================================


class BM25Tokenizer:
    """
    Lightweight BM25 tokenizer for generating sparse vectors.
    Uses IDF weighting for better retrieval quality.

    Optimized for:
    - Fast tokenization with compiled regex
    - Efficient hash-based vocabulary
    - Thread-safe IDF updates
    """

    # Common English stopwords (expanded list)
    STOPWORDS = frozenset(
        [
            "a",
            "an",
            "the",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "is",
            "it",
            "was",
            "be",
            "are",
            "were",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "must",
            "shall",
            "can",
            "this",
            "that",
            "these",
            "those",
            "i",
            "you",
            "he",
            "she",
            "we",
            "they",
            "what",
            "which",
            "who",
            "whom",
            "when",
            "where",
            "why",
            "how",
            "all",
            "each",
            "every",
            "both",
            "few",
            "more",
            "most",
            "other",
            "some",
            "such",
            "no",
            "nor",
            "not",
            "only",
            "own",
            "same",
            "so",
            "than",
            "too",
            "very",
            "as",
            "if",
            "then",
            "because",
            "while",
            "with",
            "about",
            "against",
            "between",
            "into",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "from",
            "up",
            "down",
            "out",
            "off",
            "over",
            "under",
            "again",
            "further",
            "once",
            "here",
            "there",
            "any",
            "also",
            "just",
            "now",
            "even",
            "still",
            "well",
            "way",
            "use",
            "used",
            "using",
            "one",
            "two",
            "first",
            "new",
            "time",
            "year",
            "years",
            "made",
            "make",
            "many",
            "much",
            "part",
            "see",
            "also",
            "however",
            "its",
            "his",
            "her",
            "their",
            "our",
            "your",
            "my",
        ]
    )

    TOKEN_PATTERN = re.compile(r"\b[a-z0-9]{2,}\b")

    def __init__(
        self,
        vocab_size: int = 50000,
        k1: float = 1.2,
        b: float = 0.75,
        avg_doc_length: int = 500,
    ):
        self.vocab_size = vocab_size
        self.k1 = k1
        self.b = b
        self.avg_doc_length = avg_doc_length
        self.idf_scores: dict[str, float] = {}
        self.doc_count = 0
        self._token_hash_cache: dict[str, int] = {}
        self._cache_limit = 100000

    def _hash_token(self, token: str) -> int:
        """Hash token to vocabulary index for sparse vector."""
        if token in self._token_hash_cache:
            return self._token_hash_cache[token]

        hash_val = int(hashlib.md5(token.encode()).hexdigest(), 16) % self.vocab_size
        if len(self._token_hash_cache) < self._cache_limit:
            self._token_hash_cache[token] = hash_val
        return hash_val

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
            tf = (count * (self.k1 + 1)) / (
                count
                + self.k1 * (1 - self.b + self.b * (doc_len / self.avg_doc_length))
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

    def update_idf(self, documents: list[str]) -> None:
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


@dataclass
class ExtractionResult:
    """Result of text extraction with all metadata."""

    clean_text: str
    links: set[str]
    categories: set[str]
    infobox: WikiInfobox | None
    entities: set[str]
    see_also_links: set[str]
    disambiguation_links: set[str]
    external_links: set[str]
    # Structured data from infobox (existing)
    birth_date: str | None = None
    death_date: str | None = None
    location: str | None = None
    occupation: str | None = None
    nationality: str | None = None
    # NEW: Additional relationship metadata
    spouse: str | None = None
    children: set[str] = field(default_factory=set)
    parents: set[str] = field(default_factory=set)
    education: set[str] = field(default_factory=set)
    employer: set[str] = field(default_factory=set)
    awards: set[str] = field(default_factory=set)
    author_of: set[str] = field(default_factory=set)
    genre: set[str] = field(default_factory=set)
    influenced_by: set[str] = field(default_factory=set)
    influenced: set[str] = field(default_factory=set)
    founded_by: str | None = None
    founding_date: str | None = None
    headquarters: str | None = None
    industry: str | None = None
    country: str | None = None
    capital_of: str | None = None
    part_of: str | None = None
    predecessor: str | None = None
    successor: str | None = None
    instance_of: str | None = None


class AdvancedTextPreprocessor:
    """
    Advanced text processing pipeline with:
    - Section extraction
    - Infobox parsing with structured data
    - Named entity extraction (bold terms)
    - Sentence-aware chunking
    - Rich relationship extraction

    Designed for parallel processing in a thread pool.
    """

    # Regex patterns compiled once for performance
    SECTION_PATTERN = re.compile(r"^(={2,6})\s*(.+?)\s*\1\s*$", re.MULTILINE)
    INFOBOX_START_PATTERN = re.compile(r"\{\{\s*Infobox\b", re.IGNORECASE)
    TEMPLATE_PATTERN = re.compile(r"\{\{[^{}]*\}\}", re.DOTALL)
    NESTED_TEMPLATE_PATTERN = re.compile(
        r"\{\{(?:[^{}]|\{[^{]|\}[^}])*\}\}", re.DOTALL
    )
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

    # Section patterns for relationship extraction
    SEE_ALSO_PATTERN = re.compile(
        r"==\s*See also\s*==\s*(.*?)(?===|\Z)", re.IGNORECASE | re.DOTALL
    )
    DISAMBIGUATION_PATTERN = re.compile(
        r"\{\{disambiguation\}\}|\{\{disambig\}\}", re.IGNORECASE
    )
    EXTERNAL_LINKS_PATTERN = re.compile(r"\[https?://[^\s\]]+(?:\s+[^\]]+)?\]")
    BR_TAG_PATTERN = re.compile(r"<br\s*/?>", re.IGNORECASE)
    LIST_TEMPLATE_PATTERN = re.compile(
        r"\{\{\s*(?:hlist|plainlist|flatlist|ubl|unbulleted\s*list|bulleted\s*list)\b(.*?)\}\}",
        re.IGNORECASE | re.DOTALL,
    )

    # Infobox field patterns for structured extraction
    DATE_PATTERN = re.compile(
        r"\{\{(?:birth date|death date|birth-date|death-date)[^}]*\|(\d{4})[^}]*\}\}",
        re.IGNORECASE,
    )

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        min_chunk_size: int = 100,
        max_workers: int = 8,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="preprocess"
        )

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
                key = self._normalize_infobox_key(key)
                value = value.strip()
                # Clean wiki markup from value
                value = self._normalize_infobox_value(value)
                if key and value and len(key) < 50 and len(value) < 500:
                    if key in properties:
                        if value not in properties[key]:
                            properties[key] = f"{properties[key]} | {value}"
                    else:
                        properties[key] = value

        return WikiInfobox(type=infobox_type, properties=properties)

    def _normalize_infobox_key(self, key: str) -> str:
        """Normalize infobox keys to improve structured extraction."""
        key = key.strip().lower()
        key = re.sub(r"\s+", "_", key)
        key = re.sub(r"[\-/]", "_", key)
        key = re.sub(r"__+", "_", key)
        key = re.sub(r"\d+$", "", key)
        return key.strip("_")

    def _normalize_infobox_value(self, value: str) -> str:
        """Normalize infobox values by stripping templates and markup."""
        value = self.LINK_PATTERN.sub(r"\1", value)
        value = re.sub(r"\[\[|\]\]", "", value)
        value = self.BR_TAG_PATTERN.sub(" | ", value)
        value = re.sub(r"'{2,}", "", value)
        value = re.sub(r"<[^>]+>", "", value)
        value = re.sub(r"\{\{[^}]+\}\}", "", value)
        return value.strip()

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
        for match in self.BOLD_PATTERN.finditer(text[:5000]):
            entity = match.group(1).strip()
            if 2 < len(entity) < 100 and not entity.startswith(("[[", "{{")):
                entities.add(entity)
        return entities

    def extract_see_also(self, text: str) -> set[str]:
        """Extract links from See also section for explicit relationships."""
        see_also_links = set()
        match = self.SEE_ALSO_PATTERN.search(text)
        if match:
            section_content = match.group(1)
            for link_match in self.LINK_PATTERN.finditer(section_content):
                link = link_match.group(1).strip()
                if link and not link.startswith(("File:", "Image:", "Category:")):
                    see_also_links.add(link)
        return see_also_links

    def extract_structured_data(
        self, infobox: WikiInfobox | None
    ) -> dict[str, str | set[str] | None]:
        """Extract structured facts from infobox for knowledge graph (25 fields)."""
        data: dict[str, str | set[str] | None] = {
            # Existing fields
            "birth_date": None,
            "death_date": None,
            "location": None,
            "occupation": None,
            "nationality": None,
            # NEW: 15+ additional fields
            "spouse": None,
            "children": set(),
            "parents": set(),
            "education": set(),
            "employer": set(),
            "awards": set(),
            "author_of": set(),
            "genre": set(),
            "influenced_by": set(),
            "influenced": set(),
            "founded_by": None,
            "founding_date": None,
            "headquarters": None,
            "industry": None,
            "country": None,
            "capital_of": None,
            "part_of": None,
            "predecessor": None,
            "successor": None,
            "instance_of": None,
        }

        if not infobox:
            return data

        props = infobox.properties

        # Birth date
        for key in ["birth_date", "born", "birthdate"]:
            if key in props:
                data["birth_date"] = props[key][:50]
                break

        # Death date
        for key in ["death_date", "died", "deathdate"]:
            if key in props:
                data["death_date"] = props[key][:50]
                break

        # Location
        for key in [
            "location",
            "birth_place",
            "birthplace",
            "city",
        ]:
            if key in props:
                data["location"] = props[key][:100]
                break

        # Occupation
        for key in ["occupation", "profession", "known_for", "field"]:
            if key in props:
                data["occupation"] = props[key][:100]
                break

        # Nationality
        for key in ["nationality", "citizenship"]:
            if key in props:
                data["nationality"] = props[key][:50]
                break

        # NEW: Spouse
        for key in ["spouse", "partner", "spouses"]:
            if key in props:
                data["spouse"] = self._clean_link_text(props[key])[:100]
                break

        # NEW: Children (can be multiple)
        for key in ["children", "child"]:
            if key in props:
                data["children"] = self._extract_list_from_value(props[key])
                break

        # NEW: Parents
        for key in ["parents", "parent", "father", "mother"]:
            if key in props:
                parents_set = data["parents"]
                assert isinstance(parents_set, set)
                parents_set.update(self._extract_list_from_value(props[key]))

        # NEW: Education
        for key in ["education", "alma_mater", "alma mater", "school", "university"]:
            if key in props:
                edu_set = data["education"]
                assert isinstance(edu_set, set)
                edu_set.update(self._extract_list_from_value(props[key]))

        # NEW: Employer
        for key in ["employer", "employers", "organization", "company", "studio", "label"]:
            if key in props:
                emp_set = data["employer"]
                assert isinstance(emp_set, set)
                emp_set.update(self._extract_list_from_value(props[key]))

        # NEW: Awards
        for key in ["awards", "award", "honours", "honors"]:
            if key in props:
                awards_set = data["awards"]
                assert isinstance(awards_set, set)
                awards_set.update(self._extract_list_from_value(props[key]))

        # NEW: Author of / Notable works
        for key in ["notable_works", "works", "bibliography", "discography", "filmography"]:
            if key in props:
                works_set = data["author_of"]
                assert isinstance(works_set, set)
                works_set.update(self._extract_list_from_value(props[key]))

        # NEW: Genre
        for key in ["genre", "genres", "style", "movement"]:
            if key in props:
                genre_set = data["genre"]
                assert isinstance(genre_set, set)
                genre_set.update(self._extract_list_from_value(props[key]))

        # NEW: Influences
        for key in ["influences", "influenced_by"]:
            if key in props:
                inf_by_set = data["influenced_by"]
                assert isinstance(inf_by_set, set)
                inf_by_set.update(self._extract_list_from_value(props[key]))

        for key in ["influenced", "students"]:
            if key in props:
                inf_set = data["influenced"]
                assert isinstance(inf_set, set)
                inf_set.update(self._extract_list_from_value(props[key]))

        # NEW: Founded by
        for key in ["founder", "founders", "founded_by"]:
            if key in props:
                data["founded_by"] = self._clean_link_text(props[key])[:100]
                break

        # NEW: Founding date
        for key in ["founded", "foundation", "established", "formed"]:
            if key in props:
                data["founding_date"] = props[key][:50]
                break

        # NEW: Headquarters
        for key in ["headquarters", "hq_location", "location_city"]:
            if key in props:
                data["headquarters"] = self._clean_link_text(props[key])[:100]
                break

        # NEW: Industry
        for key in ["industry", "industries", "sector"]:
            if key in props:
                data["industry"] = self._clean_link_text(props[key])[:100]
                break

        # NEW: Country
        for key in ["country", "countries", "nation"]:
            if key in props:
                data["country"] = self._clean_link_text(props[key])[:50]
                break

        # NEW: Capital of (for capital cities)
        for key in ["capital_of", "country"]:
            if key in props and infobox.type.lower() in ["settlement", "city"]:
                data["capital_of"] = self._clean_link_text(props[key])[:100]
                break

        # NEW: Part of (geographic hierarchy)
        for key in ["subdivision_name", "state", "province", "region", "district"]:
            if key in props:
                data["part_of"] = self._clean_link_text(props[key])[:100]
                break

        # NEW: Predecessor/Successor
        for key in ["predecessor", "preceded_by"]:
            if key in props:
                data["predecessor"] = self._clean_link_text(props[key])[:100]
                break

        for key in ["successor", "succeeded_by"]:
            if key in props:
                data["successor"] = self._clean_link_text(props[key])[:100]
                break

        # NEW: Instance of (type classification based on infobox type)
        data["instance_of"] = self._classify_instance_type(infobox.type)

        return data

    def _clean_link_text(self, value: str) -> str:
        """Clean wiki link markup from a value, extracting the link text."""
        # Remove [[ ]] and get display text
        cleaned = re.sub(r"\[\[([^|\]]+)(?:\|[^\]]+)?\]\]", r"\1", value)
        # Remove other wiki markup
        cleaned = re.sub(r"'{2,}", "", cleaned)
        cleaned = re.sub(r"<[^>]+>", "", cleaned)
        cleaned = re.sub(r"\{\{[^}]+\}\}", "", cleaned)
        return cleaned.strip()

    def _extract_list_from_value(self, value: str) -> set[str]:
        """Extract multiple items from a comma/newline separated value."""
        items: set[str] = set()
        value = value.strip()

        # Handle common list templates like {{hlist|A|B}} or {{plainlist|* A * B}}
        template_match = self.LIST_TEMPLATE_PATTERN.search(value)
        if template_match:
            template_body = template_match.group(1)
            # Split template arguments on top-level pipes
            parts: list[str] = []
            current: list[str] = []
            depth = 0
            i = 0
            while i < len(template_body):
                two = template_body[i : i + 2]
                if two == "{{":
                    depth += 1
                    current.append(two)
                    i += 2
                    continue
                if two == "}}":
                    depth = max(0, depth - 1)
                    current.append(two)
                    i += 2
                    continue
                if template_body[i] == "|" and depth == 0:
                    parts.append("".join(current).strip())
                    current = []
                    i += 1
                    continue
                current.append(template_body[i])
                i += 1
            if current:
                parts.append("".join(current).strip())

            # Remove template name if present
            if parts and parts[0].lower().startswith(
                ("hlist", "plainlist", "flatlist", "ubl", "unbulleted", "bulleted")
            ):
                parts = parts[1:]

            for part in parts:
                for sub in re.split(r"[\n\r]+|\*", part):
                    cleaned = self._clean_link_text(sub).strip()
                    if cleaned and len(cleaned) > 1 and len(cleaned) < 100:
                        items.add(cleaned)

            if items:
                return items
        # First extract wiki links
        for match in self.LINK_PATTERN.finditer(value):
            link = match.group(1).strip()
            if link and len(link) > 1:
                items.add(link)

        # If no links found, try splitting by common separators
        if not items:
            cleaned = self._clean_link_text(value)
            cleaned = self.BR_TAG_PATTERN.sub("\n", cleaned)
            for sep in [",", "\n", ";", " and "]:
                if sep in cleaned:
                    for part in cleaned.split(sep):
                        part = part.strip()
                        if part and len(part) > 1 and len(part) < 100:
                            items.add(part)
                    break

        return items

    def _classify_instance_type(self, infobox_type: str) -> str | None:
        """Classify the entity type based on infobox type."""
        type_lower = infobox_type.lower()

        type_mappings = {
            "person": ["person", "biography", "officeholder", "scientist", "artist",
                      "musician", "actor", "writer", "athlete", "politician"],
            "organization": ["company", "organization", "university", "school",
                           "hospital", "government agency"],
            "place": ["settlement", "city", "country", "region", "building",
                     "venue", "station", "airport"],
            "creative_work": ["album", "book", "film", "television", "video game",
                            "software", "newspaper", "magazine"],
            "event": ["event", "election", "war", "battle", "competition"],
            "product": ["automobile", "aircraft", "ship", "weapon"],
        }

        for instance_type, keywords in type_mappings.items():
            if any(kw in type_lower for kw in keywords):
                return instance_type

        return None

    def clean_and_extract(self, text: str) -> ExtractionResult:
        """
        Comprehensive text cleaning and metadata extraction.
        Returns ExtractionResult with all extracted data.
        """
        if not text:
            return ExtractionResult(
                clean_text="",
                links=set(),
                categories=set(),
                infobox=None,
                entities=set(),
                see_also_links=set(),
                disambiguation_links=set(),
                external_links=set(),
            )

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
                for p in [
                    "file:",
                    "image:",
                    "category:",
                    "help:",
                    "user:",
                    "template:",
                    "wikipedia:",
                    "portal:",
                ]
            ):
                links.add(link_target)

        # Extract infobox
        infobox = self.parse_infobox(raw_text)

        # Extract entities
        entities = self.extract_entities(raw_text)

        # Extract See also links
        see_also_links = self.extract_see_also(raw_text)

        # Check if disambiguation page
        disambiguation_links = set()
        if self.DISAMBIGUATION_PATTERN.search(raw_text):
            # All links on disambiguation pages are disambiguation links
            disambiguation_links = links.copy()

        # Extract external links
        external_links = set()
        for match in self.EXTERNAL_LINKS_PATTERN.finditer(raw_text):
            url = match.group(0)
            if url:
                external_links.add(url[:200])

        # Extract structured data from infobox
        structured = self.extract_structured_data(infobox)

        # Clean text for embedding
        clean_text = self._clean_text(raw_text)

        # Helper to safely get string values
        def get_str(key: str) -> str | None:
            val = structured.get(key)
            return val if isinstance(val, str) else None

        # Helper to safely get set values  
        def get_set(key: str) -> set[str]:
            val = structured.get(key)
            return val if isinstance(val, set) else set()

        return ExtractionResult(
            clean_text=clean_text,
            links=links,
            categories=categories,
            infobox=infobox,
            entities=entities,
            see_also_links=see_also_links,
            disambiguation_links=disambiguation_links,
            external_links=external_links,
            # Existing fields
            birth_date=get_str("birth_date"),
            death_date=get_str("death_date"),
            location=get_str("location"),
            occupation=get_str("occupation"),
            nationality=get_str("nationality"),
            # NEW: 15 additional fields
            spouse=get_str("spouse"),
            children=get_set("children"),
            parents=get_set("parents"),
            education=get_set("education"),
            employer=get_set("employer"),
            awards=get_set("awards"),
            author_of=get_set("author_of"),
            genre=get_set("genre"),
            influenced_by=get_set("influenced_by"),
            influenced=get_set("influenced"),
            founded_by=get_str("founded_by"),
            founding_date=get_str("founding_date"),
            headquarters=get_str("headquarters"),
            industry=get_str("industry"),
            country=get_str("country"),
            capital_of=get_str("capital_of"),
            part_of=get_str("part_of"),
            predecessor=get_str("predecessor"),
            successor=get_str("successor"),
            instance_of=get_str("instance_of"),
        )

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

    def process_article(self, article: WikiArticle) -> ProcessedArticle:
        """
        Process a single article: extract metadata and create chunks.

        This is the main entry point for preprocessing, designed to be
        called in parallel from a thread pool.
        """
        # Extract all metadata
        result = self.clean_and_extract(article.text)

        # Update article with extracted metadata
        article.links = result.links
        article.categories = result.categories
        article.infobox = result.infobox
        article.entities = result.entities
        article.see_also_links = result.see_also_links
        article.disambiguation_links = result.disambiguation_links
        article.external_links = result.external_links
        # Existing fields
        article.birth_date = result.birth_date
        article.death_date = result.death_date
        article.location = result.location
        article.occupation = result.occupation
        article.nationality = result.nationality
        # NEW: 15 additional relationship fields
        article.spouse = result.spouse
        article.children = result.children
        article.parents = result.parents
        article.education = result.education
        article.employer = result.employer
        article.awards = result.awards
        article.author_of = result.author_of
        article.genre = result.genre
        article.influenced_by = result.influenced_by
        article.influenced = result.influenced
        article.founded_by = result.founded_by
        article.founding_date = result.founding_date
        article.headquarters = result.headquarters
        article.industry = result.industry
        article.country = result.country
        article.capital_of = result.capital_of
        article.part_of = result.part_of
        article.predecessor = result.predecessor
        article.successor = result.successor
        article.instance_of = result.instance_of

        # Extract sections
        article.sections = self.extract_sections(article.text)

        # Extract first paragraph
        article.first_paragraph = self.extract_first_paragraph(article.text)

        # Create chunks
        chunks = self.chunk_article(article, result.clean_text)

        return ProcessedArticle(
            article=article,
            chunks=chunks,
            clean_text=result.clean_text,
        )

    def process_batch(self, articles: list[WikiArticle]) -> list[ProcessedArticle]:
        """
        Process a batch of articles in parallel using thread pool.

        Returns list of ProcessedArticle objects.
        """
        results = list(self._executor.map(self.process_article, articles))
        return results

    def close(self):
        """Shutdown the thread pool."""
        self._executor.shutdown(wait=False)
