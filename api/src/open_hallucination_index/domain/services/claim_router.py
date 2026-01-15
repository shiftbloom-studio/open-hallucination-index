"""
Claim Router - Intelligent Source Selection
============================================

Analyzes claims to determine optimal knowledge source routing.
Maps claim domains to source priorities with expected latency bands.

This enables latency-optimized evidence retrieval by:
1. Classifying claims by domain (medical, academic, news, technical, general)
2. Extracting key entities and concepts
3. Returning prioritized source lists with confidence scores
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from open_hallucination_index.domain.entities import Claim


class ClaimDomain(Enum):
    """Domain classification for claims."""

    MEDICAL = "medical"
    ACADEMIC = "academic"
    NEWS = "news"
    TECHNICAL = "technical"
    ECONOMIC = "economic"
    SECURITY = "security"
    GENERAL = "general"


class SourceTier(Enum):
    """Latency tier for knowledge sources."""

    LOCAL_FAST = "local_fast"  # <20ms (Neo4j, Qdrant)
    MCP_MEDIUM = "mcp_medium"  # 50-200ms (OHI MCP, Wikipedia MCP)
    MCP_SLOW = "mcp_slow"  # 200-500ms+ (External APIs via MCP)


@dataclass
class SourceRecommendation:
    """Recommendation for a knowledge source."""

    source_name: str
    tier: SourceTier
    priority: int  # Lower = higher priority
    relevance_score: float  # 0.0-1.0
    expected_latency_ms: int
    mcp_tool: str | None = None  # For MCP sources, which tool to call


@dataclass
class RoutingDecision:
    """Complete routing decision for a claim."""

    claim_id: str
    domain: ClaimDomain
    confidence: float
    entities: list[str]
    keywords: list[str]
    recommendations: list[SourceRecommendation] = field(default_factory=list)

    @property
    def local_sources(self) -> list[SourceRecommendation]:
        """Get local (fast) sources only."""
        return [r for r in self.recommendations if r.tier == SourceTier.LOCAL_FAST]

    @property
    def mcp_sources(self) -> list[SourceRecommendation]:
        """Get MCP sources only, sorted by priority."""
        return sorted(
            [r for r in self.recommendations if r.tier != SourceTier.LOCAL_FAST],
            key=lambda x: x.priority,
        )

    @property
    def top_mcp_tools(self) -> list[str]:
        """Get top 3 most relevant MCP tools."""
        mcp = self.mcp_sources[:3]
        return [r.mcp_tool for r in mcp if r.mcp_tool]


# Domain detection patterns
MEDICAL_PATTERNS = [
    r"\b(disease|symptom|treatment|drug|medication|patient|clinical|medical|"
    r"diagnosis|therapy|surgery|cancer|diabetes|infection|vaccine|antibio|"
    r"hospital|doctor|physician|nurse|health|syndrome|disorder|chronic|acute|"
    r"dosage|prescription|pharmaceutical|FDA|WHO|CDC|NIH|pubmed)\b",
]

ACADEMIC_PATTERNS = [
    r"\b(research|study|paper|journal|published|author|professor|university|"
    r"experiment|hypothesis|theory|peer-reviewed|citation|academic|scholar|"
    r"thesis|dissertation|conference|proceedings|DOI|arXiv|ORCID)\b",
]

NEWS_PATTERNS = [
    r"\b(announced|reported|said|statement|press|news|today|yesterday|"
    r"breaking|update|according to|sources say|officials|government|"
    r"president|minister|election|protest|crisis|incident)\b",
    r"\b(2024|2025|2026)\b",  # Recent years suggest news
]

TECHNICAL_PATTERNS = [
    r"\b(software|hardware|programming|code|API|library|framework|"
    r"algorithm|database|server|cloud|kubernetes|docker|python|javascript|"
    r"react|node|version|release|bug|feature|github|npm|package|"
    r"vulnerability|CVE|security|exploit|patch)\b",
]

ECONOMIC_PATTERNS = [
    r"\b(GDP|economy|inflation|unemployment|stock|market|trade|"
    r"currency|dollar|euro|bank|finance|investment|growth|recession|"
    r"World Bank|IMF|Federal Reserve|interest rate|fiscal|monetary)\b",
]

SECURITY_PATTERNS = [
    r"\b(vulnerability|CVE|exploit|malware|ransomware|attack|breach|"
    r"hacker|security|authentication|encryption|firewall|zero-day|"
    r"patch|advisory|CVSS|threat|risk)\b",
]


class ClaimRouter:
    """
    Intelligent claim router for optimal source selection.

    Analyzes claims to determine:
    1. Domain classification
    2. Key entities
    3. Recommended sources with priorities
    """

    def __init__(self) -> None:
        """Initialize the router with compiled patterns."""
        self._patterns = {
            ClaimDomain.MEDICAL: [re.compile(p, re.IGNORECASE) for p in MEDICAL_PATTERNS],
            ClaimDomain.ACADEMIC: [re.compile(p, re.IGNORECASE) for p in ACADEMIC_PATTERNS],
            ClaimDomain.NEWS: [re.compile(p, re.IGNORECASE) for p in NEWS_PATTERNS],
            ClaimDomain.TECHNICAL: [re.compile(p, re.IGNORECASE) for p in TECHNICAL_PATTERNS],
            ClaimDomain.ECONOMIC: [re.compile(p, re.IGNORECASE) for p in ECONOMIC_PATTERNS],
            ClaimDomain.SECURITY: [re.compile(p, re.IGNORECASE) for p in SECURITY_PATTERNS],
        }

        # Source definitions with expected latencies
        self._source_definitions = self._build_source_definitions()

    def _build_source_definitions(self) -> dict[str, SourceRecommendation]:
        """Build source recommendation templates."""
        return {
            # Local sources (always included, highest priority)
            "neo4j": SourceRecommendation(
                source_name="neo4j",
                tier=SourceTier.LOCAL_FAST,
                priority=1,
                relevance_score=1.0,
                expected_latency_ms=5,
            ),
            "qdrant": SourceRecommendation(
                source_name="qdrant",
                tier=SourceTier.LOCAL_FAST,
                priority=2,
                relevance_score=1.0,
                expected_latency_ms=10,
            ),
            # MCP sources - Wikipedia/General
            "wikipedia": SourceRecommendation(
                source_name="ohi_wikipedia",
                tier=SourceTier.MCP_MEDIUM,
                priority=10,
                relevance_score=0.8,
                expected_latency_ms=100,
                mcp_tool="search_wikipedia",
            ),
            "wikidata": SourceRecommendation(
                source_name="ohi_wikidata",
                tier=SourceTier.MCP_MEDIUM,
                priority=11,
                relevance_score=0.7,
                expected_latency_ms=120,
                mcp_tool="search_wikidata",
            ),
            "dbpedia": SourceRecommendation(
                source_name="ohi_dbpedia",
                tier=SourceTier.MCP_MEDIUM,
                priority=12,
                relevance_score=0.6,
                expected_latency_ms=150,
                mcp_tool="search_dbpedia",
            ),
            # MCP sources - Academic
            "openalex": SourceRecommendation(
                source_name="ohi_openalex",
                tier=SourceTier.MCP_MEDIUM,
                priority=20,
                relevance_score=0.9,
                expected_latency_ms=150,
                mcp_tool="search_openalex",
            ),
            "crossref": SourceRecommendation(
                source_name="ohi_crossref",
                tier=SourceTier.MCP_MEDIUM,
                priority=21,
                relevance_score=0.85,
                expected_latency_ms=200,
                mcp_tool="search_crossref",
            ),
            "europepmc": SourceRecommendation(
                source_name="ohi_europepmc",
                tier=SourceTier.MCP_MEDIUM,
                priority=22,
                relevance_score=0.7,
                expected_latency_ms=180,
                mcp_tool="search_europepmc",
            ),
            # MCP sources - Medical
            "pubmed": SourceRecommendation(
                source_name="ohi_pubmed",
                tier=SourceTier.MCP_MEDIUM,
                priority=30,
                relevance_score=0.95,
                expected_latency_ms=200,
                mcp_tool="search_pubmed",
            ),
            "clinicaltrials": SourceRecommendation(
                source_name="ohi_clinicaltrials",
                tier=SourceTier.MCP_SLOW,
                priority=31,
                relevance_score=0.8,
                expected_latency_ms=300,
                mcp_tool="search_clinical_trials",
            ),
            # MCP sources - News
            "gdelt": SourceRecommendation(
                source_name="ohi_gdelt",
                tier=SourceTier.MCP_SLOW,
                priority=40,
                relevance_score=0.85,
                expected_latency_ms=400,
                mcp_tool="search_gdelt",
            ),
            # MCP sources - Economic
            "worldbank": SourceRecommendation(
                source_name="ohi_worldbank",
                tier=SourceTier.MCP_MEDIUM,
                priority=50,
                relevance_score=0.9,
                expected_latency_ms=150,
                mcp_tool="get_world_bank_indicator",
            ),
            # MCP sources - Security
            "osv": SourceRecommendation(
                source_name="ohi_osv",
                tier=SourceTier.MCP_MEDIUM,
                priority=60,
                relevance_score=0.95,
                expected_latency_ms=100,
                mcp_tool="search_vulnerabilities",
            ),
            # Context7 - Technical documentation
            "context7": SourceRecommendation(
                source_name="context7",
                tier=SourceTier.MCP_MEDIUM,
                priority=70,
                relevance_score=0.8,
                expected_latency_ms=200,
                mcp_tool="context7_search",
            ),
        }

    def route(self, claim: Claim) -> RoutingDecision:
        """
        Analyze a claim and return routing decision.

        Args:
            claim: The claim to route.

        Returns:
            RoutingDecision with domain, entities, and source recommendations.
        """
        text = claim.text

        # Classify domain
        domain, confidence = self._classify_domain(text)

        # Extract entities
        entities = self._extract_entities(text)

        # Extract keywords
        keywords = self._extract_keywords(text)

        # Build recommendations based on domain
        recommendations = self._build_recommendations(domain, entities, keywords)

        return RoutingDecision(
            claim_id=str(claim.id),
            domain=domain,
            confidence=confidence,
            entities=entities,
            keywords=keywords,
            recommendations=recommendations,
        )

    def _classify_domain(self, text: str) -> tuple[ClaimDomain, float]:
        """Classify claim domain based on pattern matching."""
        scores: dict[ClaimDomain, int] = {}

        for domain, patterns in self._patterns.items():
            score = 0
            for pattern in patterns:
                matches = pattern.findall(text)
                score += len(matches)
            scores[domain] = score

        if not any(scores.values()):
            return ClaimDomain.GENERAL, 0.5

        # Find domain with highest score
        best_domain = max(scores, key=lambda d: scores[d])
        best_score = scores[best_domain]

        # Calculate confidence based on match density
        total_words = len(text.split())
        confidence = min(0.95, 0.5 + (best_score / max(total_words, 1)) * 2)

        return best_domain, confidence

    def _extract_entities(self, text: str) -> list[str]:
        """Extract named entities (capitalized phrases)."""
        # Simple entity extraction: capitalized words/phrases
        words = text.split()
        entities = []
        current_entity = []

        sentence_starters = {
            "the",
            "a",
            "an",
            "this",
            "that",
            "it",
            "is",
            "was",
            "are",
            "were",
        }

        for word in words:
            # Check if word starts with uppercase (potential entity)
            if word and word[0].isupper() and len(word) > 1:
                # Skip common sentence starters
                if word.lower() not in sentence_starters:
                    current_entity.append(word.rstrip(".,;:!?"))
            else:
                if current_entity:
                    entities.append(" ".join(current_entity))
                    current_entity = []

        if current_entity:
            entities.append(" ".join(current_entity))

        # Deduplicate while preserving order
        seen = set()
        unique_entities = []
        for e in entities:
            if e not in seen and len(e) > 1:
                seen.add(e)
                unique_entities.append(e)

        return unique_entities[:10]  # Limit to top 10

    def _extract_keywords(self, text: str) -> list[str]:
        """Extract significant keywords from claim."""
        # Remove common words and extract meaningful terms
        stopwords = {
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "be",
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
            "need",
            "dare",
            "ought",
            "used",
            "to",
            "of",
            "in",
            "for",
            "on",
            "with",
            "at",
            "by",
            "from",
            "as",
            "into",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "between",
            "under",
            "again",
            "further",
            "then",
            "once",
            "here",
            "there",
            "when",
            "where",
            "why",
            "how",
            "all",
            "each",
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
            "just",
            "and",
            "but",
            "or",
            "if",
            "because",
            "until",
            "while",
            "although",
            "though",
            "that",
            "which",
            "who",
            "whom",
            "whose",
            "this",
            "these",
            "those",
            "what",
            "it",
            "its",
        }

        words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
        keywords = [w for w in words if w not in stopwords]

        # Count frequency and return top keywords
        from collections import Counter

        freq = Counter(keywords)
        return [word for word, _ in freq.most_common(10)]

    def _build_recommendations(
        self,
        domain: ClaimDomain,
        entities: list[str],
        keywords: list[str],
    ) -> list[SourceRecommendation]:
        """Build source recommendations based on domain."""
        recommendations: list[SourceRecommendation] = []

        # Always include local sources
        recommendations.append(self._source_definitions["neo4j"])
        recommendations.append(self._source_definitions["qdrant"])

        # Domain-specific MCP sources
        domain_sources = self._get_domain_sources(domain)

        for source_key in domain_sources:
            if source_key in self._source_definitions:
                rec = self._source_definitions[source_key]
                # Adjust relevance based on entity/keyword matches
                adjusted_rec = SourceRecommendation(
                    source_name=rec.source_name,
                    tier=rec.tier,
                    priority=rec.priority,
                    relevance_score=rec.relevance_score,
                    expected_latency_ms=rec.expected_latency_ms,
                    mcp_tool=rec.mcp_tool,
                )
                recommendations.append(adjusted_rec)

        # Always include Wikipedia as fallback
        if "wikipedia" not in domain_sources:
            recommendations.append(self._source_definitions["wikipedia"])

        return sorted(recommendations, key=lambda r: r.priority)

    def _get_domain_sources(self, domain: ClaimDomain) -> list[str]:
        """Get relevant sources for a domain."""
        domain_mapping = {
            ClaimDomain.MEDICAL: ["pubmed", "clinicaltrials", "europepmc", "wikipedia", "wikidata"],
            ClaimDomain.ACADEMIC: ["openalex", "crossref", "europepmc", "wikipedia", "wikidata"],
            ClaimDomain.NEWS: ["gdelt", "wikipedia"],
            ClaimDomain.TECHNICAL: ["osv", "context7", "wikipedia", "wikidata"],
            ClaimDomain.ECONOMIC: ["worldbank", "wikipedia", "wikidata"],
            ClaimDomain.SECURITY: ["osv", "wikipedia"],
            ClaimDomain.GENERAL: ["wikipedia", "wikidata", "dbpedia"],
        }
        return domain_mapping.get(domain, ["wikipedia"])


# Singleton instance
_claim_router: ClaimRouter | None = None


def get_claim_router() -> ClaimRouter:
    """Get or create the singleton claim router."""
    global _claim_router
    if _claim_router is None:
        _claim_router = ClaimRouter()
    return _claim_router
