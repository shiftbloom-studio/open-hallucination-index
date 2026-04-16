"""Source credibility priors + temporal decay + provenance fingerprinting.

These are the three evidence-weight signals the L4 PCG consumes when
building unary potentials (see spec §6):

    w_e = source_credibility(e) · temporal_decay(e) · (1 - NLI_variance)

The collector (Task 1.4) stamps each `Evidence` object it emits with
`source_credibility`, `temporal_decay_factor`, and a stable `fingerprint`
key. Fingerprint also doubles as a cross-path dedup key when the same
document comes in via multiple retrieval adapters (Neo4j + Qdrant + MCP).
"""

from __future__ import annotations

import hashlib

# ---------------------------------------------------------------------------
# Default credibility priors per spec §3. Per-domain adapters override
# specific entries via `DomainAdapter.source_credibility()`.
# ---------------------------------------------------------------------------

DEFAULT_PRIORS: dict[str, float] = {
    "peer_reviewed_journal": 0.95,
    "official_gov_docs": 0.92,
    "wikipedia_featured_article": 0.88,
    "mcp_curated": 0.80,
    "wikipedia_general": 0.78,
    "news_high_repute": 0.75,
    "qdrant_general": 0.70,
    "news_general": 0.65,
    "graph_inferred": 0.60,
}

FALLBACK_CREDIBILITY = 0.50


def credibility_for(
    source: str,
    *,
    domain_overrides: dict[str, float] | None = None,
) -> float:
    """Return credibility prior for a source, respecting a domain override.

    Lookup order: domain_overrides > DEFAULT_PRIORS > FALLBACK_CREDIBILITY.
    """
    if domain_overrides and source in domain_overrides:
        return domain_overrides[source]
    return DEFAULT_PRIORS.get(source, FALLBACK_CREDIBILITY)


def temporal_decay(
    evidence_age_days: float,
    *,
    half_life_days: float = 365.0,
) -> float:
    """Half-life decay factor.

    Returns 1.0 at age 0, 0.5 at one half-life, 0.25 at two. Evidence ages
    past a few years decay to a near-zero floor — useful for news and
    fast-moving topics, essentially a no-op for Wikipedia-grade knowledge.
    """
    if evidence_age_days < 0:
        return 1.0
    return 0.5 ** (evidence_age_days / half_life_days)


def fingerprint(source_uri: str, content: str) -> str:
    """Stable sha256 over normalized (uri + content) for dedup.

    Normalization: lowercase URI, lowercase content, collapse internal
    whitespace. Same document retrieved via two paths gets the same
    fingerprint even if byte-for-byte spacing differs.
    """
    normalized = source_uri.strip().lower() + "\n" + " ".join(content.lower().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
