"""Strong-normalization helpers for the L1 decomposer. Spec §3 + Task 1.3.

Normalization runs after the LLM has returned raw claim JSON. It cleans up:

  1. **Dates → ISO 8601**. "January 5, 1879", "5 Jan 1879", "1879-01-05"
     all collapse to ``1879-01-05``. Partial dates become the most-specific
     form we can parse ("January 1879" → ``1879-01``).
  2. **Numbers → SI base units**. "5 million" → ``5000000``, "5%" →
     ``0.05``, "3.5 kg" → ``3.5`` (with unit stripped into a separate
     metadata dict if needed).
  3. **Entity QIDs** — a stub at this point. Full Wikidata MCP integration
     lands when the MCP adapter is wired; for now the function returns an
     empty dict so downstream code never has to special-case None.

All three functions are pure and deterministic. No network, no cache,
no MCP in the initial implementation.
"""

from __future__ import annotations

import re
from datetime import date
from re import Match

# ---------------------------------------------------------------------------
# Date normalization
# ---------------------------------------------------------------------------

# Month names (case-insensitive)
_MONTHS = {
    "january": 1,
    "jan": 1,
    "february": 2,
    "feb": 2,
    "march": 3,
    "mar": 3,
    "april": 4,
    "apr": 4,
    "may": 5,
    "june": 6,
    "jun": 6,
    "july": 7,
    "jul": 7,
    "august": 8,
    "aug": 8,
    "september": 9,
    "sep": 9,
    "sept": 9,
    "october": 10,
    "oct": 10,
    "november": 11,
    "nov": 11,
    "december": 12,
    "dec": 12,
}

# Patterns covering common date spellings. Ordered from most-specific to
# least-specific so the first match wins.
_DATE_PATTERNS: list[re.Pattern[str]] = [
    # Already ISO: 1879-01-05, 1879-01, 1879
    re.compile(r"\b(?P<y>\d{4})-(?P<m>\d{2})-(?P<d>\d{2})\b"),
    re.compile(r"\b(?P<y>\d{4})-(?P<m>\d{2})\b"),
    # US: "January 5, 1879" / "Jan. 5, 1879"
    re.compile(
        r"\b(?P<month>[A-Za-z]+)\.?\s+(?P<d>\d{1,2}),?\s+(?P<y>\d{4})\b",
    ),
    # EU: "5 January 1879" / "5 Jan 1879"
    re.compile(
        r"\b(?P<d>\d{1,2})\s+(?P<month>[A-Za-z]+)\.?\s+(?P<y>\d{4})\b",
    ),
    # Month + year only: "January 1879"
    re.compile(r"\b(?P<month>[A-Za-z]+)\.?\s+(?P<y>\d{4})\b"),
    # Year only (4-digit, between 1000 and 2999)
    re.compile(r"\b(?P<y>1\d{3}|2\d{3})\b"),
]


def _match_to_iso(m: Match[str]) -> str | None:
    """Convert a regex match to an ISO 8601 string if possible."""
    groups = m.groupdict()
    y = groups.get("y")
    if not y:
        return None

    month_raw = groups.get("month")
    m_num: int | None = None
    if month_raw:
        m_num = _MONTHS.get(month_raw.lower())
        if m_num is None:
            return None
    elif groups.get("m"):
        try:
            m_num = int(groups["m"])
        except ValueError:
            return None

    d_num: int | None = None
    if groups.get("d"):
        try:
            d_num = int(groups["d"])
        except ValueError:
            return None

    year = int(y)
    if m_num is not None and d_num is not None:
        try:
            return date(year, m_num, d_num).isoformat()
        except ValueError:
            return None
    if m_num is not None:
        if 1 <= m_num <= 12:
            return f"{year:04d}-{m_num:02d}"
        return None
    return f"{year:04d}"


def normalize_date_expression(text: str) -> str | None:
    """Return the ISO 8601 normalization of the first date found in ``text``.

    Returns None if no date-like substring is recognised. Useful when the
    decomposer extracts a temporal claim and we want to canonicalise the
    date for cross-claim equality.
    """
    for pattern in _DATE_PATTERNS:
        for m in pattern.finditer(text):
            iso = _match_to_iso(m)
            if iso:
                return iso
    return None


# ---------------------------------------------------------------------------
# Number normalization
# ---------------------------------------------------------------------------

_NUMBER_SCALES = {
    "thousand": 1_000,
    "k": 1_000,
    "million": 1_000_000,
    "mil": 1_000_000,
    "m": 1_000_000,
    "billion": 1_000_000_000,
    "bn": 1_000_000_000,
    "b": 1_000_000_000,
    "trillion": 1_000_000_000_000,
    "tn": 1_000_000_000_000,
}

# Percent patterns: "5%", "5 percent", "5 per cent"
_PERCENT_RE = re.compile(
    r"(?P<num>-?\d+(?:\.\d+)?)\s*(?:%|per\s*cent|percent)",
    re.IGNORECASE,
)

# Scale patterns: "5 million", "5.2bn", "1k"
_SCALE_RE = re.compile(
    r"(?P<num>-?\d+(?:\.\d+)?)\s*(?P<scale>"
    + "|".join(sorted(_NUMBER_SCALES.keys(), key=len, reverse=True))
    + r")(?![A-Za-z])",
    re.IGNORECASE,
)

# Bare number pattern
_NUMBER_RE = re.compile(r"(?P<num>-?\d+(?:\.\d+)?)")


def normalize_number_expression(text: str) -> float | None:
    """Return the SI-base-unit value of the first numeric expression in ``text``.

    - "5%" / "5 percent" → 0.05
    - "5 million" / "5m" / "5mil" → 5_000_000.0
    - "3.5" → 3.5
    - "no numbers here" → None

    Percent takes precedence over scale (so "5%" is always 0.05, never
    treated as 5 prefix + "%" token).
    """
    m = _PERCENT_RE.search(text)
    if m:
        return float(m.group("num")) / 100.0

    m = _SCALE_RE.search(text)
    if m:
        scale = _NUMBER_SCALES[m.group("scale").lower()]
        return float(m.group("num")) * scale

    m = _NUMBER_RE.search(text)
    if m:
        return float(m.group("num"))

    return None


# ---------------------------------------------------------------------------
# Claim text normalization (lowercase + whitespace + punctuation trim)
# ---------------------------------------------------------------------------


def normalize_claim_text(text: str) -> str:
    """Produce a canonical form for equality / dedup comparisons.

    Lowercases, collapses internal whitespace, and strips a single
    trailing punctuation mark. Does NOT alter dates or numbers — those
    get normalized separately and stored in dedicated Claim fields.
    """
    normalized = " ".join(text.lower().split())
    return re.sub(r"[.,;:!?]$", "", normalized)


# ---------------------------------------------------------------------------
# Entity QID lookup (stub until Wikidata MCP is wired)
# ---------------------------------------------------------------------------


async def resolve_entity_qids(entities: list[str]) -> dict[str, str]:
    """Resolve entity surface forms to Wikidata QIDs.

    Phase 1 stub: returns an empty dict. Task 3.x / infrastructure sub-
    project will wire the Wikidata MCP and replace this implementation.

    Using an async function now means the eventual network-backed
    implementation is a drop-in replacement — no signature change.
    """
    del entities  # unused in the stub
    return {}
