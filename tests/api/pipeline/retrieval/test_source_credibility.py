"""Tests for pipeline.retrieval.source_credibility.

All tests are deterministic; no infra, no network.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

_SRC_API = Path(__file__).resolve().parents[4] / "src" / "api"
if str(_SRC_API) not in sys.path:
    sys.path.insert(0, str(_SRC_API))

from pipeline.retrieval.source_credibility import (  # noqa: E402
    DEFAULT_PRIORS,
    FALLBACK_CREDIBILITY,
    credibility_for,
    fingerprint,
    temporal_decay,
)


# ---------------------------------------------------------------------------
# credibility_for
# ---------------------------------------------------------------------------


def test_credibility_known_source_uses_default_prior() -> None:
    assert credibility_for("wikipedia_general") == DEFAULT_PRIORS["wikipedia_general"]
    assert credibility_for("peer_reviewed_journal") == 0.95


def test_credibility_unknown_source_falls_back() -> None:
    assert credibility_for("some-random-source") == FALLBACK_CREDIBILITY


def test_credibility_domain_override_wins() -> None:
    overrides = {"pubmed": 0.95, "wikipedia_general": 0.60}
    assert credibility_for("pubmed", domain_overrides=overrides) == 0.95
    # Override takes precedence over the default
    assert credibility_for("wikipedia_general", domain_overrides=overrides) == 0.60


def test_credibility_domain_override_doesnt_affect_others() -> None:
    overrides = {"pubmed": 0.95}
    # pubmed isn't in DEFAULT_PRIORS, so without override it'd fall back
    assert credibility_for("pubmed") == FALLBACK_CREDIBILITY
    assert credibility_for("pubmed", domain_overrides=overrides) == 0.95


def test_credibility_values_all_in_unit_interval() -> None:
    for src, val in DEFAULT_PRIORS.items():
        assert 0.0 <= val <= 1.0, f"{src} has out-of-range credibility {val}"


# ---------------------------------------------------------------------------
# temporal_decay
# ---------------------------------------------------------------------------


def test_temporal_decay_zero_age_is_one() -> None:
    assert temporal_decay(0) == 1.0


def test_temporal_decay_one_half_life_is_half() -> None:
    assert temporal_decay(365, half_life_days=365) == pytest.approx(0.5)


def test_temporal_decay_two_half_lives_is_quarter() -> None:
    assert temporal_decay(730, half_life_days=365) == pytest.approx(0.25)


def test_temporal_decay_negative_age_returns_one() -> None:
    # Guard against clock-skew edge cases where retrieved_at > now
    assert temporal_decay(-5) == 1.0


def test_temporal_decay_respects_custom_half_life() -> None:
    # 30-day half-life → 30d old = 0.5, 60d = 0.25
    assert temporal_decay(30, half_life_days=30) == pytest.approx(0.5)
    assert temporal_decay(60, half_life_days=30) == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# fingerprint
# ---------------------------------------------------------------------------


def test_fingerprint_is_deterministic() -> None:
    a = fingerprint("https://example.com/a", "Hello World")
    b = fingerprint("https://example.com/a", "Hello World")
    assert a == b


def test_fingerprint_normalizes_whitespace_in_content() -> None:
    # Internal-whitespace normalization: "hello  world" == "hello world"
    a = fingerprint("https://example.com/a", "hello world")
    b = fingerprint("https://example.com/a", "hello    world")
    c = fingerprint("https://example.com/a", "hello\nworld")
    assert a == b == c


def test_fingerprint_normalizes_uri_case() -> None:
    a = fingerprint("https://Example.Com/A", "hello")
    b = fingerprint("https://example.com/a", "hello")
    assert a == b


def test_fingerprint_differs_across_different_content() -> None:
    a = fingerprint("https://example.com", "foo")
    b = fingerprint("https://example.com", "bar")
    assert a != b


def test_fingerprint_is_valid_sha256_hex() -> None:
    fp = fingerprint("https://example.com", "payload")
    assert len(fp) == 64  # sha256 hex
    # All hex chars
    assert all(c in "0123456789abcdef" for c in fp)
    # Matches direct sha256 of normalized input
    expected = hashlib.sha256(
        ("https://example.com".lower() + "\n" + "payload").encode("utf-8")
    ).hexdigest()
    assert fp == expected


def test_fingerprint_high_volume_no_collisions() -> None:
    """Smoke test: 1000 distinct URIs → 1000 distinct fingerprints."""
    seen = {
        fingerprint(f"https://example.com/doc-{i}", "same content") for i in range(1000)
    }
    assert len(seen) == 1000
