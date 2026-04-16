"""Unit tests for pipeline.decomposer_normalization."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_SRC_API = Path(__file__).resolve().parents[3] / "src" / "api"
if str(_SRC_API) not in sys.path:
    sys.path.insert(0, str(_SRC_API))

from pipeline.decomposer_normalization import (  # noqa: E402
    normalize_claim_text,
    normalize_date_expression,
    normalize_number_expression,
    resolve_entity_qids,
)


# ---------------------------------------------------------------------------
# normalize_date_expression
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        # Already ISO
        ("1879-03-14", "1879-03-14"),
        # US format with comma
        ("March 14, 1879", "1879-03-14"),
        # US format abbreviated
        ("Mar 14, 1879", "1879-03-14"),
        # EU format
        ("14 March 1879", "1879-03-14"),
        ("14 Mar 1879", "1879-03-14"),
        # Month + year only
        ("March 1879", "1879-03"),
        # Year only
        ("Einstein was born in 1879.", "1879"),
        # Embedded in a sentence
        ("He died on April 18, 1955 in Princeton.", "1955-04-18"),
    ],
)
def test_normalize_date_expression_parametric(text: str, expected: str) -> None:
    assert normalize_date_expression(text) == expected


def test_normalize_date_returns_none_when_no_date_present() -> None:
    assert normalize_date_expression("no dates in this sentence") is None


def test_normalize_date_skips_nonsense_numbers() -> None:
    # "42 million" has numbers but no year-like 4-digit in the 1xxx-2xxx range
    assert normalize_date_expression("42 million dollars") is None


def test_normalize_date_falls_back_to_year_on_invalid_day() -> None:
    # February 30 doesn't exist. Rather than returning None (and losing
    # *all* temporal information), we fall back to the year-only pattern.
    # Consumers who need strict-date semantics can check the length of
    # the ISO string (4 chars = year-only recovery).
    assert normalize_date_expression("February 30, 2020") == "2020"


# ---------------------------------------------------------------------------
# normalize_number_expression
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        # Bare numbers
        ("3.5", 3.5),
        ("42", 42.0),
        ("-7", -7.0),
        # Percent variations
        ("5%", 0.05),
        ("5 percent", 0.05),
        ("5 per cent", 0.05),
        ("12.5%", 0.125),
        # Scale words
        ("5 million", 5_000_000.0),
        ("3.5 billion", 3_500_000_000.0),
        ("2 trillion", 2_000_000_000_000.0),
        # Scale abbreviations
        ("5m", 5_000_000.0),
        ("2bn", 2_000_000_000.0),
        ("1k", 1_000.0),
    ],
)
def test_normalize_number_expression_parametric(text: str, expected: float) -> None:
    assert normalize_number_expression(text) == pytest.approx(expected)


def test_normalize_number_returns_none_when_no_numbers() -> None:
    assert normalize_number_expression("no numbers here") is None


def test_normalize_number_percent_wins_over_scale() -> None:
    # "5%" must be 0.05, never read as just 5 with % as unit
    assert normalize_number_expression("5%") == pytest.approx(0.05)


def test_normalize_number_extracts_first_match() -> None:
    # When multiple numbers present, first wins (deterministic)
    # "5 million dollars per year for 3 years" → 5_000_000 (first match)
    assert normalize_number_expression("5 million dollars per year") == pytest.approx(
        5_000_000.0
    )


# ---------------------------------------------------------------------------
# normalize_claim_text
# ---------------------------------------------------------------------------


def test_normalize_claim_text_lowercases() -> None:
    assert normalize_claim_text("EINSTEIN WAS BORN") == "einstein was born"


def test_normalize_claim_text_collapses_whitespace() -> None:
    assert normalize_claim_text("hello   world\n\nfoo") == "hello world foo"


def test_normalize_claim_text_strips_single_trailing_punct() -> None:
    assert normalize_claim_text("sentence.") == "sentence"
    assert normalize_claim_text("question?") == "question"
    assert normalize_claim_text("yikes!") == "yikes"


def test_normalize_claim_text_is_idempotent() -> None:
    once = normalize_claim_text("Einstein  was BORN  in 1879.")
    twice = normalize_claim_text(once)
    assert once == twice


# ---------------------------------------------------------------------------
# resolve_entity_qids (Phase 1 stub)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resolve_entity_qids_stub_returns_empty_dict() -> None:
    result = await resolve_entity_qids(["Einstein", "Princeton"])
    assert result == {}


@pytest.mark.asyncio
async def test_resolve_entity_qids_stub_empty_input() -> None:
    result = await resolve_entity_qids([])
    assert result == {}
