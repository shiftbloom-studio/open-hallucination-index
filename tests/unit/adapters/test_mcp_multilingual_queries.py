"""Unit tests for multilingual MCP query expansion in external adapters."""

from __future__ import annotations

import sys
from pathlib import Path

_SRC_API = Path(__file__).resolve().parents[3] / "src" / "api"
if str(_SRC_API) not in sys.path:
    sys.path.insert(0, str(_SRC_API))

from adapters.mcp_sources.mediawiki import MediaWikiAdapter  # noqa: E402
from adapters.mcp_sources.wikidata import WikidataAdapter  # noqa: E402
from models.entities import Claim  # noqa: E402


def test_mediawiki_build_query_candidates_extracts_entity_when_subject_missing():
    claim = Claim(
        text="Franz Beckenbauer hat die Idee des kommerziellen Freibads erfunden."
    )
    adapter = MediaWikiAdapter()

    candidates = adapter._build_query_candidates(claim)

    assert "Franz Beckenbauer" in candidates
    assert candidates[0] == "Franz Beckenbauer"


def test_mediawiki_build_query_candidates_extracts_subject_when_lowercased():
    claim = Claim(
        text="franz beckenbauer invented the idea of the commercial outdoor pool."
    )
    adapter = MediaWikiAdapter()

    candidates = adapter._build_query_candidates(claim)

    assert "franz beckenbauer" in candidates
    assert candidates[0] == "franz beckenbauer"


def test_mediawiki_build_query_candidates_includes_object_phrase():
    claim = Claim(
        text="Franz Beckenbauer invented the idea of the commercial outdoor swimming pool.",
        subject="Franz Beckenbauer",
        object="commercial outdoor swimming pool",
    )
    adapter = MediaWikiAdapter()

    candidates = adapter._build_query_candidates(claim)

    assert "commercial outdoor swimming pool" in candidates


def test_wikidata_build_search_terms_extracts_entity_when_subject_missing():
    claim = Claim(
        text="Franz Beckenbauer hat die Idee des kommerziellen Freibads erfunden."
    )
    adapter = WikidataAdapter()

    terms = adapter._build_search_terms(claim)

    assert "Franz Beckenbauer" in terms
    assert terms[0] == "Franz Beckenbauer"


def test_wikidata_build_search_terms_extracts_subject_when_lowercased():
    claim = Claim(
        text="franz beckenbauer invented the idea of the commercial outdoor pool."
    )
    adapter = WikidataAdapter()

    terms = adapter._build_search_terms(claim)

    assert "franz beckenbauer" in terms
    assert terms[0] == "franz beckenbauer"


def test_wikidata_build_search_terms_includes_object_phrase():
    claim = Claim(
        text="Franz Beckenbauer invented the idea of the commercial outdoor swimming pool.",
        subject="Franz Beckenbauer",
        object="commercial outdoor swimming pool",
    )
    adapter = WikidataAdapter()

    terms = adapter._build_search_terms(claim)

    assert "commercial outdoor swimming pool" in terms


def test_wikidata_candidate_languages_detects_german_query():
    langs = WikidataAdapter._candidate_languages(
        "Franz Beckenbauer hat die Idee des kommerziellen Freibads erfunden."
    )
    assert langs == ["de", "en"]


def test_wikidata_candidate_languages_defaults_to_english():
    langs = WikidataAdapter._candidate_languages(
        "The Pacific Ocean is the largest ocean."
    )
    assert langs == ["en"]
