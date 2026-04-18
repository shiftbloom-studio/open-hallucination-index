"""Wave 3 Stream C — entity resolver unit tests.

Stubbed graph + embedding adapters; asserts exact-label short-circuit
beats the vector hop, and that disjoint claims yield empty results.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

_SRC_API = Path(__file__).resolve().parents[3] / "src" / "api"
if str(_SRC_API) not in sys.path:
    sys.path.insert(0, str(_SRC_API))

from adapters.entity_resolver import (  # noqa: E402
    EntityResolver,
    _extract_candidate_spans,
)


@dataclass
class _StubGraph:
    """Stub graph_store with scripted Cypher responses keyed on query substring."""

    scripts: dict[str, list[dict]] = field(default_factory=dict)
    calls: list[tuple[str, dict]] = field(default_factory=list)

    async def run_cypher(self, cypher: str, params: dict):
        self.calls.append((cypher, params))
        for key, result in self.scripts.items():
            if key in cypher:
                return result
        return []


@dataclass
class _StubEmbed:
    vectors: dict[str, list[float]] = field(default_factory=dict)

    async def generate_embedding(self, text: str) -> list[float]:
        return self.vectors.get(text, [0.0] * 384)


def test_extract_candidate_spans_strips_stopwords():
    spans = _extract_candidate_spans("Marie Curie won a Nobel Prize.")
    assert "Marie Curie" in spans
    assert "Nobel Prize" in spans


def test_extract_candidate_spans_empty_on_all_lowercase():
    assert _extract_candidate_spans("the cat sat on the mat") == []


async def test_resolve_exact_match_short_circuits_vector_hop():
    graph = _StubGraph(
        scripts={
            "toLower(e.label) IN $labels": [
                {"qid": "Q7186", "label": "Marie Curie"}
            ]
        }
    )
    embed = _StubEmbed()
    resolver = EntityResolver(graph_store=graph, embedding_adapter=embed)
    out = await resolver.resolve("Marie Curie won two Nobel prizes.")
    assert out
    assert out[0].qid == "Q7186"
    assert out[0].source == "exact"
    # The vector Cypher (db.index.vector.queryNodes) must NOT be called
    # when the exact lookup hits.
    assert not any("vector.queryNodes" in c[0] for c in graph.calls)


async def test_resolve_falls_back_to_vector_when_exact_misses():
    graph = _StubGraph(
        scripts={
            # Empty exact result.
            "toLower(e.label) IN $labels": [],
            # Vector returns one hit.
            "db.index.vector.queryNodes": [
                {"qid": "Q1001", "label": "Some Entity", "similarity": 0.72}
            ],
        }
    )
    embed = _StubEmbed(vectors={"something obscure": [0.1] * 384})
    resolver = EntityResolver(graph_store=graph, embedding_adapter=embed)
    out = await resolver.resolve("something obscure")
    assert out
    assert out[0].qid == "Q1001"
    assert out[0].source == "vector"
    assert abs(out[0].similarity - 0.72) < 1e-6


async def test_resolve_empty_claim_returns_empty():
    graph = _StubGraph()
    embed = _StubEmbed()
    resolver = EntityResolver(graph_store=graph, embedding_adapter=embed)
    assert await resolver.resolve("") == []
    assert await resolver.resolve("   ") == []
