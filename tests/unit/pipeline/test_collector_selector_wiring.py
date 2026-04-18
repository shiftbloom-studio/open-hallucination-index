"""Regression test for Stream F evidence-retrieval fix.

Pins two wiring invariants of ``AdaptiveEvidenceCollector`` that Stream A's
MediaWiki work exposed only at Lambda runtime:

1. **Tier 2 runs when a selector is wired.** If ``collect(claim,
   mcp_sources=None)`` is called and ``self._mcp_selector`` is set, the
   collector must route through the selector and yield evidence from
   available sources. Without the selector the collector's ``_collect_mcp``
   falls into the ``"No sources and no selector"`` branch and returns
   ``[]``, silently producing empty supporting/refuting evidence in prod.
2. **MCP timeouts are honoured from settings.** The collector's
   hard-coded default ``mcp_timeout_ms=500`` is tighter than MediaWiki's
   round-trip; a large configured timeout must keep the in-flight task
   alive long enough to return.

Both are stubbed end-to-end: no network, no live Gemini, no $.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Put ``src/api`` on the import path so ``from interfaces.*`` / ``from
# pipeline.*`` resolve the same way they do when the Lambda image runs.
_SRC_API = Path(__file__).resolve().parents[3] / "src" / "api"
if str(_SRC_API) not in sys.path:
    sys.path.insert(0, str(_SRC_API))

# Cross-worktree import guard (plan §0, Stream B handover). See the header
# of ``tests/unit/pipeline/test_compute_posteriors.py`` for the full
# rationale. The ``tests/unit/conftest.py`` session-scoped purge already
# handles the first-purge case; this file's guarded re-purge only fires
# when some earlier module walked a stale ``src/api`` path before the
# conftest ran (paranoia for the standalone-run case).
_FLAT_PACKAGES = {
    "adapters",
    "interfaces",
    "models",
    "pipeline",
    "config",
    "server",
    "services",
}
_cached_iface = sys.modules.get("interfaces")
_cached_iface_file = getattr(_cached_iface, "__file__", "") or ""
if _cached_iface is None or str(_SRC_API) not in _cached_iface_file:
    for _cached_name in list(sys.modules):
        _root = _cached_name.split(".", 1)[0]
        if _root in _FLAT_PACKAGES:
            del sys.modules[_cached_name]

import asyncio
from uuid import uuid4

import pytest

from interfaces.mcp import MCPKnowledgeSource
from models.entities import Claim, ClaimType, Evidence, EvidenceSource
from pipeline.retrieval.collector import AdaptiveEvidenceCollector
from pipeline.retrieval.selector import SmartMCPSelector


class _StubMCPSource(MCPKnowledgeSource):
    """Canned MCP source — returns one Evidence per ``find_evidence`` call.

    Instrumented with a call counter so tests can assert the collector
    actually invoked it (rather than short-circuiting into the empty
    branch).
    """

    def __init__(self, canned: Evidence) -> None:
        self._canned = canned
        self._available = True
        self.call_count = 0

    @property
    def source_name(self) -> str:
        return "StubMW"

    @property
    def evidence_source(self) -> EvidenceSource:
        return EvidenceSource.MEDIAWIKI

    @property
    def is_available(self) -> bool:
        return self._available

    async def connect(self) -> None:
        self._available = True

    async def disconnect(self) -> None:
        self._available = False

    async def health_check(self) -> bool:
        return True

    async def find_evidence(self, claim: Claim) -> list[Evidence]:
        self.call_count += 1
        return [self._canned]

    async def search(self, query: str, limit: int = 5) -> list[dict]:
        # Unused by these tests but required by the abstract contract.
        return []


def _make_claim() -> Claim:
    return Claim(
        id=uuid4(),
        text="Marie Curie won two Nobel prizes.",
        subject="Marie Curie",
        predicate="won",
        object="two Nobel prizes",
        claim_type=ClaimType.SUBJECT_PREDICATE_OBJECT,
    )


def _make_evidence() -> Evidence:
    return Evidence(
        source=EvidenceSource.MEDIAWIKI,
        content="Marie Curie (1867-1934) was the first person to win Nobel prizes in two different sciences.",
        similarity_score=0.85,
        source_uri="https://en.wikipedia.org/wiki/Marie_Curie",
    )


@pytest.mark.asyncio
async def test_collector_without_selector_returns_empty_from_mcp_tier() -> None:
    """The pre-fix wiring: no ``mcp_selector``, pipeline passes ``mcp_sources=None``.

    This is the exact failure mode Stream F's fix addresses. With neither
    source argument nor selector, the collector cannot discover MCP
    sources and Tier 2 returns ``[]``. Asserting this here pins the
    contract so a future refactor that silently enables MCP without a
    selector does not regress us back into the "no-selector silent
    failure" state.
    """
    collector = AdaptiveEvidenceCollector(
        graph_store=None,
        vector_store=None,
        mcp_selector=None,  # bug state: mirror pre-fix dependencies.py
        mcp_timeout_ms=10_000.0,
    )
    result = await collector.collect(_make_claim(), mcp_sources=None)
    assert list(result.evidence) == [], (
        "Collector with no selector and no mcp_sources must return empty — "
        "this is the branch Stream F saw misfiring in prod Lambda."
    )


@pytest.mark.asyncio
async def test_collector_with_selector_returns_mcp_evidence() -> None:
    """Post-fix wiring: selector-only path yields evidence.

    Mirrors what ``dependencies.py::_initialize_adapters`` now does:
    build a SmartMCPSelector over the MCP sources and hand it to the
    collector. ``pipeline._retrieve_evidence`` still calls
    ``collect(claim, mcp_sources=None)``; the selector path must kick
    in.
    """
    stub = _StubMCPSource(_make_evidence())
    selector = SmartMCPSelector([stub])
    collector = AdaptiveEvidenceCollector(
        graph_store=None,
        vector_store=None,
        mcp_selector=selector,
        mcp_timeout_ms=10_000.0,
    )

    result = await collector.collect(_make_claim(), mcp_sources=None)

    assert stub.call_count == 1, "selector-fallback must reach the stub source"
    assert len(list(result.evidence)) >= 1, (
        "Selector-only wiring must surface MCP evidence through Tier 2."
    )
    evidence = list(result.evidence)[0]
    assert evidence.source == EvidenceSource.MEDIAWIKI
    assert "Marie Curie" in evidence.content


@pytest.mark.asyncio
async def test_collector_respects_mcp_timeout_from_settings() -> None:
    """Settings-driven timeout must be honoured (bug 2 at the same site).

    The collector's hard-coded default ``mcp_timeout_ms=500`` was
    tighter than MediaWiki's real round-trip. Stream F plumbed
    ``settings.verification.mcp_timeout_ms=10000`` through. This test
    pins the contract: a stub source that sleeps 200 ms must succeed
    when the configured timeout is 2 s, but must not succeed when the
    configured timeout is 50 ms.
    """

    class _SlowStub(_StubMCPSource):
        async def find_evidence(self, claim: Claim) -> list[Evidence]:
            await asyncio.sleep(0.2)
            return await super().find_evidence(claim)

    canned = _make_evidence()

    # With a generous timeout the slow stub returns its evidence.
    slow_ok = _SlowStub(canned)
    selector_ok = SmartMCPSelector([slow_ok])
    collector_ok = AdaptiveEvidenceCollector(
        graph_store=None,
        vector_store=None,
        mcp_selector=selector_ok,
        mcp_timeout_ms=2_000.0,
    )
    r_ok = await collector_ok.collect(_make_claim(), mcp_sources=None)
    assert len(list(r_ok.evidence)) >= 1

    # With a too-tight timeout the same stub is cancelled.
    slow_tight = _SlowStub(canned)
    selector_tight = SmartMCPSelector([slow_tight])
    collector_tight = AdaptiveEvidenceCollector(
        graph_store=None,
        vector_store=None,
        mcp_selector=selector_tight,
        mcp_timeout_ms=50.0,
    )
    r_tight = await collector_tight.collect(_make_claim(), mcp_sources=None)
    assert list(r_tight.evidence) == [], (
        "50ms MCP timeout must cancel a 200ms stub — this pins the "
        "bug-2 reproduction Stream F observed in prod."
    )
