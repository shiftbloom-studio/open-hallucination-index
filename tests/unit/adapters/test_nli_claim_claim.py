"""Wave 3 Stream P — ClaimClaimNliDispatcher tests.

Stubbed primary + fallback; asserts fallback-firing, overlap filter,
and hard-cap behaviours.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from uuid import UUID, uuid4

_SRC_API = Path(__file__).resolve().parents[3] / "src" / "api"
if str(_SRC_API) not in sys.path:
    sys.path.insert(0, str(_SRC_API))

_FLAT_PACKAGES = {"adapters", "interfaces", "models", "pipeline", "config", "server", "services"}
_cached_iface = sys.modules.get("interfaces")
_cached_iface_file = getattr(_cached_iface, "__file__", "") or ""
if _cached_iface is None or str(_SRC_API) not in _cached_iface_file:
    for _cached_name in list(sys.modules):
        _root = _cached_name.split(".", 1)[0]
        if _root in _FLAT_PACKAGES:
            del sys.modules[_cached_name]

from adapters.nli_claim_claim import (  # noqa: E402
    ClaimClaimNliDispatcher,
)
from interfaces.nli import NliResult  # noqa: E402
from models.entities import Claim, ClaimType  # noqa: E402


def _claim(text: str, entity_qids: dict[str, str] | None = None) -> Claim:
    return Claim(
        id=uuid4(),
        text=text,
        claim_type=ClaimType.SUBJECT_PREDICATE_OBJECT,
        entity_qids=entity_qids or {},
    )


def _support() -> NliResult:
    return NliResult(
        label="support",
        supporting_score=0.9,
        refuting_score=0.05,
        neutral_score=0.05,
        reasoning="stub",
        confidence=0.9,
    )


def _sentinel() -> NliResult:
    return NliResult(
        label="neutral",
        supporting_score=0.0,
        refuting_score=0.0,
        neutral_score=1.0,
        reasoning="nli_unavailable",
        confidence=0.0,
    )


@dataclass
class _StubAdapter:
    """Stub NliAdapter that records calls and returns a canned result."""

    result: NliResult = field(default_factory=_support)
    calls: list[tuple[str, str]] = field(default_factory=list)

    async def classify(self, claim_text: str, evidence_text: str) -> NliResult:
        self.calls.append((claim_text, evidence_text))
        return self.result

    async def health_check(self) -> bool:
        return True


async def test_dispatcher_single_claim_returns_empty() -> None:
    primary = _StubAdapter()
    fallback = _StubAdapter()
    d = ClaimClaimNliDispatcher(primary=primary, fallback=fallback)
    result = await d.classify_pairs([_claim("a")])
    assert result.distributions == {}
    assert len(primary.calls) == 0
    assert len(fallback.calls) == 0


async def test_dispatcher_primary_success_does_not_call_fallback() -> None:
    primary = _StubAdapter(result=_support())
    fallback = _StubAdapter()
    d = ClaimClaimNliDispatcher(primary=primary, fallback=fallback)
    ca = _claim("a")
    cb = _claim("b")
    result = await d.classify_pairs([ca, cb])
    # One pair — primary called once, fallback not at all.
    assert len(primary.calls) == 1
    assert len(fallback.calls) == 0
    assert result.fallback_fired_count == 0
    # Distribution recorded under canonical (id_a < id_b) key.
    key = (ca.id, cb.id) if ca.id < cb.id else (cb.id, ca.id)
    assert key in result.distributions


async def test_dispatcher_fallback_fires_on_primary_sentinel() -> None:
    primary = _StubAdapter(result=_sentinel())
    fallback = _StubAdapter(result=_support())
    d = ClaimClaimNliDispatcher(primary=primary, fallback=fallback)
    result = await d.classify_pairs([_claim("a"), _claim("b")])
    assert len(primary.calls) == 1
    assert len(fallback.calls) == 1
    assert result.fallback_fired_count == 1


async def test_dispatcher_hard_cap_truncates_pair_set() -> None:
    """With many claims and no entity-overlap data, the hard cap must
    bound the pair count. max_pairs=2 on 4 claims → 6 possible pairs,
    4 truncated."""
    primary = _StubAdapter(result=_support())
    fallback = _StubAdapter()
    d = ClaimClaimNliDispatcher(
        primary=primary, fallback=fallback, claim_claim_max_pairs=2
    )
    claims = [_claim(f"claim-{i}") for i in range(4)]
    result = await d.classify_pairs(claims)
    assert len(primary.calls) == 2
    assert result.truncated_pair_count == 4


async def test_dispatcher_entity_overlap_short_circuit_drops_disjoint_qids() -> None:
    """Two claims with disjoint qids should be dropped from the pair set."""
    primary = _StubAdapter(result=_support())
    fallback = _StubAdapter()
    d = ClaimClaimNliDispatcher(
        primary=primary, fallback=fallback, entity_overlap_threshold=1
    )
    c1 = _claim("Einstein did X", entity_qids={"subject": "Q937"})  # Einstein
    c2 = _claim("Curie did Y", entity_qids={"subject": "Q7186"})  # Curie
    result = await d.classify_pairs([c1, c2])
    # Disjoint QIDs → pair dropped.
    assert len(primary.calls) == 0
    assert len(result.distributions) == 0


async def test_dispatcher_entity_overlap_passes_shared_qid() -> None:
    """Two claims sharing a QID pass the overlap filter."""
    primary = _StubAdapter(result=_support())
    fallback = _StubAdapter()
    d = ClaimClaimNliDispatcher(
        primary=primary, fallback=fallback, entity_overlap_threshold=1
    )
    c1 = _claim("Einstein was born", entity_qids={"subject": "Q937"})
    c2 = _claim("Einstein died", entity_qids={"subject": "Q937"})
    result = await d.classify_pairs([c1, c2])
    assert len(primary.calls) == 1
    assert len(result.distributions) == 1
