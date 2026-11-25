"""Wave 3 Stream P — pipeline PCG path integration test.

Wires the pipeline with a stubbed NLI adapter, stubbed cc-NLI
dispatcher, and the real :class:`PCGBeliefPropagationAdapter`. Asserts
that ``/verify``'s internal posterior pipeline drives through PCG and
populates :class:`PCGObservability` per claim.
"""

from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

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

from adapters.pcg_belief_propagation import PCGBeliefPropagationAdapter  # noqa: E402
from interfaces.nli import NliResult  # noqa: E402
from models.domain import DomainAssignment  # noqa: E402
from models.entities import Claim, ClaimType, Evidence, EvidenceSource  # noqa: E402
from pipeline.pipeline import Pipeline  # noqa: E402


class _StubDecomposer:
    async def decompose(self, text: str) -> list[Claim]:  # pragma: no cover
        return []

    async def decompose_with_context(
        self, text: str, context=None, max_claims=None
    ) -> list[Claim]:  # pragma: no cover
        return []

    async def health_check(self) -> bool:  # pragma: no cover
        return True


class _StubConformal:
    async def calibrate(self, **kwargs):  # pragma: no cover
        raise AssertionError("conformal stub should not be called")


@dataclass
class _StubNli:
    reply: NliResult
    calls: list[tuple[str, str]] = field(default_factory=list)

    async def classify(self, claim_text: str, evidence_text: str) -> NliResult:
        self.calls.append((claim_text, evidence_text))
        await asyncio.sleep(0)
        return self.reply

    async def health_check(self) -> bool:  # pragma: no cover
        return True


def _claim(text: str) -> Claim:
    return Claim(id=uuid4(), text=text, claim_type=ClaimType.SUBJECT_PREDICATE_OBJECT)


def _evidence(content: str) -> Evidence:
    return Evidence(source=EvidenceSource.MEDIAWIKI, content=content)


_DEFAULT_ASSIGNMENT = DomainAssignment(
    weights={"general": 1.0},
    primary="general",
    soft=False,
)


def _support() -> NliResult:
    return NliResult(
        label="support",
        supporting_score=0.9,
        refuting_score=0.03,
        neutral_score=0.07,
        reasoning="stub",
        confidence=0.9,
    )


def _refute() -> NliResult:
    return NliResult(
        label="refute",
        supporting_score=0.03,
        refuting_score=0.9,
        neutral_score=0.07,
        reasoning="stub",
        confidence=0.9,
    )


async def test_pcg_path_populates_pcg_observability_on_posteriors_result() -> None:
    """With PCG adapter wired, ``_compute_posteriors`` produces a
    :class:`PCGObservability` block per claim."""
    claim = _claim("Marie Curie won a Nobel prize.")
    evidence = [_evidence("ev-1"), _evidence("ev-2")]
    stub_nli = _StubNli(_support())
    pcg = PCGBeliefPropagationAdapter()
    pipeline = Pipeline(
        decomposer=_StubDecomposer(),  # type: ignore[arg-type]
        retrieval=None,
        conformal=_StubConformal(),  # type: ignore[arg-type]
        nli_adapter=stub_nli,  # type: ignore[arg-type]
        pcg=pcg,
        cc_nli_dispatcher=None,  # single claim — no cc pairs anyway
    )
    result = await pipeline._compute_posteriors(
        [claim], {claim.id: evidence}, {claim.id: _DEFAULT_ASSIGNMENT},
    )
    # Posterior routed through PCG.
    belief = result.posteriors[claim.id]
    assert belief.p_true > 0.5
    # Observability populated.
    obs = result.pcg_observability[claim.id]
    assert obs is not None
    assert obs.algorithm in {"TRW-BP", "LBP-fallback", "LBP-nonconvergent"}
    # Buckets still driven by NLI label (supports in, refutes out).
    supporting, refuting = result.buckets[claim.id]
    assert len(supporting) == 2
    assert len(refuting) == 0


async def test_pcg_path_refute_evidence_drives_p_true_down() -> None:
    """Refute evidence + PCG → p_true below 0.5 (mirrors Beta path)."""
    claim = _claim("Einstein was born in Russia.")
    evidence = [_evidence("ev-1"), _evidence("ev-2")]
    stub_nli = _StubNli(_refute())
    pcg = PCGBeliefPropagationAdapter()
    pipeline = Pipeline(
        decomposer=_StubDecomposer(),  # type: ignore[arg-type]
        retrieval=None,
        conformal=_StubConformal(),  # type: ignore[arg-type]
        nli_adapter=stub_nli,  # type: ignore[arg-type]
        pcg=pcg,
        cc_nli_dispatcher=None,
    )
    result = await pipeline._compute_posteriors(
        [claim], {claim.id: evidence}, {claim.id: _DEFAULT_ASSIGNMENT},
    )
    belief = result.posteriors[claim.id]
    assert belief.p_true < 0.5
    # Refutes filed in the refuting bucket.
    supporting, refuting = result.buckets[claim.id]
    assert len(supporting) == 0
    assert len(refuting) == 2
