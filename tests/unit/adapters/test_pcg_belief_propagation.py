"""Wave 3 Stream P — PCGBeliefPropagationAdapter tests.

Behavioural smoke: empty inputs, rigor dispatch, wrap shape.
"""

from __future__ import annotations

import sys
from pathlib import Path
from uuid import uuid4

_SRC_API = Path(__file__).resolve().parents[3] / "src" / "api"
if str(_SRC_API) not in sys.path:
    sys.path.insert(0, str(_SRC_API))

from adapters.pcg_belief_propagation import PCGBeliefPropagationAdapter  # noqa: E402
from models.entities import Claim, ClaimType  # noqa: E402
from models.nli import NLIDistribution  # noqa: E402
from models.pcg import PosteriorBelief  # noqa: E402


def _claim(text: str = "c") -> Claim:
    return Claim(id=uuid4(), text=text, claim_type=ClaimType.SUBJECT_PREDICATE_OBJECT)


def _nli(entail: float, contradict: float) -> NLIDistribution:
    neutral = 1.0 - entail - contradict
    return NLIDistribution(
        entail=entail,
        contradict=contradict,
        neutral=max(neutral, 0.0),
        variance=0.0,
        nli_model_id="stub",
    )


async def test_infer_empty_claims_returns_empty_dict() -> None:
    adapter = PCGBeliefPropagationAdapter()
    result = await adapter.infer(
        claims=[], evidence_per_claim={}, nli_claim_evidence={}, nli_claim_claim={},
        adapter_per_claim={},
    )
    assert result == {}


async def test_infer_single_claim_with_support_evidence_pushes_p_true_high() -> None:
    """Single claim, strong support evidence → p_true > 0.6."""
    c = _claim("Marie Curie won a Nobel prize.")
    nli_ev = {c.id: [_nli(entail=0.92, contradict=0.03), _nli(entail=0.88, contradict=0.05)]}
    adapter = PCGBeliefPropagationAdapter()
    result = await adapter.infer(
        claims=[c], evidence_per_claim={}, nli_claim_evidence=nli_ev,
        nli_claim_claim={}, adapter_per_claim={},
    )
    belief = result[c.id]
    assert isinstance(belief, PosteriorBelief)
    assert belief.p_true > 0.6


async def test_infer_single_claim_with_refute_evidence_pushes_p_true_low() -> None:
    """Single claim, strong refute evidence → p_true < 0.4."""
    c = _claim("Einstein was born in Russia.")
    nli_ev = {c.id: [_nli(entail=0.02, contradict=0.95), _nli(entail=0.05, contradict=0.90)]}
    adapter = PCGBeliefPropagationAdapter()
    result = await adapter.infer(
        claims=[c], evidence_per_claim={}, nli_claim_evidence=nli_ev,
        nli_claim_claim={}, adapter_per_claim={},
    )
    belief = result[c.id]
    assert belief.p_true < 0.4


async def test_infer_fast_rigor_skips_binary_factors() -> None:
    """fast rigor must not incorporate claim-claim NLI."""
    ca = _claim("a")
    cb = _claim("b")
    # Canonical ordering.
    if ca.id > cb.id:
        ca, cb = cb, ca
    # Unary push each toward T slightly.
    nli_ev = {
        ca.id: [_nli(entail=0.6, contradict=0.3)],
        cb.id: [_nli(entail=0.6, contradict=0.3)],
    }
    # Strong SUPPORT edge that in balanced would amplify.
    nli_cc = {(ca.id, cb.id): _nli(entail=0.95, contradict=0.02)}
    adapter = PCGBeliefPropagationAdapter()
    r_balanced = await adapter.infer(
        claims=[ca, cb], evidence_per_claim={}, nli_claim_evidence=nli_ev,
        nli_claim_claim=nli_cc, adapter_per_claim={}, rigor="balanced",
    )
    r_fast = await adapter.infer(
        claims=[ca, cb], evidence_per_claim={}, nli_claim_evidence=nli_ev,
        nli_claim_claim=nli_cc, adapter_per_claim={}, rigor="fast",
    )
    # Fast rigor posteriors carry edge_count=0 (no binary layer).
    assert r_fast[ca.id].edge_count == 0
    # Balanced rigor DID incorporate the edge.
    assert r_balanced[ca.id].edge_count >= 1


async def test_infer_result_marginals_sum_to_one() -> None:
    c1 = _claim("x")
    c2 = _claim("y")
    nli_ev = {
        c1.id: [_nli(entail=0.7, contradict=0.2)],
        c2.id: [_nli(entail=0.3, contradict=0.6)],
    }
    adapter = PCGBeliefPropagationAdapter()
    result = await adapter.infer(
        claims=[c1, c2], evidence_per_claim={}, nli_claim_evidence=nli_ev,
        nli_claim_claim={}, adapter_per_claim={},
    )
    for cid, belief in result.items():
        assert abs(belief.p_true + belief.p_false - 1.0) < 1e-6
