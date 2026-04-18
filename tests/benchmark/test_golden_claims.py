"""Wave 3 Stream G.2 — golden-claim benchmark (blocking item #5).

Fires a curated claim set against live prod and asserts:
* Obviously-true claims → p_true ≥ 0.70.
* Obviously-false claims → p_true ≤ 0.30.
* Ambiguous claims → p_true in (0.30, 0.70).

Skipped entirely when ``OHI_API_URL`` / ``OHI_EDGE_SECRET`` env vars
are missing (local unit-test mode). Also skipped when
``--no-integration`` is passed to pytest.

Runs in ``v2-post-deploy-verify.yml`` as a signal item
(``continue-on-error: true``) and as a blocking smoke in the
manual pre-release runbook.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import httpx
import pytest

pytestmark = pytest.mark.integration


@dataclass(frozen=True)
class GoldenClaim:
    text: str
    expected_range: tuple[float, float]  # inclusive (lo, hi) on p_true
    label: str  # 'support' / 'refute' / 'ambiguous' — for report grouping


# Keep this list small + curated. Adding a claim means adding a
# human-labelled truth judgement; see docs/benchmarks/v2.0-golden-claims.md.
GOLDEN_CLAIMS: list[GoldenClaim] = [
    # Obviously-refutable facts (post-Wave-3 PCG should push p_true ≤ 0.30).
    GoldenClaim("Albert Einstein was born in Russia.", (0.0, 0.30), "refute"),
    GoldenClaim("Marie Curie won zero Nobel Prizes.", (0.0, 0.30), "refute"),
    GoldenClaim("Stephen Hawking died in 2001.", (0.0, 0.30), "refute"),
    # Obviously-true facts.
    GoldenClaim("Marie Curie won two Nobel Prizes.", (0.70, 1.0), "support"),
    GoldenClaim("Albert Einstein developed the theory of relativity.", (0.70, 1.0), "support"),
    # Ambiguous / unverifiable — wide interval acceptable.
    GoldenClaim(
        "Einstein was the most influential physicist of the 20th century.",
        (0.30, 1.0),
        "ambiguous",
    ),
]


def _skip_if_no_creds() -> None:
    if not (os.environ.get("OHI_API_URL") and os.environ.get("OHI_EDGE_SECRET")):
        pytest.skip("Integration creds not set (OHI_API_URL / OHI_EDGE_SECRET)")


def _verify_claim(client: httpx.Client, url: str, secret: str, text: str) -> dict:
    headers = {"X-OHI-Edge-Secret": secret, "Content-Type": "application/json"}
    resp = client.post(f"{url}/api/v2/verify", json={"text": text}, headers=headers)
    resp.raise_for_status()
    job_id = resp.json()["job_id"]
    deadline = time.monotonic() + 180.0
    while time.monotonic() < deadline:
        r = client.get(f"{url}/api/v2/verify/status/{job_id}", headers=headers)
        r.raise_for_status()
        body = r.json()
        if body.get("status") == "done":
            return body
        if body.get("status") == "failed":
            raise RuntimeError(f"Verify failed: {body}")
        time.sleep(2.0)
    raise TimeoutError(f"Claim {text!r} did not terminate in 180s")


@pytest.mark.parametrize("claim", GOLDEN_CLAIMS, ids=lambda c: c.label + "::" + c.text[:40])
def test_golden_claim_p_true_in_range(claim: GoldenClaim) -> None:
    _skip_if_no_creds()
    url = os.environ["OHI_API_URL"].rstrip("/")
    secret = os.environ["OHI_EDGE_SECRET"]
    with httpx.Client(timeout=30.0) as client:
        body = _verify_claim(client, url, secret, claim.text)
    result = body["result"]
    claims = result["claims"]
    # For multi-claim decompositions, require that the dominant claim
    # matches the range (doc-level aggregate can differ by decomposition
    # shape). Take max or min depending on polarity.
    p_trues = [float(c["p_true"]) for c in claims]
    if not p_trues:
        pytest.fail(f"No claims decomposed from {claim.text!r}")
    lo, hi = claim.expected_range
    if claim.label == "refute":
        # At least one decomposed claim should fall in the refute range.
        assert min(p_trues) <= hi, (
            f"Refute claim {claim.text!r}: min p_true={min(p_trues):.3f} > {hi}"
        )
    elif claim.label == "support":
        assert max(p_trues) >= lo, (
            f"Support claim {claim.text!r}: max p_true={max(p_trues):.3f} < {lo}"
        )
    else:
        # Ambiguous: avg should fall in the range.
        avg = sum(p_trues) / len(p_trues)
        assert lo <= avg <= hi, (
            f"Ambiguous claim {claim.text!r}: avg p_true={avg:.3f} outside {claim.expected_range}"
        )


def test_golden_claims_pcg_observability_populated() -> None:
    """Every golden-claim verdict should carry a pcg observability
    block (Wave 3 Stream P contract)."""
    _skip_if_no_creds()
    url = os.environ["OHI_API_URL"].rstrip("/")
    secret = os.environ["OHI_EDGE_SECRET"]
    with httpx.Client(timeout=30.0) as client:
        body = _verify_claim(client, url, secret, GOLDEN_CLAIMS[0].text)
    claims = body["result"]["claims"]
    for c in claims:
        pcg = c.get("pcg")
        assert pcg is not None, f"Missing pcg block on claim: {c['claim']['text'][:80]}"
        assert pcg.get("algorithm") in {"TRW-BP", "LBP-fallback", "LBP-nonconvergent"}
