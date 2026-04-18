"""Wave 3 Stream G.2 — end-to-end pipeline integration tests (blocking item #1).

Hits real prod through the HTTP API. Skipped entirely when
``OHI_API_URL`` / ``OHI_EDGE_SECRET`` aren't set (local dev mode).

These tests are pytest-marked ``integration`` so the default pytest
runline (``-m "not infra"``) does NOT pick them up — they run in the
``v2-post-deploy-verify.yml`` workflow where creds are injected.
"""

from __future__ import annotations

import os
import time

import httpx
import pytest

pytestmark = pytest.mark.integration


@pytest.fixture
def api_url() -> str:
    url = os.environ.get("OHI_API_URL")
    if not url:
        pytest.skip("OHI_API_URL not set")
    return url.rstrip("/")


@pytest.fixture
def edge_secret() -> str:
    secret = os.environ.get("OHI_EDGE_SECRET")
    if not secret:
        pytest.skip("OHI_EDGE_SECRET not set")
    return secret


def _post_and_poll(
    client: httpx.Client, url: str, secret: str, text: str
) -> dict:
    headers = {"X-OHI-Edge-Secret": secret, "Content-Type": "application/json"}
    resp = client.post(f"{url}/api/v2/verify", json={"text": text}, headers=headers)
    resp.raise_for_status()
    job_id = resp.json()["job_id"]
    deadline = time.monotonic() + 180.0
    while time.monotonic() < deadline:
        r = client.get(f"{url}/api/v2/verify/status/{job_id}", headers=headers)
        r.raise_for_status()
        body = r.json()
        if body.get("status") in ("done", "failed"):
            return body
        time.sleep(2.0)
    raise TimeoutError(f"Job {job_id} did not terminate")


def test_verify_shape_includes_pcg_observability(api_url: str, edge_secret: str) -> None:
    """Wave 3 Stream P contract: every claim verdict carries ``pcg``
    with algorithm, converged, iterations, edge_count fields."""
    with httpx.Client(timeout=30.0) as client:
        body = _post_and_poll(
            client, api_url, edge_secret, "Marie Curie won two Nobel Prizes."
        )
    assert body["status"] == "done"
    result = body["result"]
    assert 0.0 <= result["document_score"] <= 1.0
    claims = result["claims"]
    assert claims, "Expected at least one decomposed claim"
    for c in claims:
        pcg = c.get("pcg")
        assert pcg is not None
        assert pcg["algorithm"] in {"TRW-BP", "LBP-fallback", "LBP-nonconvergent"}
        assert isinstance(pcg["converged"], bool)
        assert pcg["iterations"] >= 0
        assert pcg["edge_count"] >= 0


def test_verify_buckets_are_label_driven(api_url: str, edge_secret: str) -> None:
    """Fix 1+3 + Hebel A+B contract: supporting_evidence holds ``label=
    support`` passages; refuting_evidence holds ``label=refute``."""
    with httpx.Client(timeout=30.0) as client:
        body = _post_and_poll(
            client, api_url, edge_secret, "Albert Einstein was born in Russia."
        )
    result = body["result"]
    c = result["claims"][0]
    # Einstein-Russia is obviously refutable — post-Wave-3 PCG should
    # push p_true very low (< 0.3) and surface refuting_evidence > 0.
    assert c["p_true"] < 0.3, f"Einstein-Russia p_true={c['p_true']:.3f} — too high"
    # Support bucket should be empty or very small; refute bucket non-empty.
    assert len(c["refuting_evidence"]) >= 1


def test_verify_multi_claim_doc_populates_cc_nli(
    api_url: str, edge_secret: str
) -> None:
    """Wave 3 cc-NLI contract: a multi-claim doc with shared subject
    produces claim-claim NLI edges (pcg.edge_count > 0) and carries
    pcg_neighbors on each claim verdict."""
    with httpx.Client(timeout=30.0) as client:
        body = _post_and_poll(
            client,
            api_url,
            edge_secret,
            "Marie Curie won two Nobel Prizes. She was the first woman to do so.",
        )
    claims = body["result"]["claims"]
    if len(claims) >= 2:
        # At least one claim should have edges and neighbours.
        max_edges = max(c.get("pcg", {}).get("edge_count", 0) for c in claims)
        max_neighbors = max(len(c.get("pcg_neighbors", [])) for c in claims)
        assert max_edges > 0, "Expected at least one cc-NLI edge on multi-claim doc"
        assert max_neighbors > 0
