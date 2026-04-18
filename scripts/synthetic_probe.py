"""Wave 3 Stream G.1 — synthetic probe.

Periodically POSTs a known good claim to ``/api/v2/verify``, polls
until terminal, and asserts the verdict shape + PCG observability.
Designed to run for a bounded duration with a configurable interval;
exits with code 0 on all-green, non-zero on any failure.

Used by:
* ``v2-post-deploy-verify.yml`` (short 10-min window after deploy,
  one of the blocking items).
* ``v2-post-merge-probe.yml`` (30-min window, non-blocking).
* Manual invocation from a runbook when investigating production
  behaviour.

Reads configuration from env:
* ``OHI_API_URL``       — base URL (default ``https://ohi-api.shiftbloom.studio``)
* ``OHI_EDGE_SECRET``   — X-OHI-Edge-Secret header value (required)
* ``PROBE_CLAIM_TEXT``  — override the default Marie-Curie claim text

CLI:
* ``--duration``  total seconds to run (default 600 = 10 min)
* ``--interval``  seconds between probes (default 60)
* ``--max-failures``  early-exit threshold (default 3 consecutive)
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import httpx


DEFAULT_CLAIM = (
    "Marie Curie won two Nobel Prizes, the first woman to do so."
)


def _probe_once(
    client: httpx.Client,
    api_url: str,
    edge_secret: str,
    claim_text: str,
) -> tuple[bool, str]:
    """Run one probe. Returns ``(ok, message)``."""
    headers = {"X-OHI-Edge-Secret": edge_secret, "Content-Type": "application/json"}
    try:
        resp = client.post(
            f"{api_url}/api/v2/verify",
            headers=headers,
            json={"text": claim_text},
            timeout=10.0,
        )
        if resp.status_code != 202:
            return False, f"POST status={resp.status_code} body={resp.text[:200]}"
        job_id = resp.json()["job_id"]
    except Exception as exc:  # noqa: BLE001
        return False, f"POST exc={exc}"

    # Poll until terminal.
    deadline = time.monotonic() + 120.0
    while time.monotonic() < deadline:
        try:
            resp = client.get(
                f"{api_url}/api/v2/verify/status/{job_id}",
                headers=headers,
                timeout=10.0,
            )
            resp.raise_for_status()
            state = resp.json().get("status", "?")
            if state == "done":
                body = resp.json()
                result = body.get("result") or {}
                doc_score = result.get("document_score")
                if doc_score is None:
                    return False, f"done but missing document_score: {body}"
                if not 0.0 <= float(doc_score) <= 1.0:
                    return False, f"document_score out of range: {doc_score}"
                # PCG observability surface check (Wave 3).
                claims = result.get("claims") or []
                if claims and not any(c.get("pcg") for c in claims):
                    return False, "no claim carries pcg observability block"
                return True, (
                    f"job_id={job_id} doc_score={float(doc_score):.4f} "
                    f"claims={len(claims)}"
                )
            if state == "failed":
                return False, f"job failed: {resp.json()}"
        except Exception as exc:  # noqa: BLE001
            return False, f"poll exc={exc}"
        time.sleep(2.0)
    return False, f"poll timed out (job {job_id} not terminal after 120s)"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=int, default=600, help="Total seconds (default 600)")
    parser.add_argument("--interval", type=int, default=60, help="Seconds between probes (default 60)")
    parser.add_argument(
        "--max-failures",
        type=int,
        default=3,
        help="Consecutive failures before early exit (default 3)",
    )
    args = parser.parse_args(argv)

    api_url = os.environ.get("OHI_API_URL", "https://ohi-api.shiftbloom.studio").rstrip("/")
    edge_secret = os.environ.get("OHI_EDGE_SECRET")
    if not edge_secret:
        print("ERROR: OHI_EDGE_SECRET env var is required", file=sys.stderr)
        return 2
    claim_text = os.environ.get("PROBE_CLAIM_TEXT", DEFAULT_CLAIM)

    end_time = time.monotonic() + args.duration
    consecutive_failures = 0
    total_probes = 0
    total_failures = 0

    with httpx.Client() as client:
        while time.monotonic() < end_time:
            t0 = time.monotonic()
            ok, msg = _probe_once(client, api_url, edge_secret, claim_text)
            total_probes += 1
            elapsed = time.monotonic() - t0
            if ok:
                print(f"[probe] OK ({elapsed:.1f}s) {msg}")
                consecutive_failures = 0
            else:
                print(f"[probe] FAIL ({elapsed:.1f}s) {msg}", file=sys.stderr)
                consecutive_failures += 1
                total_failures += 1
                if consecutive_failures >= args.max_failures:
                    print(
                        f"::error::{consecutive_failures} consecutive probe failures — "
                        f"early exit after {total_probes} probes",
                        file=sys.stderr,
                    )
                    return 1
            time.sleep(max(0.0, args.interval - elapsed))

    rate_ok = total_failures == 0
    print(
        f"[probe] summary: {total_probes} probes, {total_failures} failures "
        f"({'GREEN' if rate_ok else 'AMBER'})"
    )
    return 0 if rate_ok else 1


if __name__ == "__main__":
    sys.exit(main())
