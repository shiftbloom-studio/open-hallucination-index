"""Wave 3 Stream G.1 — light load test for /api/v2/verify.

Fires N concurrent POST+poll cycles, tracks p50/p95/p99 end-to-end
latency, and the fraction of requests that converged (PCG algorithm
returned ``TRW-BP`` or ``LBP-fallback`` vs ``LBP-nonconvergent``).

Designed for item 7 of the acceptance matrix (post-deploy signal,
non-blocking). Picks a small fixed request budget so we don't hammer
production; tune via CLI flags for heavier dev runs.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import statistics
import sys
import time

import httpx


_CLAIMS = [
    "Marie Curie won two Nobel Prizes.",
    "Albert Einstein developed the theory of relativity.",
    "Stephen Hawking died in 2018.",
    "The Great Wall of China is visible from space with the naked eye.",
    "Water boils at 100 degrees Celsius at standard atmospheric pressure.",
]


async def _one_request(
    client: httpx.AsyncClient,
    api_url: str,
    edge_secret: str,
    text: str,
    timeout_s: float,
) -> tuple[bool, float, str]:
    """Returns ``(ok, elapsed_seconds, algorithm_tag)``."""
    headers = {"X-OHI-Edge-Secret": edge_secret, "Content-Type": "application/json"}
    t0 = time.monotonic()
    try:
        resp = await client.post(
            f"{api_url}/api/v2/verify",
            headers=headers,
            json={"text": text},
            timeout=timeout_s,
        )
        if resp.status_code != 202:
            return False, time.monotonic() - t0, f"POST {resp.status_code}"
        job_id = resp.json()["job_id"]

        deadline = t0 + timeout_s
        while time.monotonic() < deadline:
            r = await client.get(
                f"{api_url}/api/v2/verify/status/{job_id}",
                headers=headers,
                timeout=10.0,
            )
            body = r.json()
            state = body.get("status")
            if state == "done":
                claims = (body.get("result") or {}).get("claims") or []
                algo = "unknown"
                for c in claims:
                    pcg = c.get("pcg")
                    if pcg:
                        algo = pcg.get("algorithm", "unknown")
                        break
                return True, time.monotonic() - t0, algo
            if state == "failed":
                return False, time.monotonic() - t0, "failed"
            await asyncio.sleep(1.5)
        return False, time.monotonic() - t0, "poll-timeout"
    except Exception as exc:  # noqa: BLE001
        return False, time.monotonic() - t0, f"exc={type(exc).__name__}"


async def _amain(
    api_url: str, edge_secret: str, total: int, concurrency: int, timeout_s: float
) -> int:
    sem = asyncio.Semaphore(concurrency)
    results: list[tuple[bool, float, str]] = []

    async def _task(text: str) -> None:
        async with sem:
            async with httpx.AsyncClient() as client:
                res = await _one_request(client, api_url, edge_secret, text, timeout_s)
                results.append(res)

    tasks = [_task(_CLAIMS[i % len(_CLAIMS)]) for i in range(total)]
    await asyncio.gather(*tasks)

    latencies = [r[1] for r in results if r[0]]
    algos = [r[2] for r in results if r[0]]
    failures = [r for r in results if not r[0]]

    print(f"requests     : {len(results)}")
    print(f"success      : {len(latencies)}")
    print(f"failures     : {len(failures)}")
    if latencies:
        latencies.sort()
        print(f"latency p50  : {statistics.median(latencies):.2f} s")
        print(f"latency p95  : {latencies[int(0.95 * len(latencies)) - 1]:.2f} s")
        print(f"latency p99  : {latencies[int(0.99 * len(latencies)) - 1]:.2f} s")
    if algos:
        converged = sum(1 for a in algos if a in {"TRW-BP", "LBP-fallback"})
        print(f"BP-converged : {converged}/{len(algos)} ({converged / len(algos) * 100:.1f}%)")
    for ok, elapsed, msg in failures[:5]:
        print(f"  fail sample : {msg} after {elapsed:.2f}s")
    return 0 if not failures else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--total", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args(argv)

    api_url = os.environ.get("OHI_API_URL", "https://ohi-api.shiftbloom.studio").rstrip("/")
    edge_secret = os.environ.get("OHI_EDGE_SECRET")
    if not edge_secret:
        print("ERROR: OHI_EDGE_SECRET required", file=sys.stderr)
        return 2
    return asyncio.run(_amain(api_url, edge_secret, args.total, args.concurrency, args.timeout))


if __name__ == "__main__":
    sys.exit(main())
