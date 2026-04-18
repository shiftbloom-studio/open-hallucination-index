"""Stream F per-deploy smoke — plan §5.3 phase 4 tier 1.

Asserts the deploy lifted the image and the pipeline responds with the
expected shape and model_versions. Does NOT assert signal quality
(evidence count, p_true direction) — that's the end-of-phase smoke.

Hard-coded tolerances, not exact values. Exits non-zero on any failure
so the deploy orchestrator can detect it.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from typing import Any

import urllib.request
import urllib.error

API = "https://ohi-api.shiftbloom.studio"
CLAIM_TEXT = "Marie Curie won two Nobel prizes."

EXPECTED_MV_NLI = "NliGeminiAdapter"
EXPECTED_MV_PCG = "phase2-beta-posterior-from-nli"


def _edge_secret() -> str:
    # Matches runbook: pulled from AWS SM.
    out = subprocess.check_output(
        [
            "aws", "secretsmanager", "get-secret-value",
            "--secret-id", "ohi/cf-edge-secret",
            "--region", "eu-central-1",
            "--query", "SecretString", "--output", "text",
        ],
        text=True,
    ).strip()
    return out


def _post(url: str, body: dict[str, Any], headers: dict[str, str], timeout_s: float = 60.0) -> tuple[int, dict[str, Any] | str]:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8")
            try:
                return resp.status, json.loads(raw)
            except json.JSONDecodeError:
                return resp.status, raw
    except urllib.error.HTTPError as e:
        body_raw = e.read().decode("utf-8", errors="replace")
        return e.code, body_raw
    except Exception as e:
        return -1, f"transport error: {e!r}"


def _get(url: str, headers: dict[str, str], timeout_s: float = 15.0) -> tuple[int, dict[str, Any] | str]:
    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8")
            try:
                return resp.status, json.loads(raw)
            except json.JSONDecodeError:
                return resp.status, raw
    except urllib.error.HTTPError as e:
        body_raw = e.read().decode("utf-8", errors="replace")
        return e.code, body_raw
    except Exception as e:
        return -1, f"transport error: {e!r}"


def main() -> int:
    fails: list[str] = []
    edge = _edge_secret()
    headers = {
        "Content-Type": "application/json",
        "X-OHI-Edge-Secret": edge,
        # CF WAF rejects urllib's default "Python-urllib/3.x" UA with
        # `error code: 1010` (banned browser signature). Any non-default
        # UA passes — this is the OHI smoke signature.
        "User-Agent": "ohi-stream-f-smoke/1.0",
    }

    # === /health/deep ===
    print("[1/4] GET /health/deep")
    t0 = time.perf_counter()
    status, body = _get(f"{API}/health/deep", headers)
    t_ms = (time.perf_counter() - t0) * 1000
    print(f"  status={status} t={t_ms:.0f}ms")
    if status != 200:
        fails.append(f"/health/deep HTTP {status}: {body!r}")
    elif not isinstance(body, dict):
        fails.append(f"/health/deep non-JSON body: {body!r}")
    else:
        overall = body.get("status") or body.get("overall")
        if overall != "ok":
            fails.append(f"/health/deep status != ok: {overall!r}; body={body}")
        print(f"  overall={overall}")
        model_versions = body.get("model_versions") or {}
        print(f"  model_versions={model_versions}")

    # === POST /verify (Marie Curie positive) ===
    print("[2/4] POST /api/v2/verify — Marie Curie (positive)")
    t0 = time.perf_counter()
    status, body = _post(
        f"{API}/api/v2/verify",
        {"text": CLAIM_TEXT},
        headers,
        timeout_s=60.0,
    )
    t_ms = (time.perf_counter() - t0) * 1000
    print(f"  status={status} t={t_ms:.0f}ms")

    if status != 200:
        fails.append(f"/verify HTTP {status}: {body!r}")
        return _done(fails)
    if not isinstance(body, dict):
        fails.append(f"/verify non-JSON body: {body!r}")
        return _done(fails)

    # Save for the end-of-phase smoke to reuse.
    with open("tests/stream_f/out_verify_marie_curie.json", "w", encoding="utf-8") as f:
        json.dump(body, f, indent=2, ensure_ascii=False)

    # [3/4] Shape checks
    print("[3/4] shape checks")
    doc_score = body.get("document_score")
    if not isinstance(doc_score, (int, float)) or not (0.0 <= float(doc_score) <= 1.0):
        fails.append(f"document_score not a float in [0,1]: {doc_score!r}")
    else:
        print(f"  document_score={doc_score}")

    claims = body.get("claims") or []
    if not isinstance(claims, list) or not claims:
        fails.append(f"claims array empty or missing: {claims!r}")
    else:
        print(f"  claims count={len(claims)}")
        for i, c in enumerate(claims):
            pt = c.get("p_true")
            if pt is None:
                fails.append(f"claim[{i}] p_true is null")
            else:
                print(f"  claim[{i}] text={c.get('text','')[:60]!r} p_true={pt}")

    mv = body.get("model_versions") or {}
    if mv.get("nli_adapter") != EXPECTED_MV_NLI:
        fails.append(f"model_versions.nli_adapter != {EXPECTED_MV_NLI}: {mv.get('nli_adapter')!r}")
    if mv.get("pcg") != EXPECTED_MV_PCG:
        fails.append(f"model_versions.pcg != {EXPECTED_MV_PCG}: {mv.get('pcg')!r}")
    print(f"  model_versions={mv}")

    # Processing time sanity
    pt_ms = body.get("processing_time_ms")
    if pt_ms is None or pt_ms >= 30_000:
        fails.append(f"processing_time_ms out of bounds: {pt_ms!r}")
    else:
        print(f"  processing_time_ms={pt_ms}")

    # [4/4] Evidence present (sanity, not signal-quality)
    print("[4/4] evidence sanity (any claim has at least 1 evidence item)")
    any_evidence = False
    for c in claims:
        if (c.get("supporting_evidence") or []) or (c.get("refuting_evidence") or []):
            any_evidence = True
            break
    if not any_evidence:
        # Not a hard fail for per-deploy smoke per plan §5.3 tier 1, but
        # highly informative; record as a warning so the handoff surfaces it.
        print("  WARNING: no claim has any supporting/refuting evidence (but per-deploy tier 1 doesn't fail on this).")
    else:
        print("  OK: at least one claim has evidence.")

    return _done(fails)


def _done(fails: list[str]) -> int:
    print("")
    if fails:
        print("PER-DEPLOY SMOKE FAILED:")
        for f in fails:
            print(f"  - {f}")
        return 1
    print("PER-DEPLOY SMOKE: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
