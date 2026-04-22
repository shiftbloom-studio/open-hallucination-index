# Open Hallucination Index — API Documentation (v2)

> **Status:** v2 is under active development on `main` (the
> `feat/ohi-v2-foundation` branch was merged back into `main` at commit
> `c5010cc`). Some v2 features (domain routing, per-stratum
> calibration) are deferred to v2.1 / Wave 4. Today the API runs Wave 1
> + Wave 2 + most of Wave 3 code: MediaWiki + Wikidata + DBpedia MCP
> evidence, Gemini-3-Pro NLI, async polling verification, PCG belief
> propagation (TRW-BP / LBP / Gibbs), and claim-claim NLI dispatcher.
> Corpus-grounded passage evidence (Qdrant ANN → Bedrock rerank) is
> wired but dormant until Wave 3 Phase E ingestion runs. For the
> authoritative topology see
> [docs/CURRENT_ARCHITECTURE.md](CURRENT_ARCHITECTURE.md). For
> historical v1 surface (`/api/v1/verify`, trust_score, reasoning), see
> the repository's v1 tag once cut.

---

## Base URL

```
https://ohi-api.shiftbloom.studio
```

The service is fronted by Cloudflare which proxies to AWS API Gateway
(HTTP API) → AWS Lambda. All API routes sit under `/api/v2/…`; health
probes live at the origin root (`/health/*`) and are NOT under `/api/v2`.

Local dev API (if running the FastAPI app outside Lambda):

```
http://localhost:8080
```

---

## Authentication

Production traffic is protected by an **edge shared secret** injected by
Cloudflare Transform Rules:

```
X-OHI-Edge-Secret: <edge-secret>
```

The Lambda's `EdgeSecretMiddleware` rejects any request that reaches the
origin without the correct header. The secret is stored in AWS Secrets
Manager at `ohi/cf-edge-secret`. The middleware exempts `OPTIONS`
preflights so CORS works end-to-end.

**There is currently no user-level API key.** User auth is deferred to
v2.1+ and will likely layer on top of the edge-secret path (user keys
validated by the Lambda, edge-secret still required for transport
shape).

---

## CORS

The Lambda serves `Access-Control-Allow-Origin: https://ohi.shiftbloom.studio`
(single origin, matching the frontend). `allow_credentials=true` is set
to permit future bearer-token / cookie auth.

---

## Verify flow (Wave 2 / approach-C polling)

Verification runs asynchronously. Clients submit a claim, receive a
`job_id`, then poll a status endpoint until the verdict is ready. This
replaces the original v2 SSE design (Decision: blocked by CF free-tier
Host-header limits on Lambda Function URL; polling is the pragmatic
path, with ALB/Bedrock real-time streaming flagged for v2.1+).

### 1) Submit a verify job

```
POST /api/v2/verify
```

**Request body (JSON)**

| Field | Type | Required | Description |
|---|---|---|---|
| `text` | string | ✅ | Text to verify. Typical upper bound is a few thousand characters; long documents still run but may exceed Lambda's 180s timeout. |
| `rigor` | `"fast" \| "balanced" \| "maximum"` | ❌ | Rigor tier. Default `balanced`. Affects PCG behaviour (Wave 3+). |
| `retain` | boolean | ❌ | If `true`, the original text is persisted to the job record for later audit. Default `false` — only the derived verdict is retained, not the raw text. |

**Response — `202 Accepted`**

```json
{
  "job_id": "7a11a03e-0487-47d4-9a19-b4a7c22b00d9"
}
```

Returns in ~200 ms — no pipeline work runs on this path. The Lambda
writes a pending record to DynamoDB (`ohi-verify-jobs` table, 1h TTL)
and self-invokes asynchronously to run the pipeline.

**Error responses**

- `400 Bad Request` — malformed body, missing `text`.
- `401 Unauthorized` — missing or invalid `X-OHI-Edge-Secret`.
- `403 Forbidden` — Cloudflare WAF block (rare, only on suspicious
  request shape).
- `429 Too Many Requests` — CF rate limit (free-tier: ~100 POSTs/10s
  per IP, global 1000/h per IP).

### 2) Poll job status

```
GET /api/v2/verify/status/{job_id}
```

**Response — `200 OK`, shape varies by `status`.**

While running:

```json
{
  "job_id": "7a11a03e-0487-47d4-9a19-b4a7c22b00d9",
  "status": "pending",
  "phase": "retrieving_evidence",
  "created_at": "2026-04-18T14:12:03.102Z",
  "updated_at": "2026-04-18T14:12:05.881Z"
}
```

`phase` advances through: `queued` → `decomposing` → `retrieving_evidence`
→ `classifying` → `calibrating` → `assembling`. Phase transitions are
typically 1–3 s apart; a 2-claim document completes in ~5–10 s.

On success:

```json
{
  "job_id": "7a11a03e-0487-47d4-9a19-b4a7c22b00d9",
  "status": "done",
  "phase": "assembling",
  "result": { /* DocumentVerdict — see below */ },
  "created_at": "2026-04-18T14:12:03.102Z",
  "updated_at": "2026-04-18T14:12:12.443Z",
  "completed_at": "2026-04-18T14:12:12.443Z"
}
```

On failure:

```json
{
  "job_id": "7a11a03e-0487-47d4-9a19-b4a7c22b00d9",
  "status": "error",
  "error": "<message>",
  "created_at": "2026-04-18T14:12:03.102Z",
  "updated_at": "2026-04-18T14:12:10.220Z"
}
```

`404 Not Found` if the `job_id` is unknown or the record has expired
(1h TTL).

**Recommended polling pattern:** every ~1 s with exponential backoff on
transient network errors (cap 5 s), stop on `status in {"done", "error"}`.
Give up after ~3 min wall-clock (comfortably beyond Lambda's 180 s
ceiling). See [src/frontend/src/lib/verify-controller.ts](../src/frontend/src/lib/verify-controller.ts).

### 3) DocumentVerdict schema

```json
{
  "document_score": 0.78,
  "document_interval": [0.41, 0.95],
  "internal_consistency": 0.92,
  "claims": [
    {
      "claim": {
        "id": "uuid",
        "text": "Marie Curie won two Nobel prizes.",
        "claim_type": "quantitative",
        "subject": "Marie Curie",
        "predicate": "number of Nobel prizes won",
        "object": "2"
      },
      "p_true": 0.91,
      "interval": [0.62, 0.99],
      "domain": "general",
      "supporting_evidence": [ /* EvidencePassage */ ],
      "refuting_evidence": [ /* EvidencePassage */ ],
      "pcg_neighbors": [ /* ClaimNeighborLink — Wave 3+ */ ],
      "nli_self_consistency_variance": 0.02,
      "information_gain": 0.44,
      "fallback_used": null
    }
  ],
  "decomposition_coverage": 1.0,
  "processing_time_ms": 6432.1,
  "rigor": "balanced",
  "refinement_passes_executed": 0,
  "pipeline_version": "ohi-v2.0",
  "model_versions": {
    "decomposer": "phase1-default",
    "domain_router": "phase1-placeholder-general",
    "pcg": "phase2-beta-posterior-from-nli",
    "conformal": "phase1-split-conformal-stub",
    "nli_adapter": "NliGeminiAdapter"
  },
  "request_id": "uuid"
}
```

**Field notes:**

- `document_score` — geometric mean of per-claim `p_true`. Float in
  [0, 1]. **NOTE: named `document_score`, not `doc_score`.**
- `document_interval` — bounds on the document score.
- `internal_consistency` — measure of agreement across claims sharing
  entities; surfaced by PCG in Wave 3+.
- `p_true` — posterior probability the claim is true. Uniform `0.5`
  with interval `[0, 1]` signals the GENERAL fallback (no meaningful
  evidence retrieved or NLI not yet integrated for that path).
- `fallback_used` — `null` when calibrated; `"general"` when falling
  back due to missing calibration data or empty evidence. **Expected
  in v2.0.** Removed when Wave 4 / v2.1 adds calibration data.
- `supporting_evidence` / `refuting_evidence` — arrays of
  `EvidencePassage` objects (see below).
- `pcg_neighbors` — empty in Wave 2; populated in Wave 3 with
  claim-graph neighbor links.
- `model_versions.pcg` — `"phase2-beta-posterior-from-nli"` after D1;
  will flip to `"wave3-trw-bp-lbp-gibbs-v1"` when PCG is implemented.
- `model_versions.nli_adapter` — `"NliGeminiAdapter"` after D1/F
  deploy.

### 4) EvidencePassage schema

```json
{
  "passage_id": "uuid",
  "source": "mediawiki",
  "source_url": "https://en.wikipedia.org/wiki/Marie_Curie",
  "title": "Marie Curie",
  "text": "Marie Skłodowska Curie…",
  "score": 0.88,
  "nli_label": "support",
  "nli_confidence": 0.92
}
```

Sources currently: `mediawiki` (live Wikipedia search via MCP). Wave 3+
adds `neo4j` (entity graph walks) and `qdrant` (passage vector search)
once corpus ingestion ships.

---

## Health

Health endpoints sit at the **origin root**, not under `/api/v2`.

### `GET /health/live`

Liveness. Returns `200 OK` + `{"status":"ok"}` if the process is up.
Cheap; no dependency checks.

### `GET /health/deep`

Deep probe with per-dependency detail.

```json
{
  "status": "ok",
  "timestamp": "2026-04-18T14:12:03.102Z",
  "layers": {
    "L1.decompose":          {"status": "ok", "latency_ms": 70.66,  "detail": null},
    "L1.retrieve.neo4j":     {"status": "ok", "latency_ms": 3.75,   "detail": null},
    "L1.retrieve.qdrant":    {"status": "ok", "latency_ms": 29.51,  "detail": null},
    "L1.retrieve.trace":     {"status": "skipped", "latency_ms": null, "detail": "disabled by config"},
    "pipeline.orchestrator": {"status": "ok", "latency_ms": 0.26,   "detail": null},
    "L7.verdict_store":      {"status": "ok", "latency_ms": 0.0,    "detail": null}
  },
  "model_versions": { /* same shape as DocumentVerdict.model_versions */ },
  "calibration_freshness_hours": null
}
```

Top-level `status` is `"ok"` when all non-skipped layers are healthy,
`"degraded"` when any layer is `"error"`. (A legacy alias `overall` is
also accepted by the frontend for tolerance.)

---

## Retention & privacy

- `retain=false` (default) — only derived verdict data is persisted.
  The job record sets `text=""`; original text is NOT stored.
- `retain=true` — the full request text is saved to the job record for
  audit purposes.
- Job records auto-expire via DynamoDB TTL after 1 hour
  (`OHI_ASYNC_VERIFY_TTL_SECONDS=3600`).
- No per-user identification in v2.0. Request IDs are UUIDs, not user
  or IP linked.

---

## Error format

Errors follow FastAPI's default shape:

```json
{
  "detail": "<message or validation object>"
}
```

Common conditions:

| Code | Meaning |
|---|---|
| 400 | malformed body / validation error |
| 401 | missing or wrong `X-OHI-Edge-Secret` |
| 403 | Cloudflare WAF block |
| 404 | unknown `job_id` (GET status), or expired |
| 429 | Cloudflare rate limit |
| 500 | Lambda runtime error (investigate via CloudWatch) |
| 504 | Lambda timed out at 180s OR API Gateway's 30s integration cap (sync path only — note: the sync path is used only for legacy clients) |

---

## Rigor tiers (Wave 3+)

In Wave 2 today, `rigor` is accepted as input but the PCG path is the
same in all tiers (Beta-posterior from claim-evidence NLI only). Wave 3
brings the tier semantics to life:

| Tier | PCG algorithm | Claim-claim NLI | Gibbs sanity | Target latency |
|---|---|---|---|---|
| `fast` | damped LBP only | skipped | no | minimize |
| `balanced` (default) | TRW-BP → damped LBP fallback | entity-overlap short-circuit | yes | ~10–30 s typical |
| `maximum` | TRW-BP, stricter tolerance | entity-overlap short-circuit | yes, larger sample count | minutes tolerated |

---

## Pipeline and model versions

Wave-2 baseline (post-E2 deploy):

| Stage | Implementation |
|---|---|
| Decomposer | `phase1-default` — Gemini 3 Pro native adapter, safetySettings BLOCK_NONE, thinkingLevel HIGH |
| Evidence retrieval | Live MediaWiki MCP + (empty) Neo4j Aura + (empty) Qdrant |
| NLI (claim-evidence) | `NliGeminiAdapter` — Gemini 3 Pro, self-consistency K=1 default, max_retries=3 |
| NLI (claim-claim) | not implemented in Wave 2 (arrives Wave 3) |
| Domain router | `phase1-placeholder-general` — every claim → `general` |
| PCG | `phase2-beta-posterior-from-nli` — per-claim Beta update from claim-evidence NLI; no graph propagation |
| Conformal calibrator | `phase1-split-conformal-stub` — 0 calibration samples, every verdict has `fallback_used: "general"` with wide interval |

Wave-3 targets:

| Stage | Implementation |
|---|---|
| NLI (claim-claim) | `NliOpenAiAdapter` (GPT 5.4 xhigh) primary + `NliGeminiAdapter` fallback on transport error |
| PCG | `wave3-trw-bp-lbp-gibbs-v1` — TRW-BP primary, damped LBP fallback, Gibbs MCMC sanity in balanced+ rigor |
| Evidence retrieval | + Qdrant passage vectors (PC-hosted, vector-only payload) + Neo4j entity-graph walks |
| Domain router | still `phase1-placeholder-general`; domain routing deferred to Wave 4 / v2.1 |
| Conformal calibrator | still stub; calibration data deferred to Wave 4 / v2.1 |

---

## Verdict JSON field additions (Wave 3)

```json
{
  "claims": [
    {
      "...": "existing fields",
      "pcg": {
        "algorithm": "TRW-BP",
        "converged": true,
        "iterations": 37,
        "edge_count": 4,
        "log_partition_bound": -8.42,
        "gibbs_mismatch": null
      }
    }
  ]
}
```

`log_partition_bound` is `null` for non-TRW algorithms. `gibbs_mismatch`
is `null` unless the Gibbs sanity run flags a disagreement.

---

## Changelog

- **v2.0 (in progress on `main` at `cf77b24`):**
  - Polling verify flow replaces the original SSE design (Decision K
    from Wave-3 spec kickoff: CF free-tier Host-header limits blocked
    Function URL as origin; polling is pragmatic).
  - Gemini 3 Pro NLI wired end-to-end in the pipeline.
  - Live MediaWiki evidence retrieval.
  - Lambda timeout bumped 60 → 180 s for NLI latency budget.
  - DynamoDB `ohi-verify-jobs` table for job state + self-async-invoke.
- **v1 (legacy):** `/api/v1/verify` sync endpoint with `trust_score`,
  `claims[].status`, `strategy` selector. Not deployed in v2
  infrastructure; see pre-rework tag for historical reference.

---

## Linked documents

- [README.md](../README.md) — project overview and current state
- [CLAUDE.md](../CLAUDE.md) — operational rules for agents + humans
- [docs/FRONTEND.md](FRONTEND.md) — frontend architecture (polling,
  static export on Vercel)
- [docs/PUBLIC_ACCESS.md](PUBLIC_ACCESS.md) — public access framework
- [docs/CONTRIBUTING.md](CONTRIBUTING.md) — contribution process
- Runbooks: [docs/runbooks/](runbooks/)
