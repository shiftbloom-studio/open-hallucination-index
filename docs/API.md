# Open Hallucination Index API

**Production base URL:** `https://ohi.shiftbloom.studio`

The production API is served by the Cloudflare Worker in `cloudflare/ohi-worker`. All verification API routes live under `/api/v2`; health and MCP routes live at the origin root.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health/live` | Liveness probe |
| `GET` | `/health/ready` | Binding readiness for D1, Durable Objects, Queue, Workers AI, Vectorize |
| `GET` | `/health/deep` | Pipeline smoke for decomposition, Wikimedia retrieval, Vectorize, NLI |
| `POST` | `/api/v2/verify` | Submit text for async verification |
| `GET` | `/api/v2/verify/status/{job_id}` | Poll job status and final verdict |
| `GET` | `/api/v2/calibration/report` | Public calibration report |
| `POST` | `/api/v2/feedback` | Submit claim feedback |
| `GET` | `/api/v2/admin/corpus` | Admin-only corpus overview |
| `POST` | `/api/v2/admin/corpus/runs` | Admin-only corpus seed/backfill start |
| `GET` | `/api/v2/admin/corpus/runs/{run_id}` | Admin-only corpus run status |
| `POST` | `/mcp` | Streamable HTTP MCP endpoint |

## Verify

### `POST /api/v2/verify`

Request:

```json
{
  "text": "The Eiffel Tower is in Paris.",
  "context": null,
  "domain_hint": "general",
  "options": {
    "rigor": "fast",
    "max_claims": 2,
    "coverage_target": 0.9
  },
  "request_id": null
}
```

Production requires `turnstile_token` from the frontend Cloudflare Turnstile widget. Direct public submissions without it return `403 turnstile_required`.

Response:

```json
{
  "job_id": "6044b807-f166-4a82-a6db-8bd478c2629a"
}
```

Status: `202 Accepted`

### `GET /api/v2/verify/status/{job_id}`

Pending response:

```json
{
  "job_id": "6044b807-f166-4a82-a6db-8bd478c2629a",
  "status": "pending",
  "phase": "retrieving_evidence",
  "created_at": 1783211197.025,
  "updated_at": 1783211203.125
}
```

Done response:

```json
{
  "job_id": "6044b807-f166-4a82-a6db-8bd478c2629a",
  "status": "done",
  "phase": "assembling",
  "created_at": 1783211197.025,
  "updated_at": 1783211217.752,
  "completed_at": 1783211217.752,
  "result": {
    "request_id": "0609d322-629d-420d-ad33-288e9c17be8b",
    "pipeline_version": "ohi-v2.0-cloudflare",
    "document_score": 0.4205,
    "document_interval": [0.1095, 0.8718],
    "rigor": "fast",
    "claims": []
  }
}
```

Unknown job:

```json
{
  "detail": {
    "code": "job_not_found",
    "message": "Unknown job id"
  }
}
```

Status: `404 Not Found`

## Verdict Shape

`result.claims[]` contains:

- `claim`: `id`, `text`, `claim_type`, `span`
- `p_true`: calibrated probability estimate
- `interval`: uncertainty interval
- `coverage_target`: requested coverage target
- `domain`: assigned domain
- `supporting_evidence`: evidence rows classified as support
- `refuting_evidence`: evidence rows classified as refute
- `queued_for_review`: true when evidence is weak or uncertainty is wide
- `fallback_used`: currently `"general"` while domain calibration data is sparse

Evidence rows contain:

- `source_uri`
- `content`
- `snippet`
- `source_credibility`
- `similarity_score`
- `classification_confidence`
- `structured_data`
- `retrieved_at`

## Health

### `GET /health/ready`

```json
{
  "ready": true,
  "services": {
    "d1": { "connected": true, "status": "healthy" },
    "durable_objects": { "connected": true, "status": "healthy" },
    "queue": { "connected": true, "status": "configured" },
    "workers_ai": { "connected": true, "status": "configured" },
    "vectorize": { "connected": true, "status": "configured" }
  }
}
```

### `GET /health/deep`

```json
{
  "status": "ok",
  "pipeline_version": "ohi-v2.0-cloudflare",
  "layers": {
    "L1.decompose": { "status": "ok", "latency_ms": 23 },
    "L1.retrieve.wikimedia": { "status": "ok", "latency_ms": 285 },
    "L1.retrieve.vectorize": { "status": "ok", "latency_ms": 205 },
    "L3.nli": { "status": "ok", "latency_ms": 31 }
  }
}
```

## MCP

The Worker exposes a Streamable HTTP MCP server at `/mcp`.

Initialize:

```bash
curl -sS -X POST https://ohi.shiftbloom.studio/mcp \
  -H 'content-type: application/json' \
  -H 'accept: application/json, text/event-stream' \
  --data '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"client","version":"1.0.0"}}}'
```

Tools:

- `verify_text`
- `job_status`
- `search_evidence`
- `search_all`
- `search_wikipedia`
- `get_wikipedia_summary`
- `get_summary`
- `search_wikidata`
- `query_wikidata_sparql`
- `search_dbpedia`
- `resolve-library-id`
- `query-docs`
- `search_academic`
- `search_openalex`
- `search_crossref`
- `get_doi_metadata`
- `search_pubmed`
- `search_europepmc`
- `search_clinical_trials`
- `get_citations`
- `search_gdelt`
- `get_world_bank_indicator`
- `search_vulnerabilities`
- `get_vulnerability`

## Corpus Admin

Admin endpoints require `Authorization: Bearer <ADMIN_TOKEN>` or `X-OHI-Admin-Token`.

Start a curated Wikipedia seed:

```bash
curl -sS -X POST https://ohi.shiftbloom.studio/api/v2/admin/corpus/runs \
  -H "authorization: Bearer $ADMIN_TOKEN" \
  -H "content-type: application/json" \
  --data '{"source":"wikipedia","mode":"seed","limit":1000}'
```

Corpus runs are orchestrated by Cloudflare Workflows, fanned out through `ohi-corpus-ingest`, embedded with Workers AI, stored in D1, archived as raw JSON objects in R2, and indexed into Vectorize. New runs expose `batches_total` and `batches_completed`; completion is idempotent per `(run_id, batch)` to tolerate Cloudflare Queues at-least-once delivery.

Current verified production corpus:

- D1: 987 documents, 3,366 chunks, 25 Wikidata entities, 1,263 graph edges.
- R2: bucket `ohi-corpus-prod`.
- Large seed `cfabf0a5-9c25-419c-a243-b5ea79136155`: 1,189 seen, 1,085 indexed, 3,948 chunks, zero errors.
- Post-fix idempotency seed `9a6c58df-e5bc-4ac7-b6ca-d5a83d648382`: `batches_total=2`, `batches_completed=2`, zero errors.

## Runtime Products

- Cloudflare Workers
- Worker Static Assets
- Durable Objects
- D1
- R2
- Vectorize
- Queues
- Workflows
- Workers AI
- Turnstile
- WAF / Rate Limiting Rulesets
- Cloudflare Agents SDK MCP server
