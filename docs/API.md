# Open Hallucination Index – API Documentation

> **Objective:** This specification describes the HTTP interfaces of the OHI API for verifiable fact checking, evidence aggregation, and trust scoring. All endpoints are deterministically documented and designed for reproducible research experiments.

---

## 🧪 Scientific Framework

The API models the verification process as a pipeline:

1. **Claim Decomposition**: Breaking text into atomic claims.
2. **Evidence Retrieval**: Parallel searching across graph, vector, and MCP sources.
3. **Evidence Alignment**: Mapping evidence to claims.
4. **Trust Scoring**: Evaluation through evidence-based metrics.

Main metrics are:

- **Support Ratio** $\frac{n_{supported}}{n_{total}}$
- **Refutation Ratio** $\frac{n_{refuted}}{n_{total}}$
- **Confidence** (0–1) as a confidence interval estimator
- **Overall Trust** as a weighted aggregation

---

## 🔐 Authentication

By default, the API expects an API key header:

```
X-API-Key: <YOUR_API_KEY>
```

Configuration is handled via `API_API_KEY` in the API environment.

---

## 🌐 Base URL

Default:

```
http://localhost:8080
```

---

## ✅ Core Endpoints

### 1) Verify (Single)

**Route**
```
POST /api/v1/verify
```

**Description**: Verifies a text and returns trust scores, claim evidence, and summary.

**Request Schema (JSON)**

| Field | Type | Required | Description |
|------|-----|---------|--------------|
| `text` | string | ✅ | Text for verification (max. 100,000 characters) |
| `context` | string | ❌ | Optional context for disambiguation |
| `strategy` | string | ❌ | `mcp_enhanced` · `hybrid` · `cascading` · `graph_exact` · `vector_semantic` · `adaptive` |
| `use_cache` | boolean | ❌ | Cache usage (default: `true`) |
| `target_sources` | integer | ❌ | Target number of checked sources (1–20) |

**Example**
```
curl -X POST http://localhost:8080/api/v1/verify \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_API_KEY" \
  -d '{"text": "The Eiffel Tower is in Paris and was built in 1889."}'
```

**Example Response (truncated)**
```
{
  "id": "abc123...",
  "trust_score": {
    "overall": 0.988,
    "claims_total": 2,
    "claims_supported": 2,
    "claims_refuted": 0,
    "claims_unverifiable": 0,
    "confidence": 0.92,
    "scoring_method": "weighted_average"
  },
  "claims": [
    {
      "id": "f5a1...",
      "text": "The Eiffel Tower is in Paris.",
      "status": "supported",
      "confidence": 0.91,
      "reasoning": "Found supporting evidence in sources."
    }
  ],
  "summary": "2 claims analyzed, 2 supported. Trust level: high (0.99).",
  "processing_time_ms": 42.3,
  "cached": false
}
```

---

### 2) Verify (Batch)

**Route**
```
POST /api/v1/verify/batch
```

**Description**: Parallelized verification of multiple texts.

**Request Schema**

| Field | Type | Required | Description |
|------|-----|---------|--------------|
| `texts` | array | ✅ | List of texts (max. 50) |
| `strategy` | string | ❌ | Verification strategy (optional) |
| `use_cache` | boolean | ❌ | Cache usage |

**Note**: Max. 50 texts per request.

---

### 3) Health

| Endpoint | Purpose |
|----------|------|
| `GET /health` | Basic health (alias for liveness) |
| `GET /health/live` | Liveness probe |
| `GET /health/ready` | Readiness probe including dependency status |

---

## 🧠 Verification Strategies

| Strategy | Characteristics | Recommended for |
|-----------|----------------|--------------|
| `mcp_enhanced` | Local sources + MCP sources (e.g., Wikipedia/Context7) | Highest evidence coverage |
| `hybrid` | Parallel graph + vector search | Fast local verification |
| `cascading` | Graph first, vector fallback | Precision over recall |
| `graph_exact` | Neo4j exact matching | Entity consistency |
| `vector_semantic` | Qdrant semantics | Content similarity |
| `adaptive` | Tiered retrieval with early-exit | Balances speed & coverage |

---

## 🧾 Fehlerformate

Fehler werden als strukturierte JSON‑Antwort geliefert:

```
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "text must not be empty",
    "details": { "field": "text" }
  }
}
```

**Typische Fehlercodes**

- `AUTH_REQUIRED`
- `AUTH_INVALID`
- `VALIDATION_ERROR`
- `RATE_LIMIT`
- `INTERNAL_ERROR`

---

## 🧰 Datenmodelle (konzeptionell)

**Claim**

- `id`: UUID
- `text`: string
- `status`: `supported` | `refuted` | `partially_supported` | `unverifiable` | `uncertain`
- `confidence`: float
- `reasoning`: string

**Evidence**

- `source`: string
- `snippet`: string
- `score`: float
- `url`: string

**TrustScore**

- `overall`: float
- `claims_total`: int
- `claims_supported`: int
- `claims_refuted`: int
- `claims_unverifiable`: int
- `confidence`: float
- `scoring_method`: string

---

## 🔬 Reproducibility

For scientific reproducibility, you should:

1. Fix strategies and sources via configuration.
2. Document version states of knowledge sources.
3. Version and archive requests and responses.

---

## 🧭 Knowledge Track (Provenance)

Additional endpoints provide provenance and source lists for claims:

- `GET /api/v1/knowledge-track/{claim_id}` – full knowledge track including mesh
- `HEAD /api/v1/knowledge-track/{claim_id}` – existence check
- `GET /api/v1/knowledge-track/sources/available` – available MCP sources

Parameters:
- `depth` (1–5)
- `generate_detail` (bool, default: true)

---

## 🔗 Further Documents

- [docs/FRONTEND.md](FRONTEND.md)
- [docs/CONTRIBUTING.md](CONTRIBUTING.md)
- [docs/PUBLIC_ACCESS.md](PUBLIC_ACCESS.md)
