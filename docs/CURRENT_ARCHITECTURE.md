# Current Architecture

**Single Source of Truth (SSoT)** for the OHI production and local-dev
topology. If a claim in `CLAUDE.md`, `README.md`, a runbook, or any
other doc contradicts this file, **this file wins** — other docs should
be updated to match.

Any architectural change MUST update this file in the same commit. See
[CLAUDE.md](../CLAUDE.md#single-source-of-truth-for-architecture) for
the enforcing rule.

**Last verified against prod:** 2026-04-22
- `main` tip: `cf77b24` (docs: add CURRENT_ARCHITECTURE + OHI_onepager.pdf)
- Lambda image digest (live): `sha256:0e3676a2ac41123fecd6b7e0434f0a0a8142a6cb1b26de9438927276c75b4113` — ECR tag `hotfix-20260421-nli-refute`
- ECR `:prod` tag: `sha256:7578dd4b89bbf5c5145a534f475581baf0efc8a1b4c6e44cd78581f32c995682` — tags `[prod, sha-cf77b247...]`
- Lambda `LastModified`: `2026-04-21T21:29:54Z`
- Drift note: a silent rollback on 2026-04-21 put Lambda on the hotfix
  image while `:prod` still points at the subsequent post-Bedrock
  image. Env vars are Bedrock-configured
  (`OHI_EMBEDDING_BACKEND=bedrock`) and the hotfix image supports the
  Bedrock backend selector, so prod is functional — but redeploying
  `:prod` will roll forward to `cf77b24`-content. See [runbooks/rollback-deploy.md](runbooks/rollback-deploy.md).

This document is based on the current repo state in:

- `README.md`
- `docs/FRONTEND.md`
- `infra/terraform/compute/`
- `infra/terraform/cloudflare/`
- `infra/terraform/storage/`
- `infra/terraform/jobs/`
- `infra/terraform/vercel/`
- `docker/compose/pc-data.yml`
- `docker/compose/docker-compose.yml`
- Live Lambda env (`aws lambda get-function-configuration --function-name ohi-api`)

## 1. Production Architecture

```mermaid
flowchart LR
    user["User Browser"]
    site["Vercel Static Frontend\nohi.shiftbloom.studio"]
    edge["Cloudflare Edge\nDNS + WAF + rate limiting"]

    subgraph aws["AWS eu-central-1"]
        apigw["API Gateway HTTP API\ncustom domain"]
        lambda["Lambda Container\nOHI API (2048 MB, 180 s)"]
        ddb["DynamoDB\nohi-verify-jobs (1h TTL)"]
        sm["Secrets Manager"]
        s3["S3 Artifacts"]
        logs["CloudWatch Logs"]
        br_embed["Bedrock Runtime\nTitan Text v2 (1024-dim)"]
        br_rerank["Bedrock Runtime\nCohere rerank-v3-5"]
    end

    subgraph pc["PC-side Services via Cloudflare Tunnel"]
        cfaccess["Cloudflare Tunnel + Access"]
        qdrant["Qdrant\nohi_passages_titan1024"]
    end

    aura["Neo4j Aura Pro\nFrankfurt — neo4j+s://0193408e"]
    wiki["MediaWiki + Wikidata + DBpedia"]
    gemini["Gemini API\nL1 decomposer + L3 claim-evidence NLI"]
    openai["OpenAI API\nWave 3 claim-claim NLI (optional)"]

    user -->|"GET /"| site
    user -->|"POST/GET /api/v2/*"| edge
    edge -->|"ohi-api.shiftbloom.studio"| apigw
    apigw --> lambda

    lambda --> ddb
    lambda --> sm
    lambda --> s3
    lambda --> logs
    lambda --> br_embed
    lambda --> br_rerank

    lambda --> aura
    lambda --> wiki
    lambda --> gemini
    lambda -.-> openai

    lambda -->|"HTTPS + service token"| cfaccess
    cfaccess --> qdrant
```

### Production flow summary

1. The browser loads the statically exported frontend from Vercel.
2. The browser POSTs `/api/v2/verify` → receives `202 + job_id`, then
   polls `GET /api/v2/verify/status/{job_id}` until `done` or `error`.
3. Cloudflare protects the public API entrypoint
   (`ohi-api.shiftbloom.studio`) and forwards traffic to the
   API Gateway HTTP API custom domain; an edge-secret is injected into
   `X-OHI-Edge-Secret` by a Transform Rule.
4. API Gateway invokes the Lambda-based OHI API. The sync `/verify`
   entry writes a pending record to DynamoDB, self-async-invokes, and
   returns `202`. The async handler runs the pipeline and writes the
   final verdict back to the same DynamoDB record.
5. **Embeddings** (query + passages): **AWS Bedrock Titan Text
   Embeddings V2**, 1024-dim. Env: `OHI_EMBEDDING_BACKEND=bedrock`,
   `BEDROCK_EMBED_MODEL_ID=amazon.titan-embed-text-v2:0`.
6. **Reranking**: **AWS Bedrock Cohere rerank-v3-5** on the top-40
   Qdrant ANN candidates, returning top-12 passages. Env:
   `BEDROCK_RERANK_ENABLED=true`.
7. **Graph**: Neo4j **Aura Pro** (Frankfurt) holds entities +
   passage-level text. Connected via `neo4j+s://` (bolt over TLS).
8. **Vector**: **Qdrant on PC** via Cloudflare Tunnel holds the
   `ohi_passages_titan1024` collection (1024-dim passage embeddings).
   Lambda reaches it at `https://ohi-qdrant.shiftbloom.studio` with
   CF Access service-token headers.
9. **LLM**: **native Gemini adapter** (`src/api/adapters/gemini.py`)
   for both L1 decomposition and L3 claim-evidence NLI. The Gemini
   OpenAI-compat shim is kept in-tree as a fallback (`LLM_BACKEND=openai`)
   but NOT used in prod — the shim silently drops `safetySettings` +
   `generationConfig.thinkingConfig`, unacceptable for a
   hallucination-detection product.
10. **Claim-claim NLI** (Wave 3): OpenAI `gpt-5.4-xhigh` primary,
    Gemini 3 Pro preview fallback. Today `OHI_OPENAI_API_KEY` is unset
    / empty in Lambda, so the dispatcher resolves to Gemini as both
    primary AND fallback (logged as a warning at init).

### Deploy path

- `main` push → GHA workflow `.github/workflows/v2-main-deploy.yml`:
  build image → push to ECR (`:sha-<sha>` immutable + `:prod` mutable)
  → `aws lambda update-function-code --image-uri <digest>` → health
  check → auto-rollback on failure. Terraform is skipped for
  code-only changes.
- Terraform `infra/terraform/compute/` owns env vars, IAM, memory,
  timeout, and the self-invoke permission. Apply only when TF-owned
  config changes.

## 2. Local Development Architecture

```mermaid
flowchart LR
    browser["Developer Browser"]

    subgraph app["Local App Layer"]
        next["Next.js Dev Server\nlocalhost:3000"]
        api["FastAPI / ohi-server\nlocalhost:8080"]
    end

    subgraph infra["Local Infra (docker/compose/pc-data.yml)"]
        neo4j["Neo4j\n7474 / 7687"]
        qdrant["Qdrant\n6333 / 6334"]
        postgres["Postgres 16\n5432"]
        postgrest["PostgREST"]
        redis["Redis\n6379"]
        webdis["Webdis"]
        embed["pc-embed\nMiniLM-L12-v2 (local-only)"]
    end

    browser -->|"http://localhost:3000"| next
    next -->|"verify + status requests"| api

    api --> neo4j
    api --> qdrant
    api -.-> redis
    api -.-> postgres
    api -.-> embed
```

### Local flow summary

- Preferred workflow is local-first: run Next.js and FastAPI natively,
  use Docker for supporting infra via the `local-dev` profile in
  `docker/compose/pc-data.yml`.
- Local FastAPI defaults to `OHI_EMBEDDING_BACKEND=local` (in-process
  sentence-transformers). Set `=bedrock` with AWS creds to exercise
  the prod path; set `=remote` to hit the PC-hosted `pc-embed`
  container via HTTP.
- `docker/compose/docker-compose.yml` is a legacy full-stack compose
  that also bundles a Lambda-like API container, MCP server, and
  Nginx. It is retained for end-to-end smoke but is not the canonical
  local workflow.

## 3. PC-side Data Stack

`docker/compose/pc-data.yml` defines the following services under the
`pc-prod` and `local-dev` profiles. **Only Qdrant is in the production
path today**; the others are dev-only, legacy, or future-migration
targets.

```mermaid
flowchart LR
    tunnel["cloudflared"]
    qdrant["Qdrant ✅ prod path"]
    neo4j["Neo4j (legacy)"]
    postgres["Postgres (dev)"]
    postgrest["PostgREST (dev)"]
    redis["Redis (dev)"]
    webdis["Webdis (dev)"]
    embed["pc-embed (legacy)"]

    tunnel --> qdrant
    tunnel -.-> neo4j
    tunnel -.-> postgrest
    tunnel -.-> webdis
    tunnel -.-> embed

    postgrest --> postgres
    webdis --> redis
```

| Service | Role | Status in prod | Why |
|---|---|---|---|
| Qdrant | Passage ANN index (`ohi_passages_titan1024`, 1024-dim) | ✅ used | Free-tier HTTP works through CF tunnel; no AWS alternative chosen yet |
| Neo4j 5-community | Graph store | ❌ legacy | Prod uses Aura because bolt (TCP:7687) doesn't go through CF free-tier HTTPS tunnel — planned migration back to PC over **Tailscale** (see §4) |
| pc-embed | `all-MiniLM-L12-v2` HTTP embedder | ❌ legacy | Prod switched to Bedrock Titan v2 for managed availability + 1024-dim parity with the reranker candidate pool |
| Postgres / PostgREST | Relational + REST façade | — dev-only | Never wired into Lambda's verify path |
| Redis / Webdis | Cache + trace store | ❌ disabled | Webdis-over-tunnel doesn't speak native Redis protocol; `REDIS_ENABLED=false` in Lambda. ElastiCache / Upstash migration is backlog |

## 4. Planned changes (not yet shipped)

These are **decided directions that are not reflected in the code or
infra yet** — when implementing, update this section first and then
the rest of the file.

- **Neo4j Aura Pro → PC-local Neo4j 5-community over Tailscale.**
  Driver: Aura's 64 GB RAM limit makes full biomed ingestion
  (PubMed + OpenAlex + enwiki + Wikidata) tight or impossible, and
  the non-profit / intermittent-availability use-case doesn't justify
  the cost. Tailscale solves the CF-free-tier bolt/TCP:7687 exposure
  problem that originally forced the Aura move. `docker/compose/pc-data.yml`
  already builds the `neo4j:5-community` container; the migration
  touches Lambda networking (Tailscale extension / sidecar), the
  `NEO4J_URI` env var, and the dead-adapter degraded-mode behaviour
  (Lambda must return 503 cleanly when the Tailnet peer is offline).
- **OpenAI cc-NLI primary activation.** `CC_NLI_LLM_PROVIDER=openai`
  is set, but `OHI_OPENAI_API_KEY` is currently unset / empty →
  dispatcher falls back to Gemini-as-primary. Activating requires
  populating `ohi/openai-api-key` in Secrets Manager and confirming
  TF plumbs it as `OHI_OPENAI_API_KEY` in Lambda env.
- **Redis / cache migration.** Currently `REDIS_ENABLED=false`.
  ElastiCache-for-Valkey on `cache.t4g.micro` (or Upstash) is the
  target. Unblocks D2 async-verify dedup + L2 claim cache.
- **Corpus ingestion (Wave 3 Phase E).** Qdrant collection
  `ohi_passages_titan1024` and Aura are both empty today. Downloads
  for enwiki, Wikidata, PubMed baseline, OpenAlex, PMC OA, and
  ClinicalTrials are in flight at `/c/ohi-data/`. Post-ingestion, the
  `GraphRetriever` cascade (Qdrant ANN → Aura passage fetch →
  Bedrock rerank) will activate.

## 5. Live env-var reference

The canonical list lives in `infra/terraform/compute/lambda.tf` under
`resource "aws_lambda_function" "api"`. Current values as of the
verified date above:

| Env | Value | Source |
|---|---|---|
| `OHI_ENV` | `prod` | `infra/terraform/compute/lambda.tf` |
| `OHI_REGION` | `eu-central-1` | `var.region` |
| `OHI_EMBEDDING_BACKEND` | `bedrock` | `var.embedding_backend` |
| `BEDROCK_EMBED_MODEL_ID` | `amazon.titan-embed-text-v2:0` | `var.bedrock_embed_model_id` |
| `BEDROCK_EMBED_DIM` | `1024` | `var.bedrock_embed_dim` |
| `BEDROCK_RERANK_ENABLED` | `true` | `var.bedrock_rerank_enabled` |
| `BEDROCK_RERANK_MODEL_ID` | `cohere.rerank-v3-5:0` | `var.bedrock_rerank_model_id` |
| `BEDROCK_RERANK_CANDIDATES` | `40` | `var.bedrock_rerank_candidates` |
| `BEDROCK_RERANK_TOP_N` | `12` | `var.bedrock_rerank_top_n` |
| `NEO4J_URI` | `neo4j+s://0193408e.databases.neo4j.io` | `var.neo4j_uri` |
| `QDRANT_HOST` | `ohi-qdrant.shiftbloom.studio` | `var.tunnel_hostname_qdrant` |
| `QDRANT_PORT` | `443` | constant |
| `QDRANT_HTTPS` | `true` | constant |
| `QDRANT_COLLECTION_NAME` | `ohi_passages_titan1024` | `var.retrieval_qdrant_collection_name` |
| `QDRANT_VECTOR_SIZE` | `1024` | `var.qdrant_vector_size` |
| `LLM_MODEL` | `gemini-3-flash-preview` | `var.gemini_model` |
| `LLM_BASE_URL` | `https://generativelanguage.googleapis.com/v1beta/openai/` | dead config — native Gemini adapter ignores it; only relevant if `LLM_BACKEND=openai` is ever set |
| `NLI_LLM_MODEL` | `gemini-3-pro-preview` | `var.nli_llm_model` |
| `NLI_SELF_CONSISTENCY_K` | `1` | `var.nli_self_consistency_k` |
| `NLI_THINKING_LEVEL` | `HIGH` | `var.nli_thinking_level` |
| `REDIS_ENABLED` | `false` | constant |
| `JOBS_TABLE_NAME` | `ohi-verify-jobs` | `local.jobs_table_name` |
| `OHI_ASYNC_VERIFY_TTL_SECONDS` | `3600` | `var.async_verify_ttl_seconds` |
| `OHI_CORS_ORIGINS` | `https://ohi.shiftbloom.studio` | `var.cors_origins` |
| `LLM_BACKEND` | (unset) | default → `"gemini"` (native adapter) in `src/api/config/dependencies.py:166` |
| `OHI_OPENAI_API_KEY` | (unset / empty) | would come from `ohi/openai-api-key` secret; not currently plumbed into Lambda env |

## 6. Quick-reference: file paths

- Pipeline DI wiring: [src/api/config/dependencies.py](../src/api/config/dependencies.py) (`_initialize_adapters`)
- Embedding adapter tri-mode: [src/api/adapters/embeddings.py](../src/api/adapters/embeddings.py) (`local` / `remote` / `bedrock`)
- Native Gemini adapter: [src/api/adapters/gemini.py](../src/api/adapters/gemini.py)
- Lambda TF: [infra/terraform/compute/lambda.tf](../infra/terraform/compute/lambda.tf)
- Lambda TF vars + defaults: [infra/terraform/compute/variables.tf](../infra/terraform/compute/variables.tf) and [terraform.tfvars](../infra/terraform/compute/terraform.tfvars)
- PC compose: [docker/compose/pc-data.yml](../docker/compose/pc-data.yml)
- Rollback runbook: [docs/runbooks/rollback-deploy.md](runbooks/rollback-deploy.md)

---

*If anything in this file looks wrong, prefer live verification
(`aws lambda get-function-configuration --function-name ohi-api`,
`git log`, `cat docker/compose/pc-data.yml`) over memory. Update this
file and the code in the same commit — stale architecture docs have
burned past sessions multiple times.*
