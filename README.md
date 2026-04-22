<p align="center">
  <img src="https://raw.githubusercontent.com/shiftbloom-studio/open-hallucination-index/main/docs/logo.jpg" alt="Open Hallucination Index" width="765" />
</p>

<h1 align="center">Open Hallucination Index</h1>

<p align="center">
  <strong>Scientifically grounded fact-checking for LLM outputs — with calibrated uncertainty.</strong>
</p>

<p align="center">
  <a href="#current-status">Status</a> •
  <a href="#how-it-works">How It Works</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#getting-started">Getting Started</a> •
  <a href="#documentation">Docs</a> •
  <a href="#contributing">Contributing</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.14+-blue.svg" alt="Python 3.14+" />
  <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="MIT License" />
  <a href="https://github.com/shiftbloom-studio/open-hallucination-index/actions"><img src="https://github.com/shiftbloom-studio/open-hallucination-index/workflows/CI/badge.svg" alt="CI Status" /></a>
</p>

---

**Open Hallucination Index (OHI)** decomposes LLM output into atomic
claims, retrieves real evidence for each, and returns a calibrated
`p_true ∈ [0, 1]` with a confidence interval. The goal is a traceable,
reproducible, open-source alternative to black-box "trust scores" — one
where every verdict is backed by citable evidence and every number has
a published calibration methodology.

The live service is at **<https://ohi.shiftbloom.studio>**.

---

## Current status

OHI is mid-way through its **v2 rework** on `main` (the
`feat/ohi-v2-foundation` branch was merged back into `main` in commit
`c5010cc` — all v2 development now lands on `main` directly). v2
replaces the v1 "trust score" heuristic with a proper probabilistic
pipeline: NLI-based claim verification, Bayesian posterior update via
Probabilistic Claim Graph (PCG), and split-conformal calibration.

**Live today (Wave 2):**

- Async verify flow: `POST /api/v2/verify` → `202` + `job_id`, client
  polls `/api/v2/verify/status/{job_id}` until `status=done` or `error`
- Gemini 3 Pro NLI (with `thinkingLevel=HIGH`, `safetySettings=BLOCK_NONE`)
  over live Wikipedia evidence (MediaWiki MCP)
- Verdict shape: per-claim `p_true`, confidence interval, supporting +
  refuting evidence passages, PCG Beta-posterior update from NLI
- AWS Lambda + API Gateway + Cloudflare WAF + Neo4j Aura + Qdrant-on-PC;
  Next.js 16 static export on Vercel

**In progress (Wave 3, final pre-merge):**

- Full TRW-BP belief propagation + damped LBP fallback + Gibbs MCMC
  sanity in `balanced`+ rigor (replaces the per-claim Beta placeholder)
- Claim-claim NLI via OpenAI GPT-5.4 (`reasoning.effort=xhigh`) with
  Gemini-3-Pro fallback — builds the actual probabilistic *claim graph*
- Full-enwiki + Wikidata corpus ingestion (Aura = content source of
  truth, Qdrant = vector-only passage index)
- Automated merge-to-main gate with post-deploy synthetic probe +
  auto-rollback

**Deferred to v2.1 / Wave 4:**

- Domain routing (v2.0 uses a single `general` stratum)
- Calibration data (v2.0 ships with `fallback_used: "general"` badges;
  intervals will be wide and honest)
- Qdrant-to-AWS migration (EC2 / OpenSearch / Bedrock Knowledge Base —
  sponsorship-pitch-driven choice)

See [docs/superpowers/checkpoints/](docs/superpowers/checkpoints/) for
full state (gitignored — local only; ask for access if needed).

---

## How it works

For every verify request:

1. **Decomposition.** The input text is broken into atomic claims via a
   Gemini 3 Pro call. Each claim gets an SPO shape (subject, predicate,
   object) plus a type classifier (quantitative, temporal, factual,
   etc.).
2. **Evidence retrieval.** Claims are routed through `AdaptiveEvidenceCollector`:
   - **MediaWiki MCP** (live Wikipedia search) for the current v2.0
     baseline
   - Wave 3+: Neo4j Aura graph walks (entity-centered, Wikidata-aligned)
     + Qdrant passage-vector similarity search over a pinned enwiki
     snapshot
3. **Natural-Language Inference (NLI).** Each `(claim, evidence)` pair
   goes to Gemini 3 Pro for a SUPPORT / REFUTE / NEUTRAL classification
   with calibrated scores.
4. **Probabilistic claim graph (PCG).** Wave 3 adds claim-claim NLI
   (OpenAI GPT-5.4) over claim pairs that share entities. The graph
   feeds TRW-BP belief propagation (with damped-LBP fallback + Gibbs
   MCMC sanity) to produce coherent posteriors across claims.
5. **Conformal calibration.** A split-conformal layer produces an
   interval around each `p_true`. Wave 4 adds per-domain calibration
   strata; until then every verdict carries an honest `fallback_used:
   "general"` badge.
6. **Verdict assembly.** Document-level `document_score` is the
   geometric mean of per-claim `p_true`. Full provenance (evidence
   passages with URLs) is included in the response.

See [docs/API.md](docs/API.md) for the complete API surface and response
schemas.

---

## Architecture

```
Browser (static Next.js export)
   │
   │  fetch https://ohi.shiftbloom.studio
   ▼
Vercel CDN
   │
   │  fetch https://ohi-api.shiftbloom.studio/api/v2/*
   ▼
Cloudflare edge            WAF • rate limit • Transform Rule (X-OHI-Edge-Secret)
   │
   ▼
AWS API Gateway (HTTP API, regional custom domain + ACM cert)
   │
   ▼
AWS Lambda (container image, 180s timeout)
   ├─► DynamoDB ohi-verify-jobs (async job state, 1h TTL)
   ├─► Self-invoke (async)   ─► same Lambda runs the pipeline
   ├─► Gemini API (decomposer + claim-evidence NLI)
   ├─► OpenAI API (claim-claim NLI — Wave 3)
   ├─► MediaWiki MCP (live Wikipedia evidence)
   ├─► Neo4j Aura Pro (entity graph + entity-level vector index)
   ├─► Qdrant on PC (passage vectors, vector-only payload)
   └─► AWS Secrets Manager (API keys, edge secret, Aura creds)
```

**Key architectural decisions** (full context in
[CLAUDE.md](CLAUDE.md)):

- **Lambda + API Gateway** over Lambda Function URL, because Function
  URLs strict-check the Host header and Cloudflare's free/pro tier
  can't override it
- **Flat Cloudflare naming** (`ohi-api.shiftbloom.studio` not
  `api.ohi.shiftbloom.studio`) because CF free-tier Universal SSL
  covers only one level of wildcard
- **Neo4j Aura Pro (Frankfurt)** currently hosts the graph — bolt
  (TCP:7687) doesn't work through CF's free-tier HTTPS tunnel, so PC
  Neo4j was ruled out on the first pass. A planned migration back to
  PC-local Neo4j over **Tailscale** is tracked in
  [docs/CURRENT_ARCHITECTURE.md §4](docs/CURRENT_ARCHITECTURE.md#4-planned-changes-not-yet-shipped)
- **Embeddings run on AWS Bedrock Titan Text V2** (1024-dim). Keeps
  the Lambda image slim (~500 MB instead of 2.4 GB) without requiring
  the PC to be online. The `pc-embed` container is retained for local
  dev.
- **Reranking runs on AWS Bedrock Cohere rerank-v3-5** (top-40
  candidates → top-12), slotted between Qdrant ANN and Aura passage
  fetch in the `GraphRetriever` cascade
- **Qdrant stays on PC** as vector-only index with payload
  `{passage_id, qid}` — text content lives in Aura as the single source
  of truth. v1's "redundancy pain" (text duplicated across both stores)
  is eliminated
- **Async polling** over SSE, because SSE would need Lambda Function
  URL's `RESPONSE_STREAM` mode, which CF free-tier can't proxy due to
  the Host-header issue above

---

## Repository structure

```
open-hallucination-index/
├── src/
│   ├── api/                   # FastAPI verification service (v2 rework active)
│   │   ├── adapters/          # Gemini, OpenAI, Qdrant, Neo4j, MediaWiki,
│   │   │                      #   NLI, embeddings, conformal calibrator
│   │   ├── pipeline/          # Decomposer → evidence → NLI → PCG → assembly
│   │   ├── server/            # FastAPI app, routes, middleware
│   │   ├── interfaces/        # Ports (NliAdapter, PcgInferenceService, …)
│   │   └── config/            # Settings, DI container, infra env
│   ├── frontend/              # Next.js 16 static export (Vercel)
│   │   ├── src/app/           # App Router pages
│   │   ├── src/components/    # UI (shadcn/ui + Tailwind)
│   │   └── src/lib/           # ohi-client, verify-controller, types
│   └── ohi-mcp-server/        # (legacy) standalone MCP server; Wave 2 wired
│                              #   MediaWiki directly into src/api
├── gui_ingestion_app/         # (legacy v1) Wikipedia ingestion pipeline.
│                              #   v2 Wave 3 replaces with src/api/ingestion
├── gui_benchmark_app/         # Benchmark suite (Part 3)
├── infra/
│   └── terraform/             # AWS + CF + Vercel infra (layered: bootstrap,
│                              #   storage, secrets, compute, cloudflare,
│                              #   vercel, observability, jobs)
├── docker/
│   ├── lambda/                # Lambda container image build
│   ├── compose/pc-data.yml    # PC-hosted Neo4j/Qdrant/embed stack
│   └── pc-embed/              # MiniLM embedding service
├── tests/                     # pytest: unit/, integration/, infra/
├── docs/
│   ├── API.md                 # Current API surface (v2)
│   ├── FRONTEND.md            # Frontend architecture (polling)
│   ├── runbooks/              # Bootstrap, deploy, rollback, rotation
│   └── superpowers/           # Internal plans/specs/checkpoints (gitignored)
├── .github/workflows/         # CI: plan, apply, release, drift detection
├── CLAUDE.md                  # Operational knowledge for agents + humans
└── LICENSE                    # MIT
```

---

## Getting started

### Prerequisites

- **Python 3.14+** (API + benchmark suite)
- **Node.js 22+** (frontend)
- **Docker Desktop** (Lambda image builds, local PC stack)
- **AWS CLI v2, Terraform 1.14+, jq, gh** for infra work

### Run the backend locally

```bash
cd src/api
python -m venv .venv
source .venv/bin/activate       # Windows Git Bash: source .venv/Scripts/activate
pip install -e ".[dev]"

# Runs FastAPI on http://localhost:8080
ohi-server
```

Tests:

```bash
pytest -q tests -m "not infra" --no-cov   # canonical runline (skips infra tests)
```

### Run the frontend locally

```bash
cd src/frontend
npm install
npm run dev        # http://localhost:3000
npm run test       # vitest
npm run build      # production static export
```

### Local Docker stack (PC data services)

Neo4j, Qdrant, Postgres, Redis (disabled in prod but available locally),
and `pc-embed` embedding service:

```bash
docker compose -f docker/compose/pc-data.yml --profile pc-prod up -d
```

See [docs/runbooks/pc-compose-start.md](docs/runbooks/pc-compose-start.md)
for the PC stack's bring-up details.

### Infrastructure (AWS + Cloudflare + Vercel)

Bootstrap and layered Terraform live in `infra/terraform/`. To spin up a
full clone of the production deployment:

```bash
cd infra/terraform/bootstrap
terraform init && terraform apply
# Then apply layers: storage → secrets → compute → cloudflare → vercel
#                    → observability → jobs
```

Bootstrap runbook: [docs/runbooks/bootstrap-cold-start.md](docs/runbooks/bootstrap-cold-start.md).

---

## Configuration

v2 config flows through AWS Secrets Manager + Lambda env vars. For local
dev or when running the FastAPI app outside Lambda, create
`src/api/.env` with the values below:

```env
# LLM — Gemini native adapter (production path)
LLM_BACKEND=gemini
LLM_API_KEY=<gemini-api-key>
LLM_MODEL=gemini-3-flash-preview
NLI_LLM_MODEL=gemini-3-pro-preview
NLI_THINKING_LEVEL=HIGH
NLI_SELF_CONSISTENCY_K=1

# Wave 3+: claim-claim NLI via OpenAI
CC_NLI_LLM_PROVIDER=openai
CC_NLI_LLM_MODEL=gpt-5.4-xhigh
CC_NLI_LLM_FALLBACK_MODEL=gemini-3-pro-preview
# OHI_OPENAI_API_KEY=<openai-api-key>  # from ohi/openai-api-key secret in prod

# Neo4j Aura (managed graph + vector index)
NEO4J_URI=neo4j+s://<instance-id>.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=<password>

# Qdrant (vector-only index on PC via CF tunnel in prod)
QDRANT_HOST=ohi-qdrant.shiftbloom.studio
QDRANT_PORT=443
QDRANT_HTTPS=true
QDRANT_COLLECTION_NAME=ohi_passages_titan1024
QDRANT_VECTOR_SIZE=1024

# Embeddings — Bedrock Titan v2 in prod (1024-dim).
# Local dev: set OHI_EMBEDDING_BACKEND=local to use in-process
# sentence-transformers, or =remote to call pc-embed over HTTP.
OHI_EMBEDDING_BACKEND=bedrock
BEDROCK_EMBED_MODEL_ID=amazon.titan-embed-text-v2:0
BEDROCK_EMBED_DIM=1024
BEDROCK_EMBED_REGION=eu-central-1

# Reranking — Bedrock Cohere rerank-v3-5
BEDROCK_RERANK_ENABLED=true
BEDROCK_RERANK_MODEL_ID=cohere.rerank-v3-5:0
BEDROCK_RERANK_CANDIDATES=40
BEDROCK_RERANK_TOP_N=12

# Evidence sources
MEDIAWIKI_ENABLED=true
MCP_OHI_ENABLED=false        # legacy MCP server; not wired in v2 prod

# Async verify (DynamoDB polling)
JOBS_TABLE_NAME=ohi-verify-jobs
OHI_ASYNC_VERIFY_TTL_SECONDS=3600

# Edge protection (production)
OHI_CF_EDGE_SECRET=<same-value-injected-by-cloudflare-transform-rule>
```

Full environment reference: [CLAUDE.md](CLAUDE.md) and the compute
Terraform layer's `variables.tf`.

---

## API reference

Full specification with request/response schemas, error codes, and
examples: **[docs/API.md](docs/API.md)**.

Summary:

| Endpoint | Purpose |
|---|---|
| `POST /api/v2/verify` | Submit text, receive `202 Accepted` + `job_id` |
| `GET /api/v2/verify/status/{job_id}` | Poll for pipeline progress + final verdict |
| `GET /health/live` | Liveness probe |
| `GET /health/deep` | Per-layer health (decomposer, Neo4j, Qdrant, embed, PCG version…) |

Authentication: production traffic requires the
`X-OHI-Edge-Secret` header, injected by Cloudflare's Transform Rule on
the `ohi-api.shiftbloom.studio` host. Local dev skips the header.

---

## Documentation

- [docs/API.md](docs/API.md) — v2 API reference (endpoints, schemas,
  polling flow, rigor tiers)
- [docs/FRONTEND.md](docs/FRONTEND.md) — frontend architecture (static
  Next.js export, polling state machine, design system)
- [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md) — contribution process
- [docs/CODE_OF_CONDUCT.md](docs/CODE_OF_CONDUCT.md) — community standards
- [docs/PUBLIC_ACCESS.md](docs/PUBLIC_ACCESS.md) — public access framework
- [docs/runbooks/](docs/runbooks/) — deploy, rollback, secret rotation,
  cold-start bootstrap, PC stack bring-up
- [CLAUDE.md](CLAUDE.md) — operational knowledge (gotchas, Windows
  traps, agent instructions)

Internal planning / specs / checkpoints / stream handoffs live in
`docs/superpowers/` and are gitignored.

---

## Benchmarking

A research-grade benchmark suite comparing OHI against hallucination-
detection baselines is in `gui_benchmark_app/`. The Wave 3 E2E gate adds
an automated FEVER slice run that commits `docs/benchmarks/v2.0-*.md`
per release. Once v2.0 ships, full benchmark methodology and results
will be published there.

---

## Contributing

Contributions welcome. Please read
[docs/CONTRIBUTING.md](docs/CONTRIBUTING.md) for the review process and
conventions, and [docs/CODE_OF_CONDUCT.md](docs/CODE_OF_CONDUCT.md).

1. Fork the repository
2. Create a feature branch (`git checkout -b feat/your-feature`)
3. Commit with Conventional Commits + "end state verified by" lines
   (see CLAUDE.md §Git workflow)
4. Open a pull request against `main` (v2 development lands directly
   on `main` since commit `c5010cc`)

Security issues: see [SECURITY.md](SECURITY.md).

---

## License

MIT — see [LICENSE](LICENSE).

---

## Acknowledgments

- [**FastAPI**](https://fastapi.tiangolo.com/) — Python API framework
- [**Next.js**](https://nextjs.org/) — React framework for the frontend
- [**Gemini** (Google)](https://ai.google.dev/) — decomposer + claim-
  evidence NLI
- [**Neo4j** (Aura)](https://neo4j.com/) — entity graph + vector index
- [**Qdrant**](https://qdrant.tech/) — passage vector store
- [**MCP** (Anthropic)](https://modelcontextprotocol.io/) — Model
  Context Protocol for knowledge-source aggregation
- **AWS** (Lambda, API Gateway, DynamoDB, Secrets Manager, CloudWatch,
  S3, **Bedrock** — Titan Text Embeddings v2 + Cohere rerank-v3-5),
  **Cloudflare** (edge + WAF + tunnel), **Vercel** (frontend hosting)

---

<p align="center">
  <em>Knowledge as a graph, not a list.</em>
</p>
