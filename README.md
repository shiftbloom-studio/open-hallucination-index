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

OHI is live as a **Cloudflare-native deployment** at
<https://ohi.shiftbloom.studio>. The production path is fully hosted on
Cloudflare and does not depend on local tunnels, local compute, Vercel,
AWS Lambda, API Gateway, DynamoDB, Bedrock, Neo4j, or Qdrant.

**Live today:**

- Same-origin frontend, API, health checks, and MCP server on one
  Cloudflare Worker custom domain.
- Next.js 16 static export served by Cloudflare Worker Static Assets.
- Async verify flow: `POST /api/v2/verify` returns `202 + job_id`;
  clients poll `/api/v2/verify/status/{job_id}` until `done` or `error`.
- Cloudflare Queue consumer runs the hosted verification pipeline.
- Durable Objects hold per-job live state and the MCP server.
- D1 stores job mirrors, feedback, and evidence cache rows.
- Vectorize stores/retrieves evidence vectors.
- Workers AI handles claim decomposition, NLI, embeddings, and reranking.
- Public MCP endpoint at `/mcp` exposes `verify_text`, `job_status`, and
  `search_evidence`.

---

## How it works

For every verify request:

1. **Decomposition.** Workers AI breaks input text into atomic factual
   claims.
2. **Evidence retrieval.** The Worker combines Vectorize evidence cache
   hits with live Wikipedia and Wikidata retrieval.
3. **Reranking.** Workers AI BGE reranker orders candidate evidence.
4. **Natural-Language Inference (NLI).** Workers AI classifies evidence
   as SUPPORT / REFUTE / NEUTRAL, with a deterministic fallback for
   debunking cues and lexical overlap.
5. **Verdict assembly.** The Worker returns per-claim `p_true`,
   confidence intervals, supporting/refuting evidence, and a document
   score. Current calibration is marked with `fallback_used: "general"`
   until a larger calibration set is shipped.

See [docs/API.md](docs/API.md) for the complete API surface and response
schemas.

---

## Architecture

```
Browser
   │
   │  https://ohi.shiftbloom.studio
   ▼
Cloudflare Worker custom domain
   ├─► Worker Static Assets (Next.js export)
   ├─► /api/v2/* verification API
   ├─► /health/* probes
   ├─► /mcp streamable HTTP MCP server
   ├─► Durable Objects (jobs + MCP)
   ├─► Queues (async verification + DLQ)
   ├─► D1 (jobs, feedback, evidence cache)
   ├─► Vectorize (BGE-M3 evidence vectors)
   ├─► Workers AI (Gemma 3, BGE-M3, BGE reranker)
   └─► Wikimedia APIs (Wikipedia + Wikidata evidence)
```

See [docs/CURRENT_ARCHITECTURE.md](docs/CURRENT_ARCHITECTURE.md) for
the production SSoT.

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
│   ├── frontend/              # Next.js 16 static export (Cloudflare assets)
│   │   ├── src/app/           # App Router pages
│   │   ├── src/components/    # UI (shadcn/ui + Tailwind)
│   │   └── src/lib/           # ohi-client, verify-controller, types
│   └── ohi-mcp-server/        # Legacy standalone MCP package
├── cloudflare/
│   └── ohi-worker/            # Production Cloudflare Worker/API/MCP deployment
├── gui_ingestion_app/         # (legacy v1) Wikipedia ingestion pipeline.
│                              #   v2 Wave 3 replaces with src/api/ingestion
├── gui_benchmark_app/         # Benchmark suite (Part 3)
├── infra/                     # Legacy Terraform from the pre-Cloudflare stack
├── docker/
│   ├── compose/pc-data.yml    # Legacy/local data stack
│   └── pc-embed/              # Local MiniLM embedding service
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
- **pnpm** (frontend and Cloudflare Worker packages)
- **Wrangler** (installed as a dev dependency in `cloudflare/ohi-worker`)
- **Docker Desktop** only for the legacy/local data stack

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
pnpm install
pnpm run dev        # http://localhost:3000
pnpm run test:run   # vitest
pnpm run build      # production static export
```

### Local Docker stack (PC data services)

Neo4j, Qdrant, Postgres, Redis (disabled in prod but available locally),
and `pc-embed` embedding service:

```bash
docker compose -f docker/compose/pc-data.yml --profile pc-prod up -d
```

See [docs/runbooks/pc-compose-start.md](docs/runbooks/pc-compose-start.md)
for the PC stack's bring-up details.

### Deploy to Cloudflare

Production deployment lives in `cloudflare/ohi-worker/`.

```bash
cd src/frontend
NEXT_PUBLIC_API_BASE=https://ohi.shiftbloom.studio/api/v2 \
NEXT_PUBLIC_SITE_URL=https://ohi.shiftbloom.studio \
pnpm run build

cd ../../cloudflare/ohi-worker
pnpm install
pnpm run types
pnpm run check
pnpm run build
pnpm run deploy
```

Apply D1 migrations when `cloudflare/ohi-worker/migrations/` changes:

```bash
cd cloudflare/ohi-worker
pnpm exec wrangler d1 migrations apply ohi-prod --remote
```

---

## Configuration

Cloudflare production bindings and variables are in
`cloudflare/ohi-worker/wrangler.jsonc`.

- `ENVIRONMENT=production`
- `SITE_ORIGIN=https://ohi.shiftbloom.studio`
- `VERIFY_MAX_CLAIMS=8`
- `AI` binding for Workers AI
- `OHI_DB` binding for D1 `ohi-prod`
- `OHI_VECTOR` binding for Vectorize `ohi-evidence-bge-m3`
- `VERIFY_QUEUE` binding for Queue `ohi-verify`
- Durable Object bindings `JOBS` and `MCP_OBJECT`

Frontend production public env:

```env
NEXT_PUBLIC_API_BASE=https://ohi.shiftbloom.studio/api/v2
NEXT_PUBLIC_SITE_URL=https://ohi.shiftbloom.studio
```

---

## API reference

Full specification with request/response schemas, error codes, and
examples: **[docs/API.md](docs/API.md)**.

Summary:

| Endpoint | Purpose |
|---|---|
| `POST /api/v2/verify` | Submit text, receive `202 Accepted` + `job_id` |
| `GET /api/v2/verify/status/{job_id}` | Poll for pipeline progress + final verdict |
| `GET /api/v2/calibration/report` | Public calibration report |
| `POST /api/v2/feedback` | Submit claim feedback |
| `GET /health/live` | Liveness probe |
| `GET /health/ready` | Binding readiness probe |
| `GET /health/deep` | Per-layer health for decomposition, Wikimedia, Vectorize, and NLI |
| `POST /mcp` | Streamable HTTP MCP endpoint |

The public Cloudflare deployment currently does not require API keys for
the verification endpoints. Add rate limiting or Turnstile before
opening high-volume public traffic.

---

## Documentation

- [docs/API.md](docs/API.md) — v2 API reference (endpoints, schemas,
  polling flow, rigor tiers)
- [docs/FRONTEND.md](docs/FRONTEND.md) — frontend architecture (static
  Next.js export, polling state machine, design system)
- [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md) — contribution process
- [docs/CODE_OF_CONDUCT.md](docs/CODE_OF_CONDUCT.md) — community standards
- [docs/PUBLIC_ACCESS.md](docs/PUBLIC_ACCESS.md) — public access framework
- [docs/runbooks/](docs/runbooks/) — legacy and operational runbooks
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
- [**Cloudflare Workers**](https://developers.cloudflare.com/workers/) — hosted frontend, API, health, and MCP runtime
- [**Cloudflare Workers AI**](https://developers.cloudflare.com/workers-ai/) — decomposer, NLI, embeddings, and reranker
- [**Cloudflare D1**](https://developers.cloudflare.com/d1/) — job mirror, feedback, and evidence cache
- [**Cloudflare Vectorize**](https://developers.cloudflare.com/vectorize/) — evidence vector index
- [**Cloudflare Queues**](https://developers.cloudflare.com/queues/) — async verification jobs
- [**MCP** (Anthropic)](https://modelcontextprotocol.io/) — Model
  Context Protocol for knowledge-source aggregation

---

<p align="center">
  <em>Knowledge as a graph, not a list.</em>
</p>
