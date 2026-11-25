<h1 align="center">Open Hallucination Index</h1>

<p align="center">
  <strong>Fact-checking for LLM output, claim by claim, with the evidence attached.</strong>
</p>

<p align="center">
  <a href="https://ohi.shiftbloom.studio">Live</a> ·
  <a href="docs/API.md">API</a> ·
  <a href="docs/CURRENT_ARCHITECTURE.md">Architecture</a> ·
  <a href="CONTRIBUTING.md">Contributing</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/license-AGPL--3.0--or--later-blue.svg" alt="AGPL-3.0-or-later" />
  <a href="https://github.com/shiftbloom-studio/open-hallucination-index/actions/workflows/cloudflare-production.yml"><img src="https://github.com/shiftbloom-studio/open-hallucination-index/actions/workflows/cloudflare-production.yml/badge.svg" alt="CI" /></a>
</p>

---

Paste a paragraph that a model wrote. OHI splits it into single factual
claims, looks for real sources on each one, and tells you how likely each
claim is to be true (`p_true` between 0 and 1, with an interval). Every
number links back to the passages it came from, so you can check the
check.

It runs at **<https://ohi.shiftbloom.studio>** and as an MCP server at
`https://ohi.shiftbloom.studio/mcp`, so an agent can verify its own drafts
before it answers.

## How a request flows

1. **Decompose.** Workers AI breaks the text into atomic claims.
2. **Retrieve.** The Worker pulls candidates from its Vectorize evidence
   cache and live from Wikipedia, Wikidata and up to nine other open sources
   (Crossref, OpenAlex, Europe PMC, ClinicalTrials.gov and more), picked by
   the claim's domain.
3. **Rerank.** A BGE reranker keeps the passages that address the claim.
4. **Classify.** Two models judge each passage as support, refute or
   neutral, and score how relevant it is. When they disagree, the verdict
   moves toward neutral.
5. **Assemble.** Each claim gets `p_true`, an interval and its evidence; the
   document gets an overall score.

Three rigor tiers trade cost for depth: `fast` (4 claims, one model),
`balanced` (7 claims, two models, the default) and `maximum` (13 claims,
more evidence per claim, with a cooldown per IP). Numbers and cost ceilings
are in [docs/CURRENT_ARCHITECTURE.md](docs/CURRENT_ARCHITECTURE.md).

Calibration is honest about itself: until a labeled calibration set ships,
responses carry `fallback_used: "general"` and the interval width is a
heuristic, not a coverage guarantee.

## What it runs on

Everything is one Cloudflare Worker on one domain:

| Piece | Cloudflare product |
|---|---|
| Frontend (Next.js static export) | Workers Static Assets |
| `/api/v2/*`, `/health/*` | Worker |
| `/mcp` | Agents SDK on a Durable Object |
| Async verification | Queues with a dead-letter queue |
| Per-job live state, rate limits | Durable Objects |
| Job history, feedback, corpus metadata | D1 |
| Evidence vectors (BGE-M3, 1024 dims) | Vectorize |
| Raw corpus archive | R2 |
| Corpus seeding | Workflows |
| Decomposition, NLI, embeddings, reranking | Workers AI |
| Bot protection on `/verify` | Turnstile and WAF rules |

## Repository

```
cloudflare/ohi-worker/   Worker: API, queue consumer, MCP, corpus Workflow, D1 migrations
src/frontend/            Next.js 16 frontend, built to a static export
src/ohi-mcp-server/      Standalone MCP server with the knowledge-source connectors
docker/mcp-server/       Container image for the standalone MCP server
docs/                    API reference, architecture, runbooks
```

## Run it locally

You need Node 26 and pnpm (`corepack enable pnpm`).

```sh
# Worker on http://localhost:8787
cd cloudflare/ohi-worker
pnpm install
cp .dev.vars.example .dev.vars
pnpm run dev

# Frontend on http://localhost:3000, talking to the local Worker
cd src/frontend
pnpm install
NEXT_PUBLIC_API_BASE=http://localhost:8787/api/v2 pnpm run dev
```

Local `wrangler dev` simulates D1, R2, Queues, Durable Objects and
Workflows. Vectorize has no local mode, and Workers AI calls go to
Cloudflare when you are logged in, so `/health/deep` reports `degraded`
offline. [CONTRIBUTING.md](CONTRIBUTING.md) lists which commands are safe
and which touch production.

## Deploy your own

1. Create the resources on your Cloudflare account:
   ```sh
   cd cloudflare/ohi-worker
   pnpm wrangler d1 create ohi-prod
   pnpm wrangler r2 bucket create ohi-corpus-prod
   pnpm wrangler vectorize create ohi-evidence-bge-m3 --dimensions=1024 --metric=cosine
   pnpm wrangler queues create ohi-verify
   pnpm wrangler queues create ohi-verify-dlq
   pnpm wrangler queues create ohi-corpus-ingest
   pnpm wrangler queues create ohi-corpus-ingest-dlq
   ```
2. Put the D1 id into `wrangler.jsonc` (or into the `OHI_D1_DATABASE_ID`
   secret for CI), and change `routes` and `SITE_ORIGIN` to your domain.
3. Set the Worker secrets `ADMIN_TOKEN` and `TURNSTILE_SECRET_KEY`.
4. Build the frontend, then deploy:
   ```sh
   cd src/frontend && pnpm install && pnpm run build
   cd ../../cloudflare/ohi-worker
   pnpm wrangler d1 migrations apply ohi-prod --remote
   pnpm run deploy
   ```

The GitHub workflow in `.github/workflows/cloudflare-production.yml` does
the same on every push to `main`. It reads `CLOUDFLARE_API_TOKEN`,
`CLOUDFLARE_ACCOUNT_ID`, `OHI_D1_DATABASE_ID`, `OHI_ADMIN_TOKEN`,
`TURNSTILE_SECRET_KEY` and the `OPERATOR_*` legal-page values from the
`production` environment. Required token scopes are in
[docs/runbooks/cloudflare-api-token-rotate.md](docs/runbooks/cloudflare-api-token-rotate.md).

## API

| Endpoint | Purpose |
|---|---|
| `POST /api/v2/verify` | Submit text, get `202` and a `job_id` |
| `GET /api/v2/verify/status/{job_id}` | Poll until `done` or `error` |
| `GET /api/v2/calibration/report` | Calibration report |
| `POST /api/v2/feedback` | Feedback on a single claim |
| `GET /health/live`, `/health/ready`, `/health/deep` | Health probes |
| `POST /mcp` | MCP over streamable HTTP: `verify_text`, `job_status`, `search_evidence` and the knowledge tools |

Schemas, errors and examples: [docs/API.md](docs/API.md).

## History

OHI started in January 2026 as a Python service on AWS Lambda with Neo4j
and Qdrant. In July 2026 it moved completely to Cloudflare, and the old
stack was later removed from this repository. The old code is still in the git history.

## License

[AGPL-3.0-or-later](LICENSE). If you run a modified version as a service,
publish your changes under the same license.

Security reports go to safety@shiftbloom.studio, see [SECURITY.md](SECURITY.md).
General contact: hi@shiftbloom.studio.

Made by [shiftbloom studio.](https://shiftbloom.studio)
