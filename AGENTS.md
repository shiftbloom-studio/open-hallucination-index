# Open Hallucination Index - OpenCode Agent Guide

This guide is for OpenCode (and similar agentic coding tools) working in this monorepo.
Goal: fast orientation, correct dev loops, and safe operations across API, frontend, MCP server,
Docker stack, Nginx, and the two standalone GUI apps.

## Golden rules (do these first)

- Never commit or paste secrets: `.env*`, `docker/compose/.env`, API keys, tunnel tokens, DB URLs, `*.key`, `*.pem`.
- Avoid printing expanded envs.
  - `docker compose ... config` expands secrets; use `--quiet` or `--no-interpolate`.
- Do not delete persistent data in `docker/data/` unless explicitly requested.
- Prefer local-first workflows for feature work; use full Docker only for end-to-end validation.

## Fast mental model (what talks to what)

- Backend API base prefix: `/api/v1` (usually protected via `X-API-Key` when `API_API_KEY` is set).
- Frontend uses a server-side proxy route:
  - `src/frontend/src/app/api/ohi/[...path]/route.ts` handles `/api/ohi/*`
  - It forwards to `DEFAULT_API_URL` and injects `X-API-Key` from `DEFAULT_API_KEY`.
- Nginx routing (see `docker/nginx/nginx.conf`):
  - Local dev: `/` -> Next.js, `/api/` -> FastAPI, `/health` -> FastAPI
  - Production: most traffic -> Next.js; health -> FastAPI; Cloudflare tunnel hits Nginx port 8080 internally
- API URL conventions:
  - Local Nginx allows `http://localhost/api/v1/*` (direct FastAPI proxy)
  - Public/prod prefers `https://<domain>/api/ohi/api/v1/*` (Next.js proxy)

## Repo map (what lives where)

- `src/api/` - FastAPI verification API (Python)
- `src/frontend/` - Next.js App Router frontend (TypeScript)
- `src/ohi-mcp-server/` - MCP server aggregating external knowledge sources (Node/TypeScript)
- `gui_ingestion_app/` - Ingestion GUI wrapper (runs `ingestion.gui_app`)
- `gui_ingestion_app/ingestion/` - Wikipedia ingestion pipeline package (CLI + GUI)
- `gui_benchmark_app/` - Benchmark GUI wrapper (runs `benchmark.gui_app`)
- `gui_benchmark_app/benchmark/` - Benchmark suite package (CLI + GUI)
- `docker/compose/docker-compose.yml` - Full stack orchestration
- `docker/nginx/nginx.conf` - Active Nginx config (dev + prod + Cloudflare tunnel)
- `docker/api/Dockerfile` - API image
- `docker/mcp-server/Dockerfile` - MCP server image
- `src/frontend/Dockerfile` - Frontend image

## Where to look first (high-signal entrypoints)

- API
  - App factory + middleware: `src/api/server/app.py`
  - Settings (env var mapping): `src/api/config/settings.py`
  - Dependency wiring (DI): `src/api/config/dependencies.py`
  - Routes: `src/api/server/routes/`
- Frontend
  - App routes: `src/frontend/src/app/`
  - API proxy: `src/frontend/src/app/api/ohi/[...path]/route.ts`
  - API client wrapper: `src/frontend/src/lib/api.ts`
- MCP server
  - Entry: `src/ohi-mcp-server/src/index.ts`
  - Source adapters: `src/ohi-mcp-server/src/sources/`
- Infra
  - Compose: `docker/compose/docker-compose.yml`
  - Nginx: `docker/nginx/nginx.conf`

## Ports and URLs (default Docker Compose)

- Nginx: `http://localhost` (80), `https://localhost` (443)
- API direct: `http://127.0.0.1:8080` (FastAPI)
- Frontend: container-only `3000` (use Nginx)
- MCP server: `http://127.0.0.1:8083` -> container `8080`
- vLLM: `http://127.0.0.1:8000` (GPU)
- Neo4j: `http://127.0.0.1:7474`, bolt `127.0.0.1:7687`
- Qdrant: `http://127.0.0.1:6333`, gRPC `127.0.0.1:6334`
- Redis: `127.0.0.1:6379`

Health endpoints:

- API: `GET /health/live`, `GET /health/ready`
- MCP server: `GET /health`
- vLLM: `GET /health`
- Nginx convenience: `GET http://localhost/health` (rewrites to API live probe)

## Environment files and precedence

Templates:

- Root template: `.env.example` (full stack)
- API-only template: `src/api/.env.example`
- Evidence classification profiles: `docker/compose/.env.classification-examples`

Local files (ignored by git):

- Root `.env` - main local configuration source
- `docker/compose/.env` - optional compose-local overrides; `docker/compose/docker-compose.yml` loads both, and root `.env` wins for overlapping keys on services that list both.
- `src/frontend/.env.local` - local frontend dev config (do not commit)

Security gotchas:

- `docker/nginx/certs/` contains TLS keys/certs for Cloudflare origin/mTLS; these are ignored by git but still sensitive.
- Prefer `docker compose -f docker/compose/docker-compose.yml config --quiet` when validating compose.
  - If you need to inspect the rendered config, use `--no-interpolate`.

## Recommended dev workflows

### 1) Local-first (recommended for feature work)

Use Docker only for stateful infra, run app code locally.

- Start infra (Neo4j + Qdrant + Redis + MCP server):
  - `docker compose -f docker/compose/docker-compose.yml up -d neo4j qdrant redis ohi-mcp-server`
- Run API locally (repo root venv recommended):
  - `python -m venv .venv`
  - Windows: `.venv\Scripts\activate`
  - `pip install -e "src/api[dev]"`
  - `ohi-server`
- Run frontend locally:
  - `cd src/frontend && npm install && npm run dev`
  - Create `src/frontend/.env.local` with:
    - `DEFAULT_API_URL=http://localhost:8080`
    - `DEFAULT_API_KEY=...` (same key as API)

When you do this, you typically browse `http://localhost:3000` (Next dev server) and skip Nginx.

### 2) Full Docker stack (end-to-end)

- Start everything:
  - `docker compose -f docker/compose/docker-compose.yml up -d --build`

GPU note:

- `vllm` requires NVIDIA GPU + Docker GPU support (typically WSL2 + NVIDIA Container Toolkit on Windows).
- vLLM can take 30-60s to load models; API waits via health checks.
- If you do not have a working GPU stack, do not try to run `vllm`; run the API locally and point `LLM_BASE_URL` to an external OpenAI-compatible endpoint.

### 3) Standalone GUI apps (Windows-oriented)

Both GUI launch scripts assume a repo-root venv at `.venv`.

- Ingestion GUI:
  - `gui_ingestion_app/launch_gui.ps1`
  - or `pip install -e "gui_ingestion_app/ingestion[dev]"` then `ohi-ingestion-gui`
- Benchmark GUI:
  - `gui_benchmark_app/launch_gui.ps1`
  - or `pip install -e "gui_benchmark_app/benchmark[dev]"` then `ohi-benchmark-gui`

## Build/lint/test (per subproject)

API - `src/api` (Python, strict mypy + Ruff):

- `pip install -e "src/api[dev]"`
- `ruff format .`
- `ruff check .`
- `mypy .`
- `pytest`

Frontend - `src/frontend` (Next.js):

- `npm install`
- `npm run lint`
- `npm run test:run`
- `npm run build` (for production regressions)
- `npm run test:e2e` (only if routes/auth/proxy changed)

MCP server - `src/ohi-mcp-server`:

- `npm install`
- `npm run lint`
- `npm run typecheck`
- `npm run build`

Ingestion - `gui_ingestion_app/ingestion`:

- `pip install -e "gui_ingestion_app/ingestion[dev]"`
- `ruff format . && ruff check . && mypy . && pytest`

Benchmark - `gui_benchmark_app/benchmark`:

- `pip install -e "gui_benchmark_app/benchmark[dev]"`
- `ruff format . && ruff check . && mypy . && pytest`

## Debug checklist (quick)

- API up?
  - `GET http://127.0.0.1:8080/health/ready`
- Nginx routing ok?
  - `GET http://localhost/health`
- MCP up?
  - `GET http://127.0.0.1:8083/health`
- Frontend -> API proxy ok?
  - Verify `DEFAULT_API_URL` and `DEFAULT_API_KEY` in `src/frontend/.env.local` (local dev) or in the Docker env.
- Getting 401/403 from API?
  - Confirm `API_API_KEY` (backend) and `DEFAULT_API_KEY` (frontend proxy) match.
  - If calling FastAPI directly (not via `/api/ohi/*`), include `X-API-Key: <API_API_KEY>`.
- Knowledge-track missing?
  - Redis disabled means knowledge-track is disabled (expected behavior).

## Ingestion guardrails

- Always test ingestion with a small limit first (example: `python -m ingestion --limit 1000`).
- Embedding consistency must match between ingestion and API:
  - `EMBEDDING_MODEL_NAME` must match the model used during ingestion
  - `QDRANT_VECTOR_SIZE` must match embedding dimension (384 for `all-MiniLM-L12-v2`)

## Docs for deeper context

- `docs/CONTRIBUTING.md` - conventions and PR hygiene
- `docs/API.md` - API endpoints + request/response schemas
- `docs/FRONTEND.md` - frontend UX/data flow principles
- `docker/README.md` - stack overview and ops notes
- `src/api/README.md`, `src/frontend/README.md`, `src/ohi-mcp-server/README.md` - subsystem details
