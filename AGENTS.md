# Agent Guide for Open Hallucination Index

This guide is for agentic coding tools working in this monorepo. It covers how to build,
lint, test, and follow the code style rules for each subproject.

## Repository layout

- `src/api`: FastAPI backend (Python)
- `src/frontend`: Next.js App Router frontend (TypeScript)
- `src/ohi-mcp-server`: MCP server (Node/TypeScript)
- `gui_benchmark_app/benchmark`: Benchmark suite (Python)
- `gui_ingestion_app/ingestion`: Wikipedia ingestion pipeline (Python)
- `docs/`: Project documentation and contributing guide

## Cursor/Copilot rules

- No `.cursor/rules`, `.cursorrules`, or `.github/copilot-instructions.md` found.

## Build, lint, and test commands

### API (FastAPI) - `src/api`

- Install deps: `pip install -e "src/api[dev]"`
- Run server: `ohi-server`
- Lint: `ruff check .`
- Format: `ruff format .`
- Type check: `mypy .`
- Run all tests: `pytest` (coverage and `tests/` config live in `src/api/pyproject.toml`)
- Run a single test file: `pytest tests/test_example.py -v`
- Run a single test by name: `pytest tests/test_example.py::test_name -v`

### Frontend (Next.js) - `src/frontend`

- Install deps: `npm install`
- Dev server: `npm run dev`
- Build: `npm run build`
- Lint: `npm run lint`
- Unit tests (watch): `npm run test`
- Unit tests (single run): `npm run test:run`
- Run a single test file: `npx vitest run src/app/page.test.tsx`
- Run a single test by name: `npx vitest run -t "renders hero"`
- E2E tests: `npm run test:e2e`
- Run a single Playwright spec: `npx playwright test e2e/auth.spec.ts`

### MCP server - `src/ohi-mcp-server`

- Install deps: `npm install`
- Dev server: `npm run dev`
- Build: `npm run build`
- Start built server: `npm run start`
- Lint: `npm run lint`
- Type check: `npm run typecheck`

### Benchmark suite - `gui_benchmark_app/benchmark`

- Install deps: `pip install -e "gui_benchmark_app/benchmark/[dev]"`
- Run CLI benchmark: `python -m benchmark`
- Run GUI: `ohi-benchmark-gui`
- Lint: `ruff check .`
- Format: `ruff format .`
- Type check: `mypy .`
- Run all tests: `pytest`
- Run a single test file: `pytest tests/test_config.py -v`
- Run a single test by name: `pytest tests/test_config.py::test_config_defaults -v`

### Ingestion pipeline - `gui_ingestion_app/ingestion`

- Install deps: `pip install -e "gui_ingestion_app/ingestion/[dev]"`
- CLI help: `python -m ingestion --help`
- Run GUI: `ohi-ingestion-gui`
- Lint: `ruff check .`
- Format: `ruff format .`
- Type check: `mypy .`
- Run all tests: `pytest`
- Run a single test file: `pytest tests/test_pipeline.py -v`
- Run a single test by name: `pytest tests/test_pipeline.py::test_chunking -v`

## Code style and conventions

### Python (API, benchmark, ingestion)

- Formatting and linting: Ruff is the source of truth.
- Line length: 100 characters.
- Quotes: prefer double quotes for strings.
- Imports: sorted and grouped (Ruff isort rules).
- Type hints: required on all function signatures.
- Forward refs: use `from __future__ import annotations`.
- Union types: prefer `X | None` over `Optional[X]`.
- Docstrings: Google-style for public APIs and non-trivial logic.
- Naming:
  - Modules/files: `snake_case.py`.
  - Functions/vars: `snake_case`.
  - Classes/types: `PascalCase`.
  - Constants: `SCREAMING_SNAKE_CASE`.
- Error handling:
  - Avoid bare `except`; catch specific exceptions.
  - Preserve tracebacks when rethrowing (`raise ... from exc`).
  - Use domain-specific exceptions for business logic.
- Async:
  - Prefer async I/O in adapters (httpx, async DB clients).
  - Do not block the event loop with CPU-heavy tasks.

### FastAPI-specific

- Keep hexagonal boundaries: domain is pure, adapters do I/O.
- Put business rules in services/use-cases, not in routes.
- Validate request/response models with Pydantic.
- Use explicit response models for public endpoints.
- Avoid logic in dependencies that mutates global state.

### TypeScript/Next.js (frontend)

- Use TypeScript everywhere; avoid `any` and unsafe casts.
- Prefer `type` for object shapes, `interface` for public/extensible contracts.
- Co-locate types with components when they are local.
- Keep server components default; add `"use client"` only when required.
- Prefer React Query for client-side data caching.
- Use Zod for validation at boundaries.
- Naming:
  - Components: `PascalCase` in `*.tsx`.
  - Hooks: `useThing` prefix.
  - Files: `kebab-case` or `camelCase` aligned with existing folder.
- Error handling:
  - Use `try/catch` for server actions with typed error returns.
  - Prefer user-friendly error messages in UI and log details server-side.

### TypeScript (MCP server)

- ESM modules (`type": "module"` in package.json).
- Prefer `async`/`await` for async flows.
- Validate external inputs with Zod.
- Keep adapters/source integrations small and composable.
- Use explicit return types on exported functions.

## Testing conventions

- Keep unit tests close to the component or module being tested.
- Prefer focused tests over large integration fixtures.
- Name tests by behavior and expected outcome.
- For API tests, prefer deterministic fixtures and mocked I/O.

## Commit/PR guidance

- Conventional commits are used (see `docs/CONTRIBUTING.md`).
- Commit types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`.
- Provide short, intent-focused commit messages.

## Operational notes

- Primary environment is Docker on Windows with containers running locally.
- On macOS, some services (for example `vllm`) do not run; work without running Docker tests.
- API depends on Neo4j, Qdrant, Redis, and LLM services.
- Frontend proxy uses `DEFAULT_API_URL` and `DEFAULT_API_KEY`.
- MCP server exposes `/health`, `/sse`, `/messages`, `/stats`.
- Docker compose stack lives in `docker/compose/docker-compose.yml`.
