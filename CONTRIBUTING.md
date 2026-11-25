# Contributing to Open Hallucination Index

Thanks for taking a look. Small, focused pull requests are the easiest to
review, and docs fixes count.

Pick a [good first issue](https://github.com/shiftbloom-studio/open-hallucination-index/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22),
say in a comment that you are on it, and open a PR when you have something.

Please follow the [Code of Conduct](CODE_OF_CONDUCT.md). Security problems go
to safety@shiftbloom.studio, see [SECURITY.md](SECURITY.md).

## Layout

| Path | What it is |
|---|---|
| `cloudflare/ohi-worker/` | Production Worker: API, queue consumer, health probes, MCP server, corpus Workflow |
| `src/frontend/` | Next.js static export served by the Worker |
| `src/ohi-mcp-server/` | Standalone MCP server with the multi-source knowledge tools |
| `docs/` | API reference, architecture, runbooks |

You need Node 26 and pnpm (`corepack enable pnpm`). No Docker, Python or cloud
account is needed for the local loop.

## Worker

This package has its own lockfile. Use `pnpm` from `cloudflare/ohi-worker/` (CI uses Node 26).

```bash
cd cloudflare/ohi-worker
pnpm install
pnpm run types   # wrangler types -> src/worker-configuration.d.ts
pnpm run check   # tsc --noEmit
pnpm run build   # wrangler deploy --dry-run --outdir dist
```

`pnpm run build` compiles the Worker without publishing it. That is the first-evening check.

`pnpm run dev` runs `wrangler dev` (local Miniflare by default). Pair it with the frontend loop below (`NEXT_PUBLIC_API_BASE=http://localhost:8787/api/v2`). `wrangler.jsonc` serves static assets from `src/frontend/out`; build that export if you need the Worker to serve pages.

Do not run these on a first evening (they touch production):

- `pnpm run deploy` / `wrangler deploy`
- `wrangler d1 migrations apply ohi-prod --remote`
- `wrangler secret put`
- `wrangler dev --remote` (uses deployed Cloudflare resources / the production zone edge)

### Bindings and secrets without production credentials

`wrangler.jsonc` does not set `remote: true` on bindings. Local `wrangler dev` therefore uses local simulations for D1 (`OHI_DB`), R2 (`OHI_CORPUS`), Queues, Durable Objects, and Workflows — empty, not a copy of prod. Vectorize (`OHI_VECTOR`) has no local simulation. Workers AI (`AI`) is not a local model; calls go to Cloudflare when you are authenticated.

`ADMIN_TOKEN` and `TURNSTILE_SECRET_KEY` are Worker secrets, not `wrangler.jsonc` vars. Do not put production values in the repo.

Without those secrets, the code fails closed or skips as follows (`cloudflare/ohi-worker/src/index.ts`):

- `ADMIN_TOKEN` missing: `GET/POST /api/v2/admin/*` returns `503` with `admin_token_not_configured`
- `TURNSTILE_SECRET_KEY` missing: `rejectIfAbusive` skips the Turnstile check and does not challenge the request
- `GET /health/live`: no bindings required
- `GET /health/ready`: probes local D1 and Durable Objects; it marks Workers AI and Vectorize as `configured` without calling them
- `GET /health/deep`: actually calls Workers AI, Wikipedia, and Vectorize — expect `degraded` locally when Vectorize/AI are not remote

Do not publish secrets. Do not try to bypass the production WAF or Turnstile.

## Frontend

```bash
cd src/frontend
pnpm install
cp .env.example .env.local   # point NEXT_PUBLIC_API_BASE at http://localhost:8787/api/v2
pnpm run dev                 # http://localhost:3000
pnpm run lint
pnpm run test:run
pnpm run build               # static export into src/frontend/out
```

## MCP server

```bash
cd src/ohi-mcp-server
npm install
npm run build
```

## Pull requests

- Branch from `main` and keep one topic per PR.
- Run the checks of the package you touched (lint, tests, typecheck, build).
  CI runs the same set on every PR.
- Write commit messages in English, in the imperative ("Fix claim span offsets").
- Do not commit secrets, account ids or personal data. Config goes into the
  `.example` files with placeholders.

By contributing you agree that your work is released under the
[AGPL-3.0-or-later](LICENSE) license of this project.
