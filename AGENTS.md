# Notes for coding agents and new maintainers

Read [docs/CURRENT_ARCHITECTURE.md](docs/CURRENT_ARCHITECTURE.md) first. It
describes what runs in production and wins over any other doc.

## Layout

- `cloudflare/ohi-worker/`: the production Worker. Entry `src/index.ts`
  (routing, verification pipeline, NLI ensemble, Turnstile and admin auth),
  `src/knowledge-tools.ts` (evidence connectors shared by the pipeline and
  MCP), `src/corpus-ingestion.ts` (the `ohi-corpus-ingestion` Workflow),
  `migrations/` (D1).
- `src/frontend/`: Next.js static export. API client in
  `src/lib/ohi-client.ts`, polling state machine in
  `src/lib/verify-controller.ts`. Legal pages read operator details from
  `NEXT_PUBLIC_OPERATOR_*` at build time (`src/lib/operator.ts`).
- `src/ohi-mcp-server/`: standalone MCP server with the knowledge sources.

## Checks

```sh
cd cloudflare/ohi-worker && pnpm install && pnpm run types && pnpm run check && pnpm run build
cd src/frontend && pnpm install && pnpm run lint && pnpm run test:run && pnpm run build
cd src/ohi-mcp-server && npm install && npm run lint && npm run typecheck
```

`pnpm run build` in the Worker is `wrangler deploy --dry-run`. It catches
config errors that `tsc` does not, so keep it in the loop.

## Rules

- Never commit secrets, account ids, resource ids or personal data. Config
  lives in `.example` files with placeholders; real values are GitHub
  environment secrets or Worker secrets.
- Do not run anything that touches production (`wrangler deploy`,
  `d1 migrations apply --remote`, `wrangler secret put`, `wrangler dev
  --remote`) unless the maintainer asked for it.
- English only for code, docs and commit messages.

## Choices worth knowing before you change them

1. One Worker serves frontend, API and MCP on one domain. Same origin
   means no CORS or Host header trouble.
2. D1, Vectorize and R2 are the whole corpus store. There is no external
   graph or vector database.
3. NLI is an ensemble of two Workers AI models run in parallel. When they
   disagree, the label moves toward neutral or refute. A separate
   `relevance_score` decides whether a passage counts at all. This fixed
   false claims scoring around 70 % true because one loosely related
   "support" passage dominated `p_true`.
4. Turnstile plus zone WAF and rate-limit rules protect public traffic.
   There are no user accounts.
5. Corpus growth runs as a Workflow so long fetch chains retry and resume.

## Traps

- CI overwrites the Worker secrets `ADMIN_TOKEN` and `TURNSTILE_SECRET_KEY`
  from the GitHub secrets `OHI_ADMIN_TOKEN` and `TURNSTILE_SECRET_KEY` on
  every green deploy. Rotate both places or the old value comes back.
- Plain `curl` against `/api/v2/verify` and `/mcp` hits the WAF challenge
  for scripted clients. That is intended. Admin routes are not behind it.
- Queue consumers must stay idempotent per `(run_id, batch)` or per job id.
  Delivery is at least once.
