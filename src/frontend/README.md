# OHI frontend

Next.js 16 App Router app, built as a static export (`output: "export"`) and
served by the Worker in `cloudflare/ohi-worker` from `src/frontend/out`.
There is no Next.js server in production and no API proxy: the browser talks
to `/api/v2` on the same origin.

## Run

```sh
pnpm install
cp .env.example .env.local
pnpm run dev          # http://localhost:3000
```

Point `NEXT_PUBLIC_API_BASE` at a local Worker (`http://localhost:8787/api/v2`)
or at the live API.

## Check

```sh
pnpm run lint
pnpm run test:run     # Vitest
pnpm run build        # static export into out/
pnpm run test:e2e     # Playwright smoke tests, optional
```

## Configuration

All settings are `NEXT_PUBLIC_*` variables read at build time, listed in
[.env.example](.env.example). The legal pages (Impressum, privacy, terms)
take the operator's name and address from `NEXT_PUBLIC_OPERATOR_*`, so real
values stay in the deploy environment.

## Where things are

- `src/app/`: routes (`/verify`, `/status`, `/calibration`, legal pages)
- `src/components/verify/`: verify form, claim cards, evidence and graph views
- `src/lib/ohi-client.ts`: typed API client
- `src/lib/verify-controller.ts`: polling state machine for verify jobs

More detail in [docs/FRONTEND.md](../../docs/FRONTEND.md).
