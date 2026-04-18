# Open Hallucination Index — Frontend Documentation (v2)

> **Status:** Next.js 16 static export deployed to Vercel at
> `https://ohi.shiftbloom.studio`. Talks to the OHI API at
> `https://ohi-api.shiftbloom.studio` via the polling verify flow (Wave 2
> Stream D2 swapped SSE for polling). Supabase / Stripe integration is
> **not** part of v2; those were v1 aspirational components and are not
> wired in the current deployment. This doc reflects the current static-
> export reality.

---

## Deployment

- **Hosting:** Vercel static export (Next.js 16, `output: 'export'`).
- **Domain:** `ohi.shiftbloom.studio` (apex CNAME to Vercel via
  Cloudflare DNS-only record).
- **Build trigger:** auto-deploy on push to `main`. Preview builds on
  PRs are enabled but not gated (Decision D: pre-deploy preview checks
  de-emphasised, trust real-prod smoke).
- **Env vars on Vercel** (public):
  - `NEXT_PUBLIC_API_BASE=https://ohi-api.shiftbloom.studio/api/v2`
  - `NEXT_PUBLIC_SITE_URL=https://ohi.shiftbloom.studio`

---

## Architecture (v2)

The frontend is a **single-page Next.js app that renders statically**,
with all dynamic behaviour client-side. No server components are used in
prod — the `output: 'export'` configuration produces a purely static
bundle Vercel serves from its CDN.

Verify flow lives entirely in the browser:

```
┌────────────────────────────────┐
│        Browser (client)        │
│                                │
│  User input                    │
│      │                         │
│      ▼                         │
│  verifyReducer (state)         │
│      │                         │
│      ▼                         │
│  verify-controller             │
│   ├─ ohi.verify(text)          │  POST /api/v2/verify
│   │   returns { job_id }       │
│   └─ ohi.verifyStatus(job_id)  │  GET /api/v2/verify/status/{id}
│       polled every ~1s         │    (with exp backoff on errors)
│                                │
│  state transitions:            │
│    idle → submitting           │
│    submitting → polling        │
│    polling → done | error      │
│                                │
└────────────────────────────────┘
              │
              ▼  fetch, cross-origin
┌────────────────────────────────┐
│   Cloudflare edge              │  WAF + rate limit + edge secret
└────────────────────────────────┘
              │
              ▼
┌────────────────────────────────┐
│   AWS API Gateway + Lambda     │  async verify, DynamoDB job state
└────────────────────────────────┘
```

---

## Primary pages

| Route | Purpose | State |
|---|---|---|
| `/` | Landing page — problem, how-it-works, CTA to `/verify` | Static |
| `/verify` | Main verify flow — input form, progress, verdict card, evidence drill-down | Client-side reducer (`verifyReducer`) |
| `/status` | Public status page — pulls `/health/deep` and renders layer health | Client-side fetch |
| `/about`, `/disclaimer`, `/agb`, `/datenschutz`, `/impressum`, `/eula`, `/cookies`, `/accessibility`, `/calibration` | Legal + informational | Static |

All pages pre-render at build time (16 static routes per the current
Next.js build output).

---

## `src/frontend/src/lib/` — key modules

- **`ohi-client.ts`** — thin API client.
  - `ohi.verify(text, opts)` → `{ job_id }` (POST `/verify`, 202).
  - `ohi.verifyStatus(job_id)` → `VerifyStatusResponse` (GET
    `/verify/status/{id}`).
  - `ohi.healthDeep()` → `HealthDeepResponse`.
- **`verify-controller.ts`** — client state machine.
  - Reducer pattern; actions include `SUBMIT`, `POLL_UPDATE`,
    `TERMINAL_DONE`, `TERMINAL_ERROR`, `RESET`.
  - Poll loop with ~1 s cadence; exponential backoff on transient
    network errors (cap 5 s); hard timeout ~3 min to bail on a Lambda
    that ran past its 180 s ceiling.
  - `submitSync` (legacy name kept) dispatches a fresh submit, used by
    the "Retry" button on the error screen.
- **`ohi-types.ts`** — TypeScript types mirroring `DocumentVerdict`,
  `VerifyStatusResponse`, `HealthDeepResponse`. Legacy `SseEvent` types
  still present but unused (D2 deleted `sse.ts` but left these for a
  later frontend-overhaul sweep).
- **`sse.ts`** — **deleted in Stream D2**. Do not re-introduce without
  revisiting the routing decision (CF free-tier cannot proxy Lambda
  Function URL → RESPONSE_STREAM; polling is the pragmatic path).

---

## Design system

- **Styling:** Tailwind CSS 4 + shadcn/ui components.
- **Theme:** light-mode (post-overhaul, matches shiftbloom.studio
  brand palette).
- **Typography hierarchy:** display serif for hero headlines + monospace
  tracked-uppercase for metadata labels + near-black body text.
- **Color tokens (semantic):**
  - `bg-base` — light lavender wash
  - `bg-surface` — white cards with soft gradient
  - `accent-primary` — indigo (`#6b5ce7` range)
  - `accent-danger` — red (`#e8554d` range) for refuted/contradicted
    states
  - `text-base` / `text-muted` — near-black / mid-grey

---

## State machine — verify page

```
  idle
   │
   │  user submits text
   ▼
  submitting ───┐
   │            │ POST error (5xx / network)
   │            ▼
   │          error (retryable)
   │
   │  202 + job_id
   ▼
  polling
   │   ┌───── GET /status returns {status: "pending", phase}
   │   │     → POLL_UPDATE, stay in polling
   │   └─────┐
   │         │
   │         │  phase badge re-renders
   │         │
   │   GET /status returns {status: "done", result}
   ▼
  done (renders DocumentVerdict + claims + evidence drill-down)
   │
   │  OR GET /status returns {status: "error", error}
   ▼
  error (with retry button that re-dispatches submitSync)
```

Poll cadence: 1 s; exponential backoff 1→2→4→5 s on transient
failures; give up at ~180 polls (3 min wall-clock).

---

## Claim verdict UI

Each claim in the `DocumentVerdict` renders with:

- Claim text + SPO decomposition (subject, predicate, object).
- `p_true` badge with value + confidence interval bracket.
- `GENERAL FALLBACK` badge when `fallback_used === "general"`
  (expected in v2.0 because calibration data is Wave 4 / v2.1 scope).
- Expandable list of `supporting_evidence` + `refuting_evidence`
  passages with source, title, and link (Wikipedia hotlinks for
  `mediawiki` source).
- Wave 3+: `BP non-convergent` badge when `pcg.algorithm ===
  "LBP-nonconvergent"`.

---

## Health panel

The `/status` page polls `/health/deep` on a ~10 s interval and renders
per-layer health (`L1.decompose`, `L1.retrieve.neo4j`, etc.) with
latency. The top-level `status` field supports both `status` and
`overall` field names for tolerance; the canonical name is `status`.

---

## Accessibility

- WCAG 2.1 AA is the floor. The frontend overhaul preserved contrast
  ratios when adopting the light palette.
- All interactive elements have focus states.
- `prefers-reduced-motion` respected for any animations.

---

## What is NOT in v2 (but may look like it is in older docs)

The following were mentioned in v1 docs but are **not deployed** in
v2.0:

- **Supabase auth** — no user accounts, no OAuth sign-in.
- **Stripe billing** — no token-based pricing, no checkout flow.
- **API Proxy route** (`/api/ohi/*`) — not used; frontend calls the
  OHI API directly using `NEXT_PUBLIC_API_BASE`.
- **User dashboards** — no per-user history, no saved verifications.
- **SSE streaming** — deleted in D2; see the polling pattern above.

These may come back in v2.1+ or later, but are not part of the current
v2.0 surface.

---

## Testing

Local tests:

```bash
cd src/frontend
npm run dev          # dev server
npm run test         # vitest suite (138+ tests; reducer + client logic)
npm run typecheck    # TypeScript
npm run lint
npm run build        # production build; must pass before feat merges
```

Planned (Wave 3): `npm run test:e2e` with Playwright against the Vercel
preview for the full verify flow.

---

## Related

- [README.md](../README.md) — project overview
- [docs/API.md](API.md) — API schema consumed by the frontend client
- [docs/CONTRIBUTING.md](CONTRIBUTING.md) — contribution process
