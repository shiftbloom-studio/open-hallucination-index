# OHI v2 Frontend — Design Specification

**Sub-project 3 of 3** · **Date:** 2026-04-17 · **Author:** brainstorming session on local `main`
**Companions:** [`2026-04-16-ohi-v2-algorithm-design.md`](./2026-04-16-ohi-v2-algorithm-design.md) · [`2026-04-16-ohi-v2-infrastructure-design.md`](./2026-04-16-ohi-v2-infrastructure-design.md)

---

## 0. Scope & posture

**In scope:** Touch-up of the existing Next.js 16 / React 19 / Tailwind v4 / shadcn frontend under `src/frontend/`. Rewire to consume algorithm v2 (`DocumentVerdict` / `ClaimVerdict`), delete auth/pricing/admin/tokens + their deps, add `/verify`, `/calibration`, `/status`, preserve German legal pages, light-touch copy on landing sections, and wire SSE for `/verify/stream`.

**Out of scope (deferred):**
- Infrastructure / Terraform / Cloudflare / Vercel deployment — the spec *names* the target arrangement but implementation is **code-only**. No `infra/terraform/` edits during sub-project 3.
- Aesthetic pass — user wants the design to move away from generic SaaS-boilerplate aesthetics toward something experimental. That pass comes **after** the data wiring lands (separate follow-up).
- Client-side persistence of verdicts, `/verify/[request_id]` shareable route, batch verify UI, rewrite endpoint.

**Operating constraint:** time pressure. Spec-reviewer subagent loop skipped by user directive.

---

## 1. Architecture, hosting, API transport

### 1.1 Target production arrangement (defers to infra sub-project)

```
Browser
├─ Apex: ohi.shiftbloom.studio           → Vercel (static Next.js 16, `output: 'export'`)
└─ api.ohi.shiftbloom.studio             → Cloudflare (proxied) → Lambda Function URL (FastAPI)
                                            └─ CF Transform Rule injects X-OHI-Edge-Secret
                                            └─ CF WAF + rate-limits (existing, unchanged)
```

Frontend is **fully static** (no server routes, no RSC with runtime fetches). Browser calls the API cross-origin on `api.ohi.shiftbloom.studio` — Lambda CORS-allows `https://ohi.shiftbloom.studio`.

### 1.2 Infra deltas required before production cut-over *(NOT implemented in sub-project 3)*

The infra agent picks this up later:
1. `infra/terraform/cloudflare/dns.tf` — flip apex CNAME from Lambda → Vercel (`cname.vercel-dns.com`, DNS-only/gray-cloud). Add `_vercel` TXT row. Add `api` CNAME → Lambda (proxied).
2. `infra/terraform/cloudflare/edge_secret.tf` — Transform Rule scope tightens to `(http.host eq "api.ohi.shiftbloom.studio")`.
3. `infra/terraform/cloudflare/waf.tf` / `cache.tf` — path-based rules re-anchored to the API hostname.
4. FastAPI app — add `CORSMiddleware` with `allow_origins=["https://ohi.shiftbloom.studio"]`, exempt `OPTIONS` from `EdgeSecretMiddleware`. *(Backend change, implemented by whoever owns that edit — not sub-project 3.)*

### 1.3 Why no Next.js server-side proxy

Spec §11 of the algorithm design defines an `Authorization: Bearer <internal-token>` for trusted clients (MCP server, benchmarks, CI). The public frontend is **not** a trusted client — it hits public rate limits like any browser. Therefore: no secret to hide, no proxy needed. Direct cross-origin fetch from the browser.

### 1.4 Why Vercel + `output: 'export'` and not Cloudflare Pages

User directive. Vercel: first-class Next.js support, Hobby-tier free, static export keeps the artifact portable. CF-in-front-of-Vercel is **explicitly avoided** (double-CDN issues); DNS-only at apex lets Vercel own its edge.

### 1.5 Environment variables

Only `NEXT_PUBLIC_*` since the build is static:

| Var | Value | Notes |
|---|---|---|
| `NEXT_PUBLIC_API_BASE` | `https://api.ohi.shiftbloom.studio/api/v2` | Client-side fetches use this |
| `NEXT_PUBLIC_SITE_URL` | `https://ohi.shiftbloom.studio` | Metadata / sitemap / OG URLs |
| `NEXT_PUBLIC_SHOW_DEBUG` | `false` | Toggles raw-JSON debug drawer on `/verify` |

For local dev: point `NEXT_PUBLIC_API_BASE` at `http://localhost:8080/api/v2` (the local FastAPI). No other env vars needed. `src/frontend/.env.local` template is added.

---

## 2. Route map + folder layout

### 2.1 Public routes (after cleanup)

| Route | Purpose | Rendering |
|---|---|---|
| `/` | Landing | Static; existing five sections with copy retune |
| `/verify` | Verify console (form + streaming + PCG graph + feedback) | Client (SSE) — **NEW** |
| `/calibration` | Calibration transparency report | Client fetch (cached, 5-min revalidate via SWR) — **NEW** |
| `/status` | `/health/deep` dashboard + resting-state explanation | Client (10s polling) — **NEW** |
| `/about`, `/accessibility`, `/cookies`, `/disclaimer`, `/eula` | Existing static pages | Unchanged |
| `/agb`, `/datenschutz`, `/impressum` | German statutory legal | Unchanged |
| `/sitemap.xml`, `/robots.txt` | SEO | Regenerated with new route list |

### 2.2 Deleted routes (Next.js default 404)

`app/auth/`, `app/pricing/`, `app/admin/`, `app/tokens/`, `app/dashboard/`, `app/api/auth/`, `app/api/checkout/`, `app/api/webhooks/`, `app/api/admin/`, `app/api/tokens/`, `app/api/ohi/`.

### 2.3 `src/frontend/src/` folder after touch-up

```
app/
├── layout.tsx, page.tsx, providers.tsx, globals.css    # retained with trim
├── robots.ts, sitemap.ts                                # updated route list
├── verify/{page.tsx, loading.tsx}                       # NEW
├── calibration/page.tsx                                 # NEW
├── status/page.tsx                                      # NEW
├── about/, accessibility/, cookies/, disclaimer/, eula/, agb/, datenschutz/, impressum/    # unchanged

components/
├── layout/ (Navbar, Footer, CookieConsent)              # trim auth/pricing links, add status dot
├── landing/ (Hero, Problem, ArchitectureFlow, FeatureGrid, Cta)   # copy retune only
├── ui/ (shadcn primitives)                              # unchanged
├── 3d/, analytics/                                      # unchanged
├── verify/                                              # NEW
│   ├── VerifyForm.tsx, VerifyPage.tsx
│   ├── DocumentVerdictCard.tsx, ClaimCard.tsx, ClaimList.tsx
│   ├── IntervalBar.tsx, DomainBadge.tsx, FallbackBadge.tsx
│   ├── PcgGraph.tsx, EvidenceDrawer.tsx
│   ├── FeedbackSheet.tsx, FeedbackButton.tsx
│   ├── SseProgress.tsx
│   └── ModelVersionsFootnote.tsx
├── calibration/CalibrationTable.tsx                     # NEW
├── status/HealthMatrix.tsx                              # NEW
└── common/
    ├── RestingState.tsx, LlmUnavailableState.tsx
    ├── BudgetExhaustedState.tsx, RateLimitedState.tsx
    ├── DegradedState.tsx, NetworkErrorState.tsx
    └── StatusDot.tsx                                    # shared navbar + status page

lib/
├── ohi-types.ts                                         # NEW — mirrors algorithm spec §9
├── ohi-client.ts                                        # NEW — replaces v1 lib/api.ts
├── ohi-queries.ts                                       # NEW — TanStack Query hooks
├── sse.ts                                               # NEW — POST-SSE parser
├── verify-controller.ts                                 # NEW — state machine + reducer
├── utils.ts                                             # unchanged (cn helper)
└── __tests__/                                           # updated fixtures to v2
```

### 2.4 Deleted folders & files

`app/auth/`, `app/pricing/`, `app/admin/`, `app/tokens/`, `app/dashboard/`; `app/api/auth/`, `api/checkout/`, `api/webhooks/`, `api/admin/`, `api/tokens/`, `api/ohi/`; `lib/supabase/`, `lib/stripe.ts`, `lib/db/`; `components/dashboard/` (its three useful files are promoted into `components/verify/` after a full rewrite against v2 types, not literal moves); `drizzle.config.ts`; `lib/api.ts`; `src/proxy.ts` (after verification it's an orphan).

### 2.5 `package.json` cleanup

**Remove:** `@supabase/ssr`, `@supabase/supabase-js`, `@stripe/stripe-js`, `stripe`, `drizzle-orm`, `drizzle-kit`, `postgres`. Remove the `db:*` npm scripts. Keep `@vercel/analytics` (host is Vercel). Keep `react-force-graph-3d`, `framer-motion`, `three`, `@react-three/*` — used by landing 3D and PCG graph.

---

## 3. Data contract

`lib/ohi-types.ts` literally mirrors algorithm spec §9 pydantic models. Hand-written, no codegen. A drift unit test loads a golden fixture (copied verbatim from spec §10 example) and asserts structural validity via `zod`. Added: `VerifyOptions`, `VerifyRequest`, `FeedbackRequest`, and an `SseEvent` discriminated union over the eight event types in spec §10 (`decomposition_complete`, `claim_routed`, `nli_complete`, `pcg_propagation_complete`, `refinement_pass_complete`, `claim_verdict`, `document_verdict`, `error`).

Additional shared types: `CalibrationReport` and `HealthDeep` — minimal objects mirroring spec §10 response shapes.

`lib/ohi-client.ts` exposes an `ohi` object with `verify`, `verdict`, `feedback`, `calibrationReport`, `healthDeep`, and an `OhiError` class with `isResting`, `isLlmDown`, `isBudgetExhausted`, `isDegraded`, `isRateLimited` predicates derived from status code + body. Errors preserve `Retry-After` header.

`lib/ohi-queries.ts` wraps the non-streaming calls in TanStack Query hooks: `useVerdict(id)`, `useCalibration()`, `useHealthDeep()`, `useFeedbackMutation()`.

Full type definitions per section §3 of brainstorming transcript — reproduced as-is in the implementation without further design decisions.

---

## 4. `/verify` page component tree

Layout: **two-column on ≥1024px, single-column on <1024px**. Left column (36%): `VerifyForm` + `SseProgress` (sticky on desktop). Right column (64%): `VerdictPanel` containing `DocumentVerdictCard`, `ClaimList` (virtualized via `react-window` if claims > 30), `PcgGraph`, and error overlays.

State machine in `lib/verify-controller.ts`: `idle → streaming → (partial) → complete`, with branches to `sync-fallback` (if SSE times out on first-byte) and `error`. Abort on unmount / "Cancel" button / new submit — single owned `AbortController`.

Key interactive behaviors:
- `ClaimCard` expand → `EvidenceDrawer` (supporting/refuting split)
- `ClaimCard` "Show in graph" → tweens `PcgGraph` camera to that node
- `ClaimCard` 🚩 → opens `FeedbackSheet` (Radix Dialog), POSTs `/feedback` with `labeler.kind="user"`, idempotency key = `(request_id, claim_id, labeler.id)` with labeler id derived client-side from `crypto.randomUUID()` persisted in localStorage per spec §11 anti-spam posture
- Sort dropdown above `ClaimList`: claim-ID (default, matches SSE order) | `p_true` asc | `information_gain` desc. No reorder mid-stream.

`IntervalBar` color bands: `p_true ≥ 0.8` emerald-400, `0.5 ≤ p < 0.8` amber-400, `p < 0.5` rose-500. Same component drives the DocumentVerdict headline and the per-claim mini bar via a `size` prop.

`PcgGraph` wraps `react-force-graph-3d` (already a dependency). Nodes = one per claim, color by `p_true` band, size by `information_gain`. Edges = union of all `pcg_neighbors[]` across claims, colored by `edge_type` (entail=emerald, contradict=rose, neutral=slate-500), width by `|edge_strength|`. Default hides neutral edges; toggle in controls.

Debug drawer (`NEXT_PUBLIC_SHOW_DEBUG=true`) renders raw `DocumentVerdict` JSON + `model_versions` + fallback summary — dev-only inspection tool.

---

## 5. Error and edge states

Each is a named component in `components/common/`, rendered in place of `VerdictPanel` (or as a banner for `degraded`). No generic "Something went wrong" anywhere.

| State | Trigger | Component | Primary copy |
|---|---|---|---|
| Resting | 503 + `{status: "resting"}` | `RestingState` | "Service is resting — the PC that hosts OHI's data is offline." Retry countdown from `Retry-After`. |
| Budget exhausted | 503 + `detail.includes("budget exhausted")` | `BudgetExhaustedState` | "Daily budget exhausted — resets in N hours." CTA: "Browse calibration report". |
| LLM unavailable | 503 + `{status: "llm_unavailable"}` | `LlmUnavailableState` | "Upstream Gemini is not responding." Retry CTA. |
| Rate-limited | 429 | `RateLimitedState` | "Rate limit reached (10/min)." Countdown, retry disabled until elapsed. |
| Degraded | 200 or 503 with `degraded_layers[]` OR `fallback_used != null` on any claim | `DegradedState` banner | Inline above `ClaimList`; lists affected layers. Result still renders. |
| Network / CORS / SSE give-up | fetch rejection, 504, 3×SSE reconnect failure | `NetworkErrorState` | Retry CTA forces sync-fallback to `POST /verify`. |

Global rules:
- All 503/429 use the header `Retry-After` (not client guessing) for the countdown
- Auto-retry allowed **only** for SSE reconnect (≤3 attempts, backoff 0.5/1.5/4s) and network-level errors (1 retry, 2s). **Never auto-retry 429/503** — user-initiated only.
- Navbar `StatusDot`: 30s-cached `/health/deep` fetch; green/amber/red; clicks through to `/status`.

---

## 6. SSE integration

### 6.1 Transport

POST-SSE via `fetch` + `ReadableStream`. `EventSource` is unsuitable (GET-only). `lib/sse.ts` parses `event:` / `data:` frames via `\n\n` chunking, typed to `SseEvent` discriminated union.

### 6.2 Controller

`useVerifyController()` owns an `AbortController`, a reducer over `VerifyState` (progress bag, `claims[]`, `verdict`, `error`), and the state machine from §4. Event handlers fan out:

- `decomposition_complete` → progress + ETA
- `claim_routed` → Set of routed claim-IDs
- `nli_complete`, `pcg_propagation_complete`, `refinement_pass_complete` → progress bag updates
- `claim_verdict` → append to `claims` (de-dupe on `claim.id`, required for reconnect-replay)
- `document_verdict` → `verdict` set + status → `complete`
- `error` → status → `error`, route to matching `components/common/*State`

Batched state commits via `requestIdleCallback` (batch of 5, max 100ms delay) to avoid render storms for 100-claim documents.

### 6.3 Reconnect + fallback

- Network error mid-stream → up to 3 reconnects, exponential backoff. On attempt ≥ 2, also fetch `/verdict/{request_id}` to merge any claims missed.
- No bytes within 8 s of first-byte waiting → switch to sync `POST /verify`. This covers the "corporate proxy strips streams" case.
- 3 reconnect failures → `NetworkErrorState` with sync-fallback button.

### 6.4 Backend coordination note (non-blocking)

Backend SSE handler should detect client disconnect via FastAPI's `await request.is_disconnected()` and cancel the pipeline. Not blocking the frontend, but listed for the algorithm sub-project's awareness.

### 6.5 Tests

- `lib/__tests__/sse.parse.test.ts` — partial frames, multiple-per-chunk, CRLF tolerance
- `lib/__tests__/verify-controller.test.ts` — MSW-mocked event sequence, assert state transitions
- Playwright e2e: deterministic SSE fixture, assert claim cards appear in emitted order

---

## 7. Landing copy retune

Existing five sections keep structure, visuals, animations, and `KnowledgeGraphCanvas`. **Copy only** changes. Full copy in brainstorming transcript §7; highlights:

- **HeroSection** — headline shifts from generic "verify AI outputs" to **"How much should you trust that answer?"**; subhead leads with "calibrated probability + uncertainty interval"; CTAs become `Try /verify` + `Read the calibration report`.
- **ProblemSection** — reframe around "LLM self-reported confidence is uncorrelated with truth; OHI's 0.85 [0.78, 0.91] at 90% coverage is a coverage guarantee, not a vibe."
- **ArchitectureFlow** — relabel nodes to v2 pipeline: L1 decomposition → L2 domain router → L3 NLI cross-encoder → L4 PCG (TRW-BP) → L5 conformal → L7 Gaussian-copula assembly. Footer note references `fallback_used`.
- **FeatureGrid** — replace three cards with: **Calibrated probabilities**, **Probabilistic Claim Graph**, **Open, auditable, rest-respecting**. Icons/gradients/hover motion retained.
- **CtaSection** — "Try it now. No account. No waitlist." → `/verify`. Secondary: API docs / v2 spec on GitHub.
- **Navbar** — drop Login/Signup; add `/verify`, `/calibration`, `/status`, `/about`. `StatusDot` on right.
- **Footer** — drop Pricing/Login/Sign up. Keep German legal + Disclaimer/EULA/Cookies/Accessibility.
- **`CookieConsent`** — categories reduced to "Essential" + "Analytics (Vercel)". Remove Supabase/Stripe-era categories.
- **Root `metadata`** — `description` matches new hero.

---

## 8. Testing & deletion sequence

### 8.1 Test suite

Kept: Vitest + RTL + MSW + Playwright. `test/setup.ts` unchanged. Re-targeted fixtures: v1 `TrustScore`/`ClaimSummary` → v2 `DocumentVerdict`/`ClaimVerdict` golden JSON (copied verbatim from algorithm spec §10 example).

New test files:
- `lib/__tests__/ohi-types.contract.test.ts` — zod schema validates golden fixture
- `lib/__tests__/ohi-client.test.ts` — MSW-mocked success + each error-state body → correct `OhiError` predicate fires
- `lib/__tests__/sse.parse.test.ts` — SSE frame parser
- `lib/__tests__/verify-controller.test.ts` — state machine transitions
- `components/verify/__tests__/ClaimCard.test.tsx` — renders p_true band, interval bar, fallback badge
- `components/common/__tests__/*.test.tsx` — smoke tests per error-state component
- `e2e/verify-streaming.spec.ts` — Playwright, MSW-driven SSE fixture, claim-cards-in-order assertion
- `e2e/verify-resting.spec.ts` — backend 503 resting response → `RestingState` renders + countdown
- `e2e/calibration.spec.ts`, `e2e/status.spec.ts` — smoke

Deleted: `e2e/auth.*`, `e2e/checkout.*`, `components/dashboard/__tests__/*` (components themselves are deleted/rewritten).

### 8.2 Deletion & migration order

Safe sequence — each step leaves the tree buildable:

1. **Prep branch.** Work on `feat/ohi-v2-foundation` (same branch as algorithm + infra). No remote push.
2. **Write `lib/ohi-types.ts` + `lib/ohi-client.ts` + `lib/sse.ts` + `lib/verify-controller.ts` + `lib/ohi-queries.ts`** with their tests. Old `lib/api.ts` still in place, so builds still pass.
3. **Build `components/verify/*` and `components/common/*`** against new types. Still no route changes.
4. **Add `app/verify/`, `app/calibration/`, `app/status/`** — new routes live, old routes untouched.
5. **Retune landing copy in `components/landing/*`** (HeroSection, ProblemSection, ArchitectureFlow, FeatureGrid, CtaSection).
6. **Trim `Navbar`, `Footer`, `CookieConsent`** — remove auth/pricing links, add status dot + new routes.
7. **Delete `app/api/*` (auth, checkout, webhooks, admin, tokens, ohi)** + `lib/supabase/`, `lib/stripe.ts`, `lib/db/`, `lib/api.ts`. Typecheck catches any stragglers.
8. **Delete `app/auth/`, `app/pricing/`, `app/admin/`, `app/tokens/`, `app/dashboard/`**. Typecheck + build + lint + test.
9. **Remove packages from `package.json`** (Supabase / Stripe / Drizzle / Postgres). `npm install`, re-run full test suite.
10. **Update `sitemap.ts`, `robots.ts`, root `metadata`, README**. Verify build artifacts under `out/` (static export).
11. **Manual smoke test against local FastAPI** (`NEXT_PUBLIC_API_BASE=http://localhost:8080/api/v2`): verify request, SSE stream, resting-state simulation (stop local Postgres container), calibration report render.

**Do not** in sub-project 3:
- Edit any Terraform file under `infra/`
- Deploy anything to Vercel
- Change any backend Python file (CORS/OPTIONS adjustments from §1.2 are separate work)
- Push to `origin`

### 8.3 Verification gate before calling implementation complete

- `npm run lint` green
- `npm run test:run` green
- `npm run test:e2e` green
- `npm run build` green with `output: 'export'` producing `src/frontend/out/`
- Manual smoke test checklist in §8.2 step 11 passed
- All deleted files actually gone (no orphans); `git grep -i "supabase\|stripe\|drizzle" src/frontend` returns empty

---

## 9. Aesthetic pass — explicit follow-up scope

User has flagged (2026-04-17, saved as feedback memory) that the visual design should lean **experimental / innovative**, not "100th SaaS template". This sub-project delivers the **wiring and content** using the existing Tailwind-v4 + shadcn aesthetic as a baseline. A separate follow-up — invoking `frontend-design` — will rework typography, spacing, motion, color identity, and the hero / sections visual treatment after the data wiring lands. Tangling that into sub-project 3 mixes two very different concerns (and time pressure forecloses it now).

---

## 10. Open coordination items with other sub-projects

1. **Backend CORS** — FastAPI needs `CORSMiddleware(allow_origins=["https://ohi.shiftbloom.studio"])`, `EdgeSecretMiddleware` must exempt `OPTIONS`. Not sub-project 3's edit, but blocks cut-over.
2. **Backend SSE disconnect handling** — pipeline cancels on `request.is_disconnected()` to avoid wasted compute.
3. **Infra DNS flip + `api.` subdomain** — per §1.2.
4. **Synthetic calibration seed** — spec §10 mentions a static HTML snapshot of the calibration report on S3. Out of scope here; the `/calibration` page in this sub-project hits the live `GET /api/v2/calibration/report` endpoint.

End of spec.
