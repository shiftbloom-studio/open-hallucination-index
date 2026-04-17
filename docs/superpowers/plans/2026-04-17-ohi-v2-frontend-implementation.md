# OHI v2 Frontend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewire the existing Next.js 16 frontend to consume algorithm v2 (`DocumentVerdict` / `ClaimVerdict`), delete account-scoped features (auth/pricing/admin/tokens/dashboard), add public routes (`/verify`, `/calibration`, `/status`), preserve German legal pages, and ship SSE streaming for `/verify/stream` — all without touching infrastructure or deploying.

**Architecture:** Static Next.js export (`output: 'export'`) consuming the v2 API cross-origin. No server routes. TanStack Query for non-streaming; custom POST-SSE client for `/verify/stream`. Existing visual system (Tailwind v4 + shadcn + Framer Motion + R3F) retained; aesthetic pass deferred to a separate follow-up.

**Tech Stack:** Next.js 16, React 19, TypeScript 5, Tailwind v4, shadcn/ui, Framer Motion, React Three Fiber, react-force-graph-3d, TanStack Query 5, Vitest + RTL + MSW, Playwright, zod.

**Spec:** [`docs/superpowers/specs/2026-04-17-ohi-v2-frontend-design.md`](../specs/2026-04-17-ohi-v2-frontend-design.md)

**Branch:** `feat/ohi-v2-foundation` (local only, not pushed).

**Hard constraints:**
- Only touch files under `src/frontend/`
- DO NOT edit any `infra/terraform/*` file
- DO NOT deploy to Vercel or anywhere else
- DO NOT push to remote
- DO NOT edit Python backend (`src/api/*`) — CORS / OPTIONS / SSE-disconnect adjustments are separate work

---

## File Structure

### New files under `src/frontend/src/`

| Path | Responsibility |
|---|---|
| `lib/ohi-types.ts` | Type mirror of algorithm spec §9 (`Claim`, `Evidence`, `ClaimEdge`, `ClaimVerdict`, `DocumentVerdict`, `VerifyRequest`, `FeedbackRequest`, `SseEvent`, etc.) |
| `lib/ohi-client.ts` | `ohi` client + `OhiError` class with error-kind predicates |
| `lib/ohi-queries.ts` | TanStack Query hooks wrapping non-streaming calls |
| `lib/sse.ts` | POST-SSE parser `streamVerify()` |
| `lib/verify-controller.ts` | `useVerifyController()` state machine / reducer |
| `lib/__tests__/ohi-types.contract.test.ts` | zod-validates golden fixture against types |
| `lib/__tests__/ohi-client.test.ts` | MSW-mocked error predicates |
| `lib/__tests__/sse.parse.test.ts` | Frame parser unit tests |
| `lib/__tests__/verify-controller.test.ts` | State-machine transitions |
| `test/fixtures/document-verdict.golden.json` | Copied verbatim from algorithm spec §10 |
| `test/fixtures/sse-event-stream.txt` | Deterministic SSE byte sequence |
| `components/verify/VerifyForm.tsx` | Text input + option chips + submit |
| `components/verify/SseProgress.tsx` | Layer-completion timeline |
| `components/verify/DocumentVerdictCard.tsx` | Headline score + interval bar |
| `components/verify/IntervalBar.tsx` | Reusable horizontal interval bar (variants: `lg`, `sm`) |
| `components/verify/DomainBadge.tsx` | Per-domain chip |
| `components/verify/FallbackBadge.tsx` | Amber chip when `fallback_used != null` |
| `components/verify/ClaimCard.tsx` | Per-claim row |
| `components/verify/ClaimList.tsx` | Sortable list wrapper |
| `components/verify/EvidenceDrawer.tsx` | Supporting/refuting evidence expansion |
| `components/verify/FeedbackButton.tsx` + `FeedbackSheet.tsx` | Radix Dialog, POSTs `/feedback` |
| `components/verify/PcgGraph.tsx` | 3D force graph wrapper |
| `components/verify/VerifyPage.tsx` | Top-level state host |
| `components/common/RestingState.tsx` | 503 `status: "resting"` |
| `components/common/BudgetExhaustedState.tsx` | 503 budget exhausted |
| `components/common/LlmUnavailableState.tsx` | 503 `status: "llm_unavailable"` |
| `components/common/RateLimitedState.tsx` | 429 with `Retry-After` |
| `components/common/DegradedState.tsx` | Inline banner for `fallback_used` / `degraded_layers` |
| `components/common/NetworkErrorState.tsx` | Fetch/CORS/SSE give-up |
| `components/common/StatusDot.tsx` | Navbar health indicator |
| `components/calibration/CalibrationTable.tsx` | Per-domain coverage + intervals |
| `components/status/HealthMatrix.tsx` | Per-layer health/latency |
| `app/verify/page.tsx` + `loading.tsx` | `/verify` route |
| `app/calibration/page.tsx` | `/calibration` route |
| `app/status/page.tsx` | `/status` route |

### Files to modify under `src/frontend/`

| Path | Change |
|---|---|
| `next.config.ts` | Add `output: 'export'`; keep security headers (they translate to static-export meta tags where possible) |
| `src/app/layout.tsx` | Strip Supabase imports from `Providers`; update metadata description |
| `src/app/providers.tsx` | Remove Supabase context; keep TanStack Query + next-themes |
| `src/app/sitemap.ts`, `src/app/robots.ts` | Update route list |
| `src/components/layout/navbar.tsx` | Drop Login/Signup; add `/verify` `/calibration` `/status`; add `StatusDot` |
| `src/components/layout/footer.tsx` | Drop Pricing / Login / Sign up |
| `src/components/layout/cookie-consent.tsx` | Simplify categories to Essential + Analytics |
| `src/components/landing/HeroSection.tsx` | Copy retune (headline/subhead/CTAs) |
| `src/components/landing/ProblemSection.tsx` | Copy retune |
| `src/components/landing/ArchitectureFlow.tsx` | Node labels → v2 pipeline |
| `src/components/landing/FeatureGrid.tsx` | Replace three cards (titles/descriptions/icons) |
| `src/components/landing/CtaSection.tsx` | Replace CTAs; no sign-up |
| `vitest.config.ts` | Remove `supabase/**` and `db/index.ts` from coverage excludes (those paths are gone); adjust include globs |
| `package.json` | Remove Supabase/Stripe/Drizzle/Postgres deps + `db:*` scripts |
| `src/frontend/.env.example` | Rewrite: `NEXT_PUBLIC_API_BASE`, `NEXT_PUBLIC_SITE_URL`, `NEXT_PUBLIC_SHOW_DEBUG` |

### Files / directories to delete

| Path | Note |
|---|---|
| `src/app/auth/` | Login / signup / reset |
| `src/app/pricing/` | Stripe-era |
| `src/app/admin/` | Token admin UI |
| `src/app/tokens/` | Token management |
| `src/app/dashboard/` | Account dashboard |
| `src/app/api/auth/` | Supabase SSR callbacks |
| `src/app/api/checkout/` | Stripe |
| `src/app/api/webhooks/` | Stripe |
| `src/app/api/admin/` | |
| `src/app/api/tokens/` | |
| `src/app/api/ohi/` | Old server proxy — superseded by direct cross-origin calls |
| `src/lib/supabase/` | |
| `src/lib/stripe.ts` | |
| `src/lib/db/` | |
| `src/lib/api.ts` | Replaced by `ohi-client.ts` |
| `src/components/dashboard/` | Three files were useful but require full rewrite against v2 schema; rewritten into `components/verify/` by Phase 2 tasks |
| `src/proxy.ts` | Orphan wrapper — verify unused before deleting |
| `drizzle.config.ts` | |
| `e2e/auth*.spec.ts`, `e2e/checkout*.spec.ts` | Any matching E2E |

---

## Phase 0 — Setup & baseline

### Task 0.1: Verify baseline build and tests pass on current branch

**Files:** no changes

- [ ] **Step 1: Ensure dependencies installed**

Run: `cd src/frontend && npm ci`
Expected: exit 0, `node_modules/` populated

- [ ] **Step 2: Run lint**

Run: `cd src/frontend && npm run lint`
Expected: exit 0 (or documented pre-existing warnings)

- [ ] **Step 3: Run unit tests**

Run: `cd src/frontend && npm run test:run`
Expected: all green (v1 tests)

- [ ] **Step 4: Run build**

Run: `cd src/frontend && npm run build`
Expected: exit 0

- [ ] **Step 5: Record baseline in a working note**

Write: `src/frontend/MIGRATION_NOTES.md` — one line: "Baseline: lint+test+build green at commit `<sha>`". Do NOT commit this file; it's scratch.

**No commit yet** — starts clean.

### Task 0.2: Add golden fixture + contract test infrastructure

**Files:**
- Create: `src/frontend/src/test/fixtures/document-verdict.golden.json`
- Create: `src/frontend/src/test/fixtures/sse-event-stream.txt`

- [ ] **Step 1: Create `document-verdict.golden.json`** — copy the truncated example from algorithm spec §10 verbatim. It must have `request_id`, `pipeline_version`, `model_versions`, `document_score`, `document_interval`, `internal_consistency`, `decomposition_coverage`, `processing_time_ms`, `rigor`, `refinement_passes_executed`, and `claims` array with one complete `ClaimVerdict` including `pcg_neighbors`.

- [ ] **Step 2: Create `sse-event-stream.txt`** — a deterministic byte sequence with exactly one of each event type from spec §10:

```
event: decomposition_complete
data: {"claim_count": 2, "estimated_total_ms": 60000}

event: claim_routed
data: {"claim_id": "c1", "domain": "general", "weights": {"general": 0.94, "biomedical": 0.06}}

event: nli_complete
data: {"claim_evidence_pairs_scored": 24, "claim_pair_pairs_scored": 2}

event: pcg_propagation_complete
data: {"iterations": 4, "converged": true, "algorithm": "TRW-BP", "internal_consistency": 0.83, "gibbs_validated": null}

event: claim_verdict
data: <stringified first claim from golden.json>

event: claim_verdict
data: <stringified second claim>

event: document_verdict
data: <entire golden.json>

```

(Include trailing blank line.)

- [ ] **Step 3: Commit**

```bash
git add src/frontend/src/test/fixtures/
git commit -m "test(fixtures): add v2 DocumentVerdict + SSE golden fixtures"
```

---

## Phase 1 — Foundation: types + client + SSE + controller

### Task 1.1: `lib/ohi-types.ts` with zod contract test

**Files:**
- Create: `src/frontend/src/lib/ohi-types.ts`
- Create: `src/frontend/src/lib/__tests__/ohi-types.contract.test.ts`

- [ ] **Step 1: Write contract test first (failing)**

```ts
// src/frontend/src/lib/__tests__/ohi-types.contract.test.ts
import { describe, it, expect } from "vitest";
import { z } from "zod";
import golden from "../../test/fixtures/document-verdict.golden.json";
import { DocumentVerdictSchema } from "../ohi-types";

describe("ohi-types contract", () => {
  it("DocumentVerdictSchema accepts the spec §10 golden fixture", () => {
    const parsed = DocumentVerdictSchema.safeParse(golden);
    expect(parsed.success).toBe(true);
    if (!parsed.success) {
      throw new Error(`Golden fixture invalid: ${JSON.stringify(parsed.error.issues, null, 2)}`);
    }
  });

  it("rejects payloads missing document_score", () => {
    const { document_score, ...rest } = golden as Record<string, unknown>;
    const parsed = DocumentVerdictSchema.safeParse(rest);
    expect(parsed.success).toBe(false);
  });
});
```

- [ ] **Step 2: Run test — expected FAIL (types module doesn't exist)**

Run: `cd src/frontend && npm run test:run -- ohi-types.contract`
Expected: FAIL — "Cannot find module '../ohi-types'"

- [ ] **Step 3: Write `lib/ohi-types.ts`** — the full type file from spec §3, augmented with zod schemas (`ClaimSchema`, `EvidenceSchema`, `ClaimEdgeSchema`, `ClaimVerdictSchema`, `DocumentVerdictSchema`, `VerifyRequestSchema`, `FeedbackRequestSchema`). Export both the TS types and the schemas. Use `z.infer<>` for type derivation so types and schemas can't drift.

Key schemas:
```ts
export const DomainSchema = z.enum(["general", "biomedical", "legal", "code", "social"]);
export type Domain = z.infer<typeof DomainSchema>;

export const ClaimSchema = z.object({
  id: z.string(),
  text: z.string(),
  claim_type: z.string().nullable().optional(),
  span: z.tuple([z.number(), z.number()]).nullable().optional(),
});
// ... etc for the full schema set from spec §3
```

- [ ] **Step 4: Run test — expected PASS**

Run: `cd src/frontend && npm run test:run -- ohi-types.contract`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/frontend/src/lib/ohi-types.ts src/frontend/src/lib/__tests__/ohi-types.contract.test.ts
git commit -m "feat(types): v2 DocumentVerdict type + zod schema mirroring algorithm spec §9"
```

### Task 1.2: `lib/ohi-client.ts` + `OhiError` + MSW error predicate tests

**Files:**
- Create: `src/frontend/src/lib/ohi-client.ts`
- Create: `src/frontend/src/lib/__tests__/ohi-client.test.ts`

- [ ] **Step 1: Write failing tests**

```ts
import { describe, it, expect, vi, beforeAll, afterAll, afterEach } from "vitest";
import { setupServer } from "msw/node";
import { http, HttpResponse } from "msw";
import { ohi, OhiError } from "../ohi-client";
import golden from "../../test/fixtures/document-verdict.golden.json";

const API_BASE = "https://api.ohi.shiftbloom.studio/api/v2";
const server = setupServer();

beforeAll(() => {
  process.env.NEXT_PUBLIC_API_BASE = API_BASE;
  server.listen();
});
afterEach(() => server.resetHandlers());
afterAll(() => server.close());

describe("ohi-client", () => {
  it("verify() returns DocumentVerdict on 200", async () => {
    server.use(http.post(`${API_BASE}/verify`, () => HttpResponse.json(golden)));
    const res = await ohi.verify({ text: "hi" });
    expect(res.document_score).toBe((golden as any).document_score);
  });

  it.each([
    { status: 503, body: { status: "resting", reason: "origin-unreachable" }, predicate: "isResting" },
    { status: 503, body: { status: "llm_unavailable" }, predicate: "isLlmDown" },
    { status: 503, body: { detail: "OHI public budget exhausted, resets in 7h" }, predicate: "isBudgetExhausted" },
    { status: 503, body: { degraded_layers: ["L3"] }, predicate: "isDegraded" },
    { status: 429, body: { detail: "rate limited" }, predicate: "isRateLimited" },
  ])("OhiError exposes $predicate for status=$status", async ({ status, body, predicate }) => {
    server.use(http.post(`${API_BASE}/verify`, () => HttpResponse.json(body, { status, headers: { "Retry-After": "60" } })));
    await expect(ohi.verify({ text: "x" })).rejects.toSatisfy((e: any) => e instanceof OhiError && e[predicate] === true);
  });

  it("parses Retry-After header", async () => {
    server.use(http.post(`${API_BASE}/verify`, () => HttpResponse.json({ detail: "slow down" }, { status: 429, headers: { "Retry-After": "48" } })));
    try { await ohi.verify({ text: "x" }); }
    catch (e) { expect((e as OhiError).retryAfterSec).toBe(48); return; }
    throw new Error("expected rejection");
  });
});
```

- [ ] **Step 2: Run — expected FAIL** (`Cannot find module '../ohi-client'`)

- [ ] **Step 3: Implement `lib/ohi-client.ts`** per spec §3. Exports: `ohi` object with `verify`, `verdict`, `feedback`, `calibrationReport`, `healthDeep`; `OhiError` class with predicates `isResting`, `isLlmDown`, `isBudgetExhausted`, `isDegraded`, `isRateLimited` + `retryAfterSec`. Reads `process.env.NEXT_PUBLIC_API_BASE`.

- [ ] **Step 4: Run — expected PASS**

Run: `cd src/frontend && npm run test:run -- ohi-client`

- [ ] **Step 5: Commit**

```bash
git add src/frontend/src/lib/ohi-client.ts src/frontend/src/lib/__tests__/ohi-client.test.ts
git commit -m "feat(client): v2 ohi-client with OhiError classification"
```

### Task 1.3: `lib/sse.ts` POST-SSE parser

**Files:**
- Create: `src/frontend/src/lib/sse.ts`
- Create: `src/frontend/src/lib/__tests__/sse.parse.test.ts`

- [ ] **Step 1: Write failing tests for the frame parser**

Tests to cover:
1. Single `event:` + `data:` frame round-trips to an `SseEvent`
2. Multiple frames in one chunk parse independently
3. Partial frame split across chunks reassembles
4. CRLF line endings tolerated
5. Malformed JSON in `data:` returns `null` (doesn't throw)
6. Unknown event name yields `null`

Write tests calling an internal `parseFrame()` exported for testability, PLUS an integration test that feeds `streamVerify` a mocked `ReadableStream` built from `test/fixtures/sse-event-stream.txt` and asserts the handler sees all 7 events in order.

- [ ] **Step 2: Run — expected FAIL**

- [ ] **Step 3: Implement `lib/sse.ts`** per spec §6.1. Export `streamVerify(body, handlers, signal)` and internal `parseFrame(raw)` (used by tests).

- [ ] **Step 4: Run — expected PASS**

Run: `cd src/frontend && npm run test:run -- sse.parse`

- [ ] **Step 5: Commit**

```bash
git add src/frontend/src/lib/sse.ts src/frontend/src/lib/__tests__/sse.parse.test.ts
git commit -m "feat(sse): POST-SSE client for /verify/stream"
```

### Task 1.4: `lib/verify-controller.ts` state machine

**Files:**
- Create: `src/frontend/src/lib/verify-controller.ts`
- Create: `src/frontend/src/lib/__tests__/verify-controller.test.ts`

- [ ] **Step 1: Write failing test** — reducer handling each SSE event type, transitions idle → streaming → complete, dedupe of `claim_verdict` by `claim.id`, error routing.

- [ ] **Step 2: Run — expected FAIL**

- [ ] **Step 3: Implement** `verify-controller.ts`:
  - Export `verifyReducer(state, action)` (pure fn, testable without React)
  - Export `useVerifyController()` React hook owning `AbortController` + `useReducer(verifyReducer, initialState)`

- [ ] **Step 4: Run — expected PASS**

- [ ] **Step 5: Commit**

```bash
git add src/frontend/src/lib/verify-controller.ts src/frontend/src/lib/__tests__/verify-controller.test.ts
git commit -m "feat(controller): verify page state machine"
```

### Task 1.5: `lib/ohi-queries.ts` TanStack Query hooks

**Files:**
- Create: `src/frontend/src/lib/ohi-queries.ts`

- [ ] **Step 1: Implement** — thin hooks: `useCalibration()`, `useHealthDeep()`, `useFeedbackMutation()`. Query keys per spec §3. 30s stale for health, 5m stale for calibration.

- [ ] **Step 2: Verify typecheck**

Run: `cd src/frontend && npx tsc --noEmit`

- [ ] **Step 3: Commit**

```bash
git add src/frontend/src/lib/ohi-queries.ts
git commit -m "feat(queries): TanStack Query hooks for calibration + health + feedback"
```

---

## Phase 2 — Verify page components

### Task 2.1: `IntervalBar`, `DomainBadge`, `FallbackBadge`

**Files:**
- Create: `src/frontend/src/components/verify/IntervalBar.tsx`
- Create: `src/frontend/src/components/verify/DomainBadge.tsx`
- Create: `src/frontend/src/components/verify/FallbackBadge.tsx`
- Create: `src/frontend/src/components/verify/__tests__/IntervalBar.test.tsx`

- [ ] **Step 1: Write tests** for `IntervalBar`:
  - renders with correct filled-range width given `interval={[0.4, 0.9]}` (50% width, offset 40%)
  - applies green/amber/red class based on `pTrue` bands (≥0.8, 0.5-0.8, <0.5)
  - `size="sm"` variant applies compact class

- [ ] **Step 2: Run — FAIL**

- [ ] **Step 3: Implement three components**

`IntervalBar`: horizontal bar with `[lower, upper]` filled range and a tick marker at `p_true`. Props: `interval: [number, number]`, `pTrue: number`, `size?: "sm" | "lg"`. Tailwind classes.

`DomainBadge`: chip per `Domain` with distinct per-domain color. Props: `domain: Domain`, `weight?: number` (optional, shown as opacity).

`FallbackBadge`: amber chip showing `fallback_used` value when not null. Props: `kind: FallbackKind`.

- [ ] **Step 4: Run — PASS**

- [ ] **Step 5: Commit**

```bash
git add src/frontend/src/components/verify/IntervalBar.tsx src/frontend/src/components/verify/DomainBadge.tsx src/frontend/src/components/verify/FallbackBadge.tsx src/frontend/src/components/verify/__tests__/IntervalBar.test.tsx
git commit -m "feat(verify): IntervalBar + DomainBadge + FallbackBadge primitives"
```

### Task 2.2: `DocumentVerdictCard`

**Files:**
- Create: `src/frontend/src/components/verify/DocumentVerdictCard.tsx`
- Create: `src/frontend/src/components/verify/__tests__/DocumentVerdictCard.test.tsx`

- [ ] **Step 1: Test** — given golden fixture, renders score, interval bar, internal consistency, rigor, processing_time, refinement passes count.

- [ ] **Step 2: Run — FAIL**

- [ ] **Step 3: Implement** per spec §4. Accepts `verdict: DocumentVerdict | null` — renders skeleton if null, full content if set.

- [ ] **Step 4: Run — PASS**

- [ ] **Step 5: Commit**

```bash
git add src/frontend/src/components/verify/DocumentVerdictCard.tsx src/frontend/src/components/verify/__tests__/DocumentVerdictCard.test.tsx
git commit -m "feat(verify): DocumentVerdictCard"
```

### Task 2.3: `ClaimCard` + `EvidenceDrawer`

**Files:**
- Create: `src/frontend/src/components/verify/ClaimCard.tsx`
- Create: `src/frontend/src/components/verify/EvidenceDrawer.tsx`
- Create: `src/frontend/src/components/verify/__tests__/ClaimCard.test.tsx`

- [ ] **Step 1: Tests**:
  - renders claim text, p_true, interval mini-bar, domain badge
  - shows `FallbackBadge` when `fallback_used != null`
  - shows "N PCG neighbors" when `pcg_neighbors.length > 0`
  - "Expand evidence" toggles `EvidenceDrawer`
  - emits `onShowInGraph(claim.id)` when "Show in graph" clicked
  - emits `onFlag(claim.id)` when flag clicked

- [ ] **Step 2: Run — FAIL**

- [ ] **Step 3: Implement** both components. `EvidenceDrawer` splits `supporting_evidence` vs `refuting_evidence` into two columns with source URIs as links + credibility score.

- [ ] **Step 4: Run — PASS**

- [ ] **Step 5: Commit**

```bash
git add src/frontend/src/components/verify/ClaimCard.tsx src/frontend/src/components/verify/EvidenceDrawer.tsx src/frontend/src/components/verify/__tests__/ClaimCard.test.tsx
git commit -m "feat(verify): ClaimCard + EvidenceDrawer"
```

### Task 2.4: `ClaimList` with sort dropdown

**Files:**
- Create: `src/frontend/src/components/verify/ClaimList.tsx`
- Create: `src/frontend/src/components/verify/__tests__/ClaimList.test.tsx`

- [ ] **Step 1: Tests**: renders cards for each claim; sort dropdown changes render order.

- [ ] **Step 2-4:** TDD loop.

- [ ] **Step 5: Commit**

```bash
git add src/frontend/src/components/verify/ClaimList.tsx src/frontend/src/components/verify/__tests__/ClaimList.test.tsx
git commit -m "feat(verify): ClaimList with sort by id|p_true|information_gain"
```

### Task 2.5: `FeedbackButton` + `FeedbackSheet`

**Files:**
- Create: `src/frontend/src/components/verify/FeedbackButton.tsx`
- Create: `src/frontend/src/components/verify/FeedbackSheet.tsx`
- Create: `src/frontend/src/components/verify/__tests__/FeedbackSheet.test.tsx`

- [ ] **Step 1: Tests** — opens on click, renders radios, enforces 2000-char rationale limit, calls `useFeedbackMutation` on submit, idempotency key = `(request_id, claim_id, labelerId)` where `labelerId` is from `localStorage.getItem("ohi:labeler-id") ?? crypto.randomUUID()` persisted back.

- [ ] **Step 2-4:** TDD loop.

- [ ] **Step 5: Commit**

```bash
git add src/frontend/src/components/verify/FeedbackButton.tsx src/frontend/src/components/verify/FeedbackSheet.tsx src/frontend/src/components/verify/__tests__/FeedbackSheet.test.tsx
git commit -m "feat(verify): FeedbackButton + FeedbackSheet POSTing /feedback"
```

### Task 2.6: `VerifyForm` + option chips

**Files:**
- Create: `src/frontend/src/components/verify/VerifyForm.tsx`
- Create: `src/frontend/src/components/verify/__tests__/VerifyForm.test.tsx`

- [ ] **Step 1: Tests** — enforces 50 000 char limit, emits `onSubmit(VerifyRequest)` with chosen rigor + domain_hint + coverage_target, disables during streaming, transitions to "Cancel" button during streaming and calls `onCancel`.

- [ ] **Step 2-4:** TDD loop.

- [ ] **Step 5: Commit**

```bash
git add src/frontend/src/components/verify/VerifyForm.tsx src/frontend/src/components/verify/__tests__/VerifyForm.test.tsx
git commit -m "feat(verify): VerifyForm with rigor + domain + coverage chips"
```

### Task 2.7: `SseProgress` timeline

**Files:**
- Create: `src/frontend/src/components/verify/SseProgress.tsx`
- Create: `src/frontend/src/components/verify/__tests__/SseProgress.test.tsx`

- [ ] **Step 1: Tests** — given a progress bag, renders one step per pipeline stage with pending/active/done state; transitions as bag updates.

- [ ] **Step 2-4:** TDD loop.

- [ ] **Step 5: Commit**

```bash
git add src/frontend/src/components/verify/SseProgress.tsx src/frontend/src/components/verify/__tests__/SseProgress.test.tsx
git commit -m "feat(verify): SseProgress pipeline timeline"
```

### Task 2.8: `PcgGraph` wrapper

**Files:**
- Create: `src/frontend/src/components/verify/PcgGraph.tsx`

This one is largely visual; TDD on the graph itself is low-value. Smoke-render only.

- [ ] **Step 1: Smoke test**

```tsx
// src/frontend/src/components/verify/__tests__/PcgGraph.test.tsx
// Mock react-force-graph-3d; assert it receives a graphData with nodes.length === claims.length
// and edges.length === unique (claimId, neighborId) pairs across all pcg_neighbors.
```

- [ ] **Step 2-4:** TDD loop.

- [ ] **Step 3: Implement** — wraps `react-force-graph-3d`, computes `graphData` from `claims: ClaimVerdict[]`. Node color by `p_true` band (same palette as `IntervalBar`), size by `information_gain`. Edges from union of `pcg_neighbors`, color by `edge_type`. Exposes imperative `focusNode(claimId)` via `useImperativeHandle`. Default hides neutral edges (toggle in UI controls).

- [ ] **Step 5: Commit**

```bash
git add src/frontend/src/components/verify/PcgGraph.tsx src/frontend/src/components/verify/__tests__/PcgGraph.test.tsx
git commit -m "feat(verify): PcgGraph 3D force-graph wrapper"
```

### Task 2.9: `VerifyPage` — top-level composition

**Files:**
- Create: `src/frontend/src/components/verify/VerifyPage.tsx`
- Create: `src/frontend/src/components/verify/__tests__/VerifyPage.integration.test.tsx`

- [ ] **Step 1: Integration test** — uses MSW to stream the SSE fixture, asserts:
  - `SseProgress` shows progression
  - `ClaimCard`s appear as `claim_verdict` events arrive (in order)
  - `DocumentVerdictCard` appears on `document_verdict` event
  - On simulated 503 resting response, `RestingState` appears in place of the verdict panel (will fail until Task 3.1 — mark as `it.skip`, un-skip in Task 3.6)

- [ ] **Step 2-4:** TDD loop.

- [ ] **Step 3: Implement** per spec §4. Two-column layout ≥1024px, single-column <1024px. Uses `useVerifyController()`.

- [ ] **Step 5: Commit**

```bash
git add src/frontend/src/components/verify/VerifyPage.tsx src/frontend/src/components/verify/__tests__/VerifyPage.integration.test.tsx
git commit -m "feat(verify): VerifyPage top-level composition"
```

---

## Phase 3 — Error / edge state components

### Task 3.1: `RestingState`

**Files:**
- Create: `src/frontend/src/components/common/RestingState.tsx`
- Create: `src/frontend/src/components/common/__tests__/RestingState.test.tsx`

- [ ] **Step 1: Test** — renders message; countdown uses prop `retryAfterSec`; retry button disabled while counting; clicking retry when enabled fires `onRetry`.

- [ ] **Step 2-4:** TDD loop. Use `vi.useFakeTimers()` for countdown.

- [ ] **Step 5: Commit**

```bash
git add src/frontend/src/components/common/RestingState.tsx src/frontend/src/components/common/__tests__/RestingState.test.tsx
git commit -m "feat(common): RestingState"
```

### Task 3.2-3.5: `BudgetExhaustedState`, `LlmUnavailableState`, `RateLimitedState`, `NetworkErrorState`

Same TDD pattern as 3.1; copy differs per spec §5. One task per file. Each gets its own commit.

### Task 3.6: `DegradedState` banner + `StatusDot`

**Files:**
- Create: `src/frontend/src/components/common/DegradedState.tsx`
- Create: `src/frontend/src/components/common/StatusDot.tsx`
- Create: `src/frontend/src/components/common/__tests__/DegradedState.test.tsx`
- Create: `src/frontend/src/components/common/__tests__/StatusDot.test.tsx`

- [ ] **Step 1: Tests**:
  - `DegradedState` — renders affected-layer chips; shows banner when `fallback_used` on any claim OR `degraded_layers.length > 0`
  - `StatusDot` — uses `useHealthDeep()`; green when all systems up, amber on any degraded layer, red on fetch failure or 503; 30s polling

- [ ] **Step 2-4:** TDD loop.

- [ ] **Step 5: Un-skip the VerifyPage integration test for resting state** (Task 2.9 step 1). Rerun: `npm run test:run -- VerifyPage`. PASS.

- [ ] **Step 6: Commit**

```bash
git add src/frontend/src/components/common/DegradedState.tsx src/frontend/src/components/common/StatusDot.tsx src/frontend/src/components/common/__tests__/
git commit -m "feat(common): DegradedState + StatusDot"
```

---

## Phase 4 — New routes

### Task 4.1: `/verify` route

**Files:**
- Create: `src/frontend/src/app/verify/page.tsx`
- Create: `src/frontend/src/app/verify/loading.tsx`

- [ ] **Step 1: Implement** — thin route file that renders `<VerifyPage />`. Add metadata: title, description, og image.

- [ ] **Step 2: Playwright e2e**

Create `src/frontend/e2e/verify-streaming.spec.ts`:
- Navigate to `/verify`
- Mock API via `page.route('**/api/v2/verify/stream', ...)` with fixture stream
- Fill textarea, click verify
- Assert claim cards render in order; DocumentVerdictCard appears

Create `src/frontend/e2e/verify-resting.spec.ts`:
- Mock 503 `{status: "resting"}`, `Retry-After: 60`
- Assert `RestingState` appears with countdown

- [ ] **Step 3: Run**

```bash
cd src/frontend && npm run test:e2e -- verify
```

- [ ] **Step 4: Commit**

```bash
git add src/frontend/src/app/verify/ src/frontend/e2e/verify-*.spec.ts
git commit -m "feat(app): /verify route with streaming + resting e2e"
```

### Task 4.2: `/calibration` route + `CalibrationTable`

**Files:**
- Create: `src/frontend/src/components/calibration/CalibrationTable.tsx`
- Create: `src/frontend/src/components/calibration/__tests__/CalibrationTable.test.tsx`
- Create: `src/frontend/src/app/calibration/page.tsx`
- Create: `src/frontend/e2e/calibration.spec.ts`

- [ ] **Step 1: Test** `CalibrationTable` with a fixture (small `CalibrationReport` object) — renders per-domain rows, interval widths, empirical coverage.

- [ ] **Step 2-4:** TDD loop. Route page uses `useCalibration()` + renders table with `RestingState`/`NetworkErrorState` on error.

- [ ] **Step 5: Playwright e2e** — smoke.

- [ ] **Step 6: Commit**

```bash
git add src/frontend/src/components/calibration/ src/frontend/src/app/calibration/ src/frontend/e2e/calibration.spec.ts
git commit -m "feat(app): /calibration transparency page"
```

### Task 4.3: `/status` route + `HealthMatrix`

**Files:**
- Create: `src/frontend/src/components/status/HealthMatrix.tsx`
- Create: `src/frontend/src/components/status/__tests__/HealthMatrix.test.tsx`
- Create: `src/frontend/src/app/status/page.tsx`
- Create: `src/frontend/e2e/status.spec.ts`

- [ ] **Step 1: Test** `HealthMatrix` — per-layer rows with live/ready/latency; green/amber/red per row.

- [ ] **Step 2-4:** TDD loop. Route uses `useHealthDeep()` with 10s polling. Renders `RestingState` when the endpoint itself is unreachable (treat as PC-off).

- [ ] **Step 5: Playwright e2e** — smoke.

- [ ] **Step 6: Commit**

```bash
git add src/frontend/src/components/status/ src/frontend/src/app/status/ src/frontend/e2e/status.spec.ts
git commit -m "feat(app): /status health dashboard"
```

---

## Phase 5 — Landing retune + layout trim

### Task 5.1: Landing copy retune

**Files:**
- Modify: `src/frontend/src/components/landing/HeroSection.tsx`
- Modify: `src/frontend/src/components/landing/ProblemSection.tsx`
- Modify: `src/frontend/src/components/landing/ArchitectureFlow.tsx`
- Modify: `src/frontend/src/components/landing/FeatureGrid.tsx`
- Modify: `src/frontend/src/components/landing/CtaSection.tsx`

- [ ] **Step 1: Rewrite body copy per spec §7** — text-only changes. Do NOT touch animation code, JSX structure, component imports, or styling.

- [ ] **Step 2: Update existing landing tests** — snapshot tests will shift; update expected text assertions.

- [ ] **Step 3: Run**

```bash
cd src/frontend && npm run test:run -- landing
```

- [ ] **Step 4: Visual smoke** — `npm run dev`, browse `/`, confirm 5 sections render, no broken imports.

- [ ] **Step 5: Commit**

```bash
git add src/frontend/src/components/landing/
git commit -m "refactor(landing): retune copy to v2 pitch (calibration, PCG, transparency)"
```

### Task 5.2: Navbar trim + StatusDot

**Files:**
- Modify: `src/frontend/src/components/layout/navbar.tsx`

- [ ] **Step 1: Remove** Login / Signup / Pricing / Dashboard links.
- [ ] **Step 2: Add** `/verify` `/calibration` `/status` `/about` links.
- [ ] **Step 3: Mount** `<StatusDot />` on the right.
- [ ] **Step 4: Update tests** in `src/components/layout/__tests__/navbar.test.tsx` (if present).
- [ ] **Step 5: Run all tests** — green.
- [ ] **Step 6: Commit**

```bash
git add src/frontend/src/components/layout/navbar.tsx src/frontend/src/components/layout/__tests__/
git commit -m "refactor(navbar): v2 links + StatusDot; drop Login/Signup/Pricing"
```

### Task 5.3: Footer trim + CookieConsent simplify + metadata

**Files:**
- Modify: `src/frontend/src/components/layout/footer.tsx`
- Modify: `src/frontend/src/components/layout/cookie-consent.tsx`
- Modify: `src/frontend/src/app/layout.tsx` (metadata description)

- [ ] **Step 1: Footer** — drop Pricing/Login/Sign up. Keep German legal + Disclaimer/EULA/Cookies/Accessibility.
- [ ] **Step 2: CookieConsent** — categories reduced to Essential + Analytics (Vercel). Remove any Supabase/Stripe-era cookie category references.
- [ ] **Step 3: Layout metadata** — update `description` to match new hero.
- [ ] **Step 4: Run tests**.
- [ ] **Step 5: Commit**

```bash
git add src/frontend/src/components/layout/footer.tsx src/frontend/src/components/layout/cookie-consent.tsx src/frontend/src/app/layout.tsx
git commit -m "refactor(layout): footer/cookie/metadata trim for v2"
```

---

## Phase 6 — Deletions + package cleanup + config

### Task 6.1: Delete `app/api/*` directories

**Files (delete):**
- `src/frontend/src/app/api/auth/`
- `src/frontend/src/app/api/checkout/`
- `src/frontend/src/app/api/webhooks/`
- `src/frontend/src/app/api/admin/`
- `src/frontend/src/app/api/tokens/`
- `src/frontend/src/app/api/ohi/`

- [ ] **Step 1: Delete directories**

Run: `cd src/frontend && git rm -r src/app/api/auth src/app/api/checkout src/app/api/webhooks src/app/api/admin src/app/api/tokens src/app/api/ohi`

- [ ] **Step 2: Typecheck**

Run: `cd src/frontend && npx tsc --noEmit` → expect errors referencing those routes; follow imports back.

- [ ] **Step 3: Fix any broken imports** — likely none outside the deleted dirs at this point since nothing new imports them.

- [ ] **Step 4: Run all tests + build**

```bash
npm run test:run && npm run build
```

- [ ] **Step 5: Commit**

```bash
git commit -m "chore(api): delete auth/checkout/webhooks/admin/tokens/ohi server routes"
```

### Task 6.2: Delete page directories

**Files (delete):**
- `src/frontend/src/app/auth/`
- `src/frontend/src/app/pricing/`
- `src/frontend/src/app/admin/`
- `src/frontend/src/app/tokens/`
- `src/frontend/src/app/dashboard/`

- [ ] **Step 1: Delete**

```bash
cd src/frontend && git rm -r src/app/auth src/app/pricing src/app/admin src/app/tokens src/app/dashboard
```

- [ ] **Step 2: Typecheck + build + test**

- [ ] **Step 3: Commit**

```bash
git commit -m "chore(app): delete auth/pricing/admin/tokens/dashboard routes"
```

### Task 6.3: Delete lib/ Supabase + Stripe + db + api

**Files (delete):**
- `src/frontend/src/lib/supabase/`
- `src/frontend/src/lib/stripe.ts`
- `src/frontend/src/lib/db/`
- `src/frontend/src/lib/api.ts`
- `src/frontend/src/components/dashboard/`
- `src/frontend/drizzle.config.ts`

- [ ] **Step 1: Verify `src/frontend/src/proxy.ts` is an orphan**

Run: `cd src/frontend && grep -rn "from.*proxy" src/ | grep -v "api/ohi"` → expect empty. If non-empty, keep the file; if empty, include it in deletion.

- [ ] **Step 2: Delete**

```bash
cd src/frontend && git rm -r src/lib/supabase src/lib/stripe.ts src/lib/db src/lib/api.ts src/components/dashboard drizzle.config.ts
# include src/proxy.ts if confirmed orphan
```

- [ ] **Step 3: Typecheck + build + test**

- [ ] **Step 4: Commit**

```bash
git commit -m "chore(lib): delete Supabase/Stripe/Drizzle + v1 api client + dashboard components"
```

### Task 6.4: Remove deps from package.json + run install

**Files:**
- Modify: `src/frontend/package.json`

- [ ] **Step 1: Remove** from `dependencies`: `@supabase/ssr`, `@supabase/supabase-js`, `@stripe/stripe-js`, `stripe`, `drizzle-orm`, `postgres`. Remove from `devDependencies`: `drizzle-kit`. Remove scripts: `db:generate`, `db:migrate`, `db:push`, `db:studio`.

- [ ] **Step 2: Install**

Run: `cd src/frontend && npm install`
Expected: clean, `package-lock.json` updated.

- [ ] **Step 3: Full test sweep**

```bash
npm run lint && npm run test:run && npm run build
```

- [ ] **Step 4: Commit**

```bash
git add src/frontend/package.json src/frontend/package-lock.json
git commit -m "chore(deps): remove Supabase/Stripe/Drizzle/Postgres"
```

### Task 6.5: Update `next.config.ts` for static export + `sitemap.ts`/`robots.ts`/`.env.example`

**Files:**
- Modify: `src/frontend/next.config.ts`
- Modify: `src/frontend/src/app/sitemap.ts`
- Modify: `src/frontend/src/app/robots.ts`
- Create: `src/frontend/.env.example`
- Modify: `src/frontend/vitest.config.ts`

- [ ] **Step 1: `next.config.ts`** — add `output: 'export'`; remove the `async headers()` block (doesn't apply to static export; the CSP will move to a `<meta http-equiv="Content-Security-Policy">` tag in the root layout if we want to retain it — for now, **document as a follow-up**, don't port).

- [ ] **Step 2: `sitemap.ts`** — update route list to `/`, `/verify`, `/calibration`, `/status`, `/about`, `/accessibility`, `/cookies`, `/disclaimer`, `/eula`, `/agb`, `/datenschutz`, `/impressum`.

- [ ] **Step 3: `robots.ts`** — confirm still valid for static; adjust if needed.

- [ ] **Step 4: Create `.env.example`**:

```env
# Public API base URL (Cloudflare-fronted Lambda)
NEXT_PUBLIC_API_BASE=https://api.ohi.shiftbloom.studio/api/v2

# Public site URL (used in metadata / sitemap / OG)
NEXT_PUBLIC_SITE_URL=https://ohi.shiftbloom.studio

# Optional: show raw-JSON debug drawer on /verify
NEXT_PUBLIC_SHOW_DEBUG=false

# ── Local dev (point at local FastAPI) ──
# NEXT_PUBLIC_API_BASE=http://localhost:8080/api/v2
```

- [ ] **Step 5: `vitest.config.ts`** — remove `src/lib/db/index.ts`, `src/lib/stripe.ts`, `src/lib/supabase/**` from coverage excludes (paths no longer exist); remove `src/components/dashboard/**/*.tsx` from includes; add `src/lib/ohi-*.ts`, `src/lib/sse.ts`, `src/lib/verify-controller.ts`, `src/components/verify/**/*.tsx`, `src/components/common/**/*.tsx`.

- [ ] **Step 6: Build**

Run: `cd src/frontend && npm run build`
Expected: `out/` directory populated with static artifacts.

- [ ] **Step 7: Commit**

```bash
git add src/frontend/next.config.ts src/frontend/src/app/sitemap.ts src/frontend/src/app/robots.ts src/frontend/.env.example src/frontend/vitest.config.ts
git commit -m "chore(config): static export + v2 sitemap + .env.example"
```

---

## Phase 7 — Final verification gate

### Task 7.1: Full test + build sweep

- [ ] **Step 1: Lint**

```bash
cd src/frontend && npm run lint
```
Expected: exit 0.

- [ ] **Step 2: Unit + integration tests**

```bash
npm run test:run
```
Expected: all green.

- [ ] **Step 3: E2E tests**

```bash
npm run test:e2e
```
Expected: all green.

- [ ] **Step 4: Build**

```bash
npm run build
```
Expected: exit 0, `out/` populated with `.html` files for every public route.

### Task 7.2: Orphan + residual check

- [ ] **Step 1: No Supabase/Stripe/Drizzle leftovers**

```bash
cd src/frontend && git grep -i "supabase\|stripe\|drizzle"
```
Expected: empty (or only matches inside `.env.example` comments that are legitimate).

- [ ] **Step 2: No dead imports**

```bash
cd src/frontend && npx tsc --noEmit --strict
```

- [ ] **Step 3: Manual smoke test against local FastAPI**

Precondition: local FastAPI up at `http://localhost:8080`.

```bash
cd src/frontend
echo "NEXT_PUBLIC_API_BASE=http://localhost:8080/api/v2" > .env.local
echo "NEXT_PUBLIC_SITE_URL=http://localhost:3000" >> .env.local
npm run dev
```

Browser checks:
- `http://localhost:3000/` — landing loads, 5 sections visible, animations play
- `http://localhost:3000/verify` — form renders; submit short text; if API healthy, DocumentVerdictCard appears and claim cards stream in
- `http://localhost:3000/verify` — stop local FastAPI and retry; assert `RestingState` renders
- `http://localhost:3000/calibration` — table renders with data from live endpoint
- `http://localhost:3000/status` — HealthMatrix shows per-layer status
- `http://localhost:3000/agb`, `/datenschutz`, `/impressum` — render unchanged

- [ ] **Step 4: Remove scratch notes**

```bash
rm src/frontend/MIGRATION_NOTES.md 2>/dev/null || true
```

- [ ] **Step 5: Final commit (if anything leftover)**

If any env/config tweaks emerged from the smoke test:

```bash
git add -A src/frontend/
git commit -m "chore(frontend): final polish after smoke test"
```

Otherwise, nothing to commit — implementation complete.

---

## Handoff notes (for post-implementation)

After this plan is complete, coordinate these follow-ups (NOT part of this plan):

1. **Aesthetic pass** — separate session invoking `superpowers:frontend-design` to rework visual identity per the "not-a-100th-SaaS" direction saved as project feedback memory. This reworks typography, color, motion, layout hierarchy on top of the wired base.
2. **Backend CORS + OPTIONS exemption** — FastAPI on branch `feat/ohi-v2-foundation`: add `CORSMiddleware` with `allow_origins=["https://ohi.shiftbloom.studio"]`, exempt `OPTIONS` from `EdgeSecretMiddleware`. Needs a test for a preflight call.
3. **Backend SSE disconnect** — handle `await request.is_disconnected()` in `/verify/stream` to cancel the pipeline when the client aborts.
4. **Infra DNS flip** — infra agent applies spec §1.2 DNS changes (apex → Vercel, `api.` → Lambda + Transform Rule rescope).
5. **Vercel project linkage** — out of this plan's scope; user decides when to create the Vercel project and link repo.
6. **CSP for static export** — port security headers from the deleted `next.config.ts` headers block to a `<meta http-equiv="Content-Security-Policy">` tag in `app/layout.tsx` if retention is desired post-deploy.

End of plan.
