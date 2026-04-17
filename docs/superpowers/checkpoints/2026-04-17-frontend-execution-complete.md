# OHI v2 — Frontend Sub-Project Execution Complete (Handoff)

**Date:** 2026-04-17
**Purpose:** Close out sub-project 3 of 3 (Next.js frontend touch-up). Status + decisions record.
**Companions:** algorithm spec + infra spec + infra execution checkpoint.

---

## 1. Scope delivered

Implementation on `feat/ohi-v2-foundation` (local, not pushed). Strictly code-only — zero infra/Terraform/Vercel/backend activity per user directive ("time pressure → implementation phase now, only Next.js/React").

### Phases complete

- **Phase 0** — baseline verified; v2 golden fixtures committed
- **Phase 1** — foundation lib: `ohi-types.ts` (zod-validated vs. algorithm spec §9 golden), `ohi-client.ts` with `OhiError` classification, `sse.ts` POST-SSE parser (+CRLF/chunk-split/partial-frame tolerance), `verify-controller.ts` reducer state machine, `ohi-queries.ts` TanStack hooks
- **Phase 2** — verify-page components: `IntervalBar`, `DomainBadge`, `FallbackBadge`, `DocumentVerdictCard`, `ClaimCard`, `EvidenceDrawer`, `ClaimList` (sortable), `FeedbackButton`+`FeedbackSheet` (modal POSTing `/feedback` with localStorage labeler id), `VerifyForm`, `SseProgress`, `PcgGraph` (wrapping `react-force-graph-3d`), `VerifyPage` composition
- **Phase 3** — error-state surfaces: `RestingState`, `BudgetExhaustedState`, `LlmUnavailableState`, `RateLimitedState`, `NetworkErrorState`, `DegradedState`, `StatusDot`, shared `RetryCountdown`
- **Phase 4** — new routes: `/verify`, `/calibration` (+ `CalibrationTable`), `/status` (+ `HealthMatrix`); server-metadata + client-content split for static export
- **Phase 5** — landing copy retune per spec §7 (HeroSection, ProblemSection, ArchitectureFlow, FeatureGrid, CtaSection), Navbar trimmed to `/verify`, `/calibration`, `/status`, `/about` + `StatusDot`; root metadata description updated
- **Phase 6** — deletion of `app/auth/`, `app/pricing/`, `app/admin/`, `app/tokens/`, `app/dashboard/`, `app/api/*` (auth/checkout/webhooks/admin/tokens/ohi); removal of `lib/supabase/`, `lib/stripe.ts`, `lib/db/`, `lib/api.ts`, `src/proxy.ts`, `drizzle.config.ts`, v1 e2e specs; `package.json` cleanup (Supabase/Stripe/Drizzle/Postgres deps + `db:*` scripts removed); `next.config.ts` rewritten for `output: 'export'`; `sitemap.ts` + `robots.ts` updated with v2 route list + `dynamic = "force-static"`; fresh `.env.example`; `vitest.config.ts` coverage targets refocused
- **Phase 7** — lint ✓, 128/128 vitest tests ✓, static build ✓ (15 static pages under `src/frontend/out/`)

### Commits on `feat/ohi-v2-foundation` (18 new, on top of the infra execution branch)

```
1f8ffd2  chore(frontend): delete auth/pricing/admin/tokens/dashboard + Supabase/Stripe/Drizzle + v1 proxy
532164a  refactor(landing+layout): retune copy to v2 pitch; trim Navbar
4acca3e  feat(app): /verify, /calibration, /status routes + VerifyPage composition
79ba660  feat(common): error states, StatusDot, RetryCountdown
814f682  feat(verify): PcgGraph 3D force-graph wrapper
f6b59b4  feat(verify): SseProgress pipeline timeline
1be58fc  feat(verify): VerifyForm with rigor + domain + coverage chips
ff68f66  feat(verify): FeedbackButton + FeedbackSheet POSTing /feedback
68bbaa6  feat(verify): ClaimList with sort by id|p_true|information_gain
5fc1484  feat(verify): ClaimCard + EvidenceDrawer
ab88147  feat(verify): DocumentVerdictCard
4553136  feat(verify): IntervalBar + DomainBadge + FallbackBadge primitives
9ee3601  feat(queries): TanStack Query hooks for calibration + health + feedback
b6e68fd  feat(controller): verify page state machine
7c48cf1  feat(sse): POST-SSE client for /verify/stream
3ebd048  feat(client): v2 ohi-client with OhiError classification
8bc03ee  feat(types): v2 DocumentVerdict type + zod schema mirroring algorithm spec §9
97ff2ad  test(fixtures): add v2 DocumentVerdict + SSE golden fixtures
```

### Test suite state

- 128 tests pass across 16 test files
- Coverage focused on `lib/ohi-*`, `lib/sse`, `lib/verify-controller`, `components/verify/*`, `components/common/*`, `components/calibration/*`, `components/status/*`
- Two surviving v1 integration tests in `src/test/integration/*` continue to pass (MSW-driven HTTP tests, no Supabase coupling left)
- Playwright e2e reduced to a single `smoke.spec.ts` landing/navbar check

---

## 2. Deviations from the written plan

Recorded so you and future agents know what's **different** vs. the spec:

### 2.1 Plan deviations accepted during execution

1. **No spec-reviewer or plan-reviewer subagent loops.** User invoked time pressure explicitly. Spec went in at `docs/superpowers/specs/2026-04-17-ohi-v2-frontend-design.md`; plan at `docs/superpowers/plans/2026-04-17-ohi-v2-frontend-implementation.md`. Both on `main`.
2. **E2E suite deferred.** Plan tasks 4.1–4.3 specified Playwright specs for `verify-streaming`, `verify-resting`, `calibration`, `status`. Only `smoke.spec.ts` ships. Full e2e coverage is a post-touch-up follow-up.
3. **Radix Dialog replaced with a minimal custom modal.** Plan mentioned Radix Dialog for `FeedbackSheet`; it wasn't installed. Rather than add `@radix-ui/react-dialog`, the sheet uses a hand-rolled overlay + Escape handler — saves ~40 KB of JS and one dep. If a11y review flags something, switching later is trivial.
4. **`PcgGraph` tests cover `buildGraphData` (pure fn) only.** Rendering `react-force-graph-3d` under jsdom pulls in `three/webgpu` which blows up. The exported `buildGraphData` helper tests nodes/edges/dedupe logic directly. Visual rendering is verified manually via `npm run dev`.
5. **Sync-fallback path is partial.** State machine has the `sync_fallback` status wired, but the manual "Retry with sync fallback" button currently navigates to `#verdict=<id>`. A full dispatch of the fetched verdict into the controller's reducer is queued as follow-up — the 8s first-byte auto-fallback works, and the manual path is a graceful-degradation stub.
6. **CSP + security headers dropped from `next.config.ts`.** `async headers()` is incompatible with `output: 'export'`. The CSP and Permissions-Policy that were in v1 need to be recreated either as `<meta>` tags in `app/layout.tsx` or (better) at the Vercel / Cloudflare edge. Punt flagged below (§3.5).

### 2.2 Spec/plan deviations that should be re-examined

1. **Legal pages (`agb/`, `datenschutz/`, `impressum/`, `cookies/`) still describe Supabase + Stripe.** The spec said "German legal pages unchanged", but those pages document data processors (Supabase auth, Stripe checkout) that no longer exist in v2. `datenschutz/page.tsx` in particular names Supabase and Stripe as processors, and `cookies/page.tsx` lists their cookies. A German privacy policy that names non-existent processors is actively wrong under GDPR — **user should update copy before public launch**. Not auto-rewritten because this is legal text requiring human judgment.
2. **Accessibility statement** also mentions "payment processing via Stripe" — same issue, user-review needed.

---

## 3. Open follow-ups — not implemented, recorded for later

1. **Aesthetic pass** — saved as feedback memory `feedback_visual_design.md`. Plan: after ship, invoke `superpowers:frontend-design` for typography / motion / color-identity / hero-visual rework to move the UI off the generic SaaS-template aesthetic. Kept **out** of this sub-project because tangling visual identity with data wiring would double the review surface.
2. **Backend CORS + OPTIONS exemption on FastAPI.** For the cross-origin `api.ohi.shiftbloom.studio` → `ohi.shiftbloom.studio` shape, Lambda needs `CORSMiddleware(allow_origins=["https://ohi.shiftbloom.studio"])` and `EdgeSecretMiddleware` must bypass `OPTIONS`.
3. **Backend SSE disconnect handling.** Pipeline should cancel on `await request.is_disconnected()`. Otherwise client-side abort just wastes server compute.
4. **Infra DNS + Cloudflare flip** per spec §1.2 — apex → Vercel (DNS-only), `api.` → Lambda (proxied), Transform Rule scoped to `api.` host.
5. **Port CSP / security headers** to Vercel config (`vercel.json` headers block) or as `<meta http-equiv="Content-Security-Policy">` in the root layout.
6. **Legal-page copy refresh** per §2.2 above. User-authored; review required.
7. **Full Playwright coverage** — SSE streaming, resting state, calibration, status — deferred from this pass. Existing vitest + MSW integration tests cover the same logic; e2e is for cross-browser + real navigation.
8. **Sync-fallback completion path** — wire dispatch of fetched verdict into reducer instead of URL hash.
9. **Labeler-id persistence story** — currently a random UUID in localStorage. For the trusted-labeler tier in spec §11, a short-lived bearer token flow is needed. Not in Phase 1 scope.

---

## 4. What the next agent should NOT do

- Do not push `feat/ohi-v2-foundation` to remote without explicit user direction.
- Do not rewrite the German legal pages — the user needs to review and decide.
- Do not touch `infra/terraform/*` from a frontend context.
- Do not deploy to Vercel without the user's say-so.
- Do not reintroduce `next.config.ts` `async headers()` — it's incompatible with `output: 'export'`.

---

## 5. Artifact index

| Path | Purpose | Where it lives |
|---|---|---|
| `docs/superpowers/specs/2026-04-17-ohi-v2-frontend-design.md` | Frontend design spec | local `main` |
| `docs/superpowers/plans/2026-04-17-ohi-v2-frontend-implementation.md` | TDD implementation plan | local `main` |
| `docs/superpowers/checkpoints/2026-04-17-frontend-execution-complete.md` | **This file** | local `main` |
| Implementation commits | 18 commits above the infra execution tip | local `feat/ohi-v2-foundation` |
| Memory: visual-design stance | Aesthetic-pass intent preserved across sessions | `~/.claude/projects/.../memory/feedback_visual_design.md` |

End of handoff.
