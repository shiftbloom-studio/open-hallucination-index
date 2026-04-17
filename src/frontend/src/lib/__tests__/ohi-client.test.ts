import { describe, it, expect, beforeAll, afterAll, afterEach } from "vitest";
import { setupServer } from "msw/node";
import { http, HttpResponse } from "msw";
import { ohi, OhiError } from "../ohi-client";
import golden from "../../test/fixtures/document-verdict.golden.json";

const API_BASE = "https://api.ohi.test/api/v2";

const server = setupServer();

beforeAll(() => {
  process.env.NEXT_PUBLIC_API_BASE = API_BASE;
  server.listen({ onUnhandledRequest: "error" });
});
afterEach(() => server.resetHandlers());
afterAll(() => server.close());

describe("ohi-client", () => {
  it("verify() returns DocumentVerdict on 200", async () => {
    server.use(http.post(`${API_BASE}/verify`, () => HttpResponse.json(golden)));
    const res = await ohi.verify({ text: "hi" });
    expect(res.document_score).toBeCloseTo((golden as { document_score: number }).document_score);
    expect(res.claims).toHaveLength(2);
  });

  it("OhiError.isResting on 503 {status:'resting'}", async () => {
    server.use(
      http.post(`${API_BASE}/verify`, () =>
        HttpResponse.json(
          { status: "resting", reason: "origin-unreachable" },
          { status: 503, headers: { "Retry-After": "300" } },
        ),
      ),
    );
    try {
      await ohi.verify({ text: "hi" });
      throw new Error("expected rejection");
    } catch (e) {
      expect(e).toBeInstanceOf(OhiError);
      const err = e as OhiError;
      expect(err.isResting).toBe(true);
      expect(err.isLlmDown).toBe(false);
      expect(err.retryAfterSec).toBe(300);
    }
  });

  it("OhiError.isLlmDown on 503 {status:'llm_unavailable'}", async () => {
    server.use(
      http.post(`${API_BASE}/verify`, () =>
        HttpResponse.json({ status: "llm_unavailable" }, { status: 503 }),
      ),
    );
    await expect(ohi.verify({ text: "hi" })).rejects.toSatisfy(
      (e: unknown) => e instanceof OhiError && e.isLlmDown && !e.isResting,
    );
  });

  it("OhiError.isBudgetExhausted when detail mentions it", async () => {
    server.use(
      http.post(`${API_BASE}/verify`, () =>
        HttpResponse.json(
          { detail: "OHI public budget exhausted, resets in 7h" },
          { status: 503 },
        ),
      ),
    );
    await expect(ohi.verify({ text: "hi" })).rejects.toSatisfy(
      (e: unknown) => e instanceof OhiError && e.isBudgetExhausted,
    );
  });

  it("OhiError.isDegraded when degraded_layers present", async () => {
    server.use(
      http.post(`${API_BASE}/verify`, () =>
        HttpResponse.json({ degraded_layers: ["L3", "L1.retrieval"] }, { status: 503 }),
      ),
    );
    await expect(ohi.verify({ text: "hi" })).rejects.toSatisfy(
      (e: unknown) => e instanceof OhiError && e.isDegraded,
    );
  });

  it("OhiError.isRateLimited on 429 with Retry-After", async () => {
    server.use(
      http.post(`${API_BASE}/verify`, () =>
        HttpResponse.json(
          { detail: "rate limited" },
          { status: 429, headers: { "Retry-After": "48" } },
        ),
      ),
    );
    try {
      await ohi.verify({ text: "x" });
      throw new Error("expected rejection");
    } catch (e) {
      expect(e).toBeInstanceOf(OhiError);
      const err = e as OhiError;
      expect(err.isRateLimited).toBe(true);
      expect(err.retryAfterSec).toBe(48);
    }
  });

  it("throws missing-env error when NEXT_PUBLIC_API_BASE is unset", async () => {
    const saved = process.env.NEXT_PUBLIC_API_BASE;
    delete process.env.NEXT_PUBLIC_API_BASE;
    try {
      await expect(ohi.verify({ text: "x" })).rejects.toThrow(/NEXT_PUBLIC_API_BASE/);
    } finally {
      process.env.NEXT_PUBLIC_API_BASE = saved;
    }
  });

  it("feedback() returns queued:true on 202", async () => {
    server.use(
      http.post(`${API_BASE}/feedback`, () =>
        HttpResponse.json({ feedback_id: "fb1", queued: true }, { status: 202 }),
      ),
    );
    const res = await ohi.feedback({
      request_id: "r1",
      claim_id: "c1",
      label: "false",
      labeler: { kind: "user", id: "u1", credential_level: 0 },
      rationale: "wrong year",
    });
    expect(res.feedback_id).toBe("fb1");
    expect(res.queued).toBe(true);
  });

  it("calibrationReport() returns parsed JSON", async () => {
    const fixture = {
      report_date: "2026-04-16",
      global_coverage_target: 0.9,
      domains: {},
    };
    server.use(http.get(`${API_BASE}/calibration/report`, () => HttpResponse.json(fixture)));
    const res = await ohi.calibrationReport();
    expect(res.report_date).toBe("2026-04-16");
  });
});
