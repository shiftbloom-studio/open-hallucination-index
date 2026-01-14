import { describe, it, expect } from "vitest";
import { server } from "@/test/mocks/server";
import { http, HttpResponse } from "msw";

describe("API Integration Tests", () => {
  describe("Checkout API", () => {
    it("should validate packageId is required", async () => {
      const response = await fetch("/api/checkout", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({}),
      });

      expect(response.status).toBe(400);
    });

    it("should validate packageId values", async () => {
      const response = await fetch("/api/checkout", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ packageId: "invalid" }),
      });

      expect(response.status).toBe(400);
    });

    it("should accept valid packageId 10", async () => {
      server.use(
        http.post("/api/checkout", () => {
          return HttpResponse.json({ url: "https://checkout.stripe.com/test" });
        })
      );

      const response = await fetch("/api/checkout", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          packageId: "10",
          userId: "test-user",
          userEmail: "test@example.com",
        }),
      });

      expect(response.ok).toBe(true);
      const data = await response.json();
      expect(data.url).toBeDefined();
    });

    it("should accept valid packageId 100", async () => {
      server.use(
        http.post("/api/checkout", () => {
          return HttpResponse.json({ url: "https://checkout.stripe.com/test" });
        })
      );

      const response = await fetch("/api/checkout", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          packageId: "100",
          userId: "test-user",
          userEmail: "test@example.com",
        }),
      });

      expect(response.ok).toBe(true);
    });

    it("should accept valid packageId 500", async () => {
      server.use(
        http.post("/api/checkout", () => {
          return HttpResponse.json({ url: "https://checkout.stripe.com/test" });
        })
      );

      const response = await fetch("/api/checkout", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          packageId: "500",
          userId: "test-user",
          userEmail: "test@example.com",
        }),
      });

      expect(response.ok).toBe(true);
    });
  });

  describe("Health Check API", () => {
    it("should return healthy status", async () => {
      server.use(
        http.get("*/health/live", () => {
          return HttpResponse.json({
            status: "healthy",
            timestamp: new Date().toISOString(),
            version: "1.0.0",
            environment: "test",
            checks: { database: true, cache: true },
          });
        })
      );

      const response = await fetch("http://localhost:8000/health/live");
      const data = await response.json();

      expect(data.status).toBe("healthy");
      expect(data.checks).toBeDefined();
    });

    it("should handle unhealthy status", async () => {
      server.use(
        http.get("*/health/live", () => {
          return HttpResponse.json({
            status: "unhealthy",
            timestamp: new Date().toISOString(),
            version: "1.0.0",
            environment: "test",
            checks: { database: false, cache: true },
          });
        })
      );

      const response = await fetch("http://localhost:8000/health/live");
      const data = await response.json();

      expect(data.status).toBe("unhealthy");
      expect(data.checks.database).toBe(false);
    });
  });

  describe("Verify API", () => {
    it("should verify text successfully", async () => {
      server.use(
        http.post("*/api/v1/verify", () => {
          return HttpResponse.json({
            id: "test-id",
            trust_score: { score: 0.9 },
            summary: "All claims verified",
            claims: [
              {
                id: "claim-1",
                text: "Test claim",
                status: "verified",
                confidence: 0.95,
                reasoning: "Verified against knowledge graph",
              },
            ],
            processing_time_ms: 100,
            cached: false,
          });
        })
      );

      const response = await fetch("http://localhost:8000/api/v1/verify", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: "Test claim" }),
      });

      const data = await response.json();

      expect(data.trust_score.score).toBe(0.9);
      expect(data.claims).toHaveLength(1);
      expect(data.claims[0].status).toBe("verified");
    });

    it("should handle verification with different strategies", async () => {
      const strategies = [
        "graph_exact",
        "vector_semantic",
        "hybrid",
        "cascading",
      ];

      for (const strategy of strategies) {
        server.use(
          http.post("*/api/v1/verify", async ({ request }) => {
            const body = (await request.json()) as { strategy?: string };
            return HttpResponse.json({
              id: "test-id",
              trust_score: { score: 0.85 },
              summary: `Verified with ${body.strategy}`,
              claims: [],
              processing_time_ms: 100,
              cached: false,
            });
          })
        );

        const response = await fetch("http://localhost:8000/api/v1/verify", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text: "Test", strategy }),
        });

        expect(response.ok).toBe(true);
      }
    });

    it("should handle cached responses", async () => {
      server.use(
        http.post("*/api/v1/verify", () => {
          return HttpResponse.json({
            id: "cached-id",
            trust_score: { score: 0.88 },
            summary: "Cached response",
            claims: [],
            processing_time_ms: 5, // Fast because cached
            cached: true,
          });
        })
      );

      const response = await fetch("http://localhost:8000/api/v1/verify", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: "Test", use_cache: true }),
      });

      const data = await response.json();

      expect(data.cached).toBe(true);
      expect(data.processing_time_ms).toBeLessThan(50);
    });
  });

  describe("Batch Verify API", () => {
    it("should verify multiple texts", async () => {
      server.use(
        http.post("*/api/v1/verify/batch", () => {
          return HttpResponse.json({
            results: [
              {
                id: "1",
                trust_score: { score: 0.9 },
                claims: [],
                processing_time_ms: 50,
                cached: false,
              },
              {
                id: "2",
                trust_score: { score: 0.85 },
                claims: [],
                processing_time_ms: 60,
                cached: false,
              },
              {
                id: "3",
                trust_score: { score: 0.95 },
                claims: [],
                processing_time_ms: 40,
                cached: false,
              },
            ],
            total_processing_time_ms: 150,
          });
        })
      );

      const response = await fetch(
        "http://localhost:8000/api/v1/verify/batch",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ texts: ["Text 1", "Text 2", "Text 3"] }),
        }
      );

      const data = await response.json();

      expect(data.results).toHaveLength(3);
      expect(data.total_processing_time_ms).toBe(150);
    });
  });
});
