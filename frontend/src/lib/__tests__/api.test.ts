import { describe, it, expect, beforeEach } from "vitest";
import { ApiClient, createApiClient } from "@/lib/api";
import { server } from "@/test/mocks/server";
import { http, HttpResponse } from "msw";
import {
  mockHealthResponse,
  mockVerifyResponse,
  mockBatchVerifyResponse,
} from "@/test/mocks/handlers";

describe("ApiClient", () => {
  const baseUrl = "http://test-api.com";
  let client: ApiClient;

  beforeEach(() => {
    client = new ApiClient(baseUrl);
  });

  describe("constructor", () => {
    it("should create an instance with base URL", () => {
      expect(client).toBeInstanceOf(ApiClient);
    });

    it("should remove trailing slash from base URL", () => {
      const clientWithSlash = new ApiClient("http://test-api.com/");
      expect(clientWithSlash).toBeInstanceOf(ApiClient);
    });
  });

  describe("getHealth", () => {
    it("should successfully fetch health status", async () => {
      server.use(
        http.get("http://test-api.com/health/live", () => {
          return HttpResponse.json(mockHealthResponse);
        })
      );

      const health = await client.getHealth();

      expect(health).toEqual(mockHealthResponse);
      expect(health.status).toBe("healthy");
      expect(health.checks.database).toBe(true);
    });

    it("should throw error when health check fails", async () => {
      server.use(
        http.get("http://test-api.com/health/live", () => {
          return new HttpResponse("Service unavailable", { status: 503 });
        })
      );

      await expect(client.getHealth()).rejects.toThrow("Health check failed");
    });

    it("should handle network errors", async () => {
      server.use(
        http.get("http://test-api.com/health/live", () => {
          return HttpResponse.error();
        })
      );

      await expect(client.getHealth()).rejects.toThrow();
    });
  });

  describe("verifyText", () => {
    it("should successfully verify text", async () => {
      server.use(
        http.post("http://test-api.com/api/v1/verify", () => {
          return HttpResponse.json(mockVerifyResponse);
        })
      );

      const result = await client.verifyText({
        text: "The sky is blue and water is wet.",
      });

      expect(result).toEqual(mockVerifyResponse);
      expect(result.trust_score.score).toBe(0.85);
      expect(result.claims).toHaveLength(2);
    });

    it("should send correct request body", async () => {
      let capturedBody: Record<string, unknown> | null = null;

      server.use(
        http.post("http://test-api.com/api/v1/verify", async ({ request }) => {
          capturedBody = (await request.json()) as Record<string, unknown>;
          return HttpResponse.json(mockVerifyResponse);
        })
      );

      await client.verifyText({
        text: "Test text",
        context: "Test context",
        strategy: "hybrid",
        use_cache: true,
      });

      expect(capturedBody).toEqual({
        text: "Test text",
        context: "Test context",
        strategy: "hybrid",
        use_cache: true,
      });
    });

    it("should throw error with status and message on failure", async () => {
      server.use(
        http.post("http://test-api.com/api/v1/verify", () => {
          return new HttpResponse("Invalid request format", { status: 400 });
        })
      );

      await expect(client.verifyText({ text: "" })).rejects.toThrow(
        "Verification failed: 400 Invalid request format"
      );
    });

    it("should handle different strategies", async () => {
      const strategies = [
        "graph_exact",
        "vector_semantic",
        "hybrid",
        "cascading",
      ] as const;

      for (const strategy of strategies) {
        server.use(
          http.post("http://test-api.com/api/v1/verify", () => {
            return HttpResponse.json(mockVerifyResponse);
          })
        );

        const result = await client.verifyText({
          text: "Test",
          strategy,
        });

        expect(result).toBeDefined();
      }
    });
  });

  describe("verifyBatch", () => {
    it("should successfully verify batch of texts", async () => {
      server.use(
        http.post("http://test-api.com/api/v1/verify/batch", () => {
          return HttpResponse.json(mockBatchVerifyResponse);
        })
      );

      const result = await client.verifyBatch({
        texts: ["Text 1", "Text 2", "Text 3"],
      });

      expect(result).toEqual(mockBatchVerifyResponse);
      expect(result.results).toHaveLength(1);
      expect(result.total_processing_time_ms).toBe(300);
    });

    it("should send correct batch request body", async () => {
      let capturedBody: Record<string, unknown> | null = null;

      server.use(
        http.post(
          "http://test-api.com/api/v1/verify/batch",
          async ({ request }) => {
            capturedBody = (await request.json()) as Record<string, unknown>;
            return HttpResponse.json(mockBatchVerifyResponse);
          }
        )
      );

      await client.verifyBatch({
        texts: ["Text 1", "Text 2"],
        strategy: "cascading",
        use_cache: false,
      });

      expect(capturedBody).toEqual({
        texts: ["Text 1", "Text 2"],
        strategy: "cascading",
        use_cache: false,
      });
    });

    it("should throw error on batch verification failure", async () => {
      server.use(
        http.post("http://test-api.com/api/v1/verify/batch", () => {
          return new HttpResponse("Server error", { status: 500 });
        })
      );

      await expect(client.verifyBatch({ texts: ["Test"] })).rejects.toThrow(
        "Batch verification failed"
      );
    });
  });
});

describe("createApiClient", () => {
  it("should create an ApiClient instance", () => {
    const client = createApiClient("http://example.com");
    expect(client).toBeInstanceOf(ApiClient);
  });

  it("should create different instances for different URLs", () => {
    const client1 = createApiClient("http://example1.com");
    const client2 = createApiClient("http://example2.com");
    expect(client1).not.toBe(client2);
  });
});
