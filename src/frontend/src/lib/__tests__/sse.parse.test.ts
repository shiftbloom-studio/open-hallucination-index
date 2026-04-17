import { describe, it, expect, beforeAll, afterAll, afterEach, vi } from "vitest";
import { setupServer } from "msw/node";
import { http, HttpResponse } from "msw";
// chunkedResponse returns a native Response (stream body). Cast via unknown
// because msw types only permit HttpResponse<BodyType> at the handler site.
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { parseFrame, streamVerify } from "../sse";
import type { SseEvent } from "../ohi-types";

describe("parseFrame", () => {
  it("parses a single event+data frame", () => {
    const evt = parseFrame(
      [
        "event: nli_complete",
        'data: {"claim_evidence_pairs_scored":24,"claim_pair_pairs_scored":2}',
      ].join("\n"),
    );
    expect(evt?.event).toBe("nli_complete");
    expect(evt?.data).toEqual({
      claim_evidence_pairs_scored: 24,
      claim_pair_pairs_scored: 2,
    });
  });

  it("tolerates CRLF line endings", () => {
    const evt = parseFrame(
      ["event: decomposition_complete", 'data: {"claim_count":3,"estimated_total_ms":60000}'].join(
        "\r\n",
      ),
    );
    expect(evt?.event).toBe("decomposition_complete");
  });

  it("returns null on malformed JSON", () => {
    const evt = parseFrame(["event: nli_complete", "data: {not json}"].join("\n"));
    expect(evt).toBeNull();
  });

  it("returns null on unknown event name", () => {
    const evt = parseFrame(["event: made_up_event", "data: {}"].join("\n"));
    expect(evt).toBeNull();
  });

  it("ignores comment lines (starting with ':') and blank lines", () => {
    const evt = parseFrame(
      [
        ": this is a comment",
        "",
        "event: nli_complete",
        'data: {"claim_evidence_pairs_scored":0,"claim_pair_pairs_scored":0}',
      ].join("\n"),
    );
    expect(evt?.event).toBe("nli_complete");
  });
});

const API_BASE = "https://api.ohi.test/api/v2";
const server = setupServer();

beforeAll(() => {
  process.env.NEXT_PUBLIC_API_BASE = API_BASE;
  server.listen({ onUnhandledRequest: "error" });
});
afterEach(() => server.resetHandlers());
afterAll(() => server.close());

function readFixture(): string {
  return readFileSync(
    resolve(__dirname, "../../test/fixtures/sse-event-stream.txt"),
    "utf-8",
  );
}

function chunkedResponse(chunks: string[]): Response {
  const encoder = new TextEncoder();
  const stream = new ReadableStream({
    async start(controller) {
      for (const chunk of chunks) {
        controller.enqueue(encoder.encode(chunk));
        // yield microtask to simulate chunked delivery
        await Promise.resolve();
      }
      controller.close();
    },
  });
  return new Response(stream, {
    headers: { "Content-Type": "text/event-stream" },
  });
}

describe("streamVerify", () => {
  it("emits all 8 events in order for the golden fixture delivered in one chunk", async () => {
    const text = readFixture();
    server.use(
      http.post(`${API_BASE}/verify/stream`, () =>
        chunkedResponse([text]) as unknown as HttpResponse<string>,
      ),
    );

    const events: SseEvent["event"][] = [];
    const onError = vi.fn();
    const onComplete = vi.fn();
    await streamVerify(
      { text: "hello" },
      {
        onEvent: (e) => events.push(e.event),
        onError,
        onComplete,
      },
      new AbortController().signal,
    );
    expect(onError).not.toHaveBeenCalled();
    expect(onComplete).toHaveBeenCalledOnce();
    expect(events).toEqual([
      "decomposition_complete",
      "claim_routed",
      "claim_routed",
      "nli_complete",
      "pcg_propagation_complete",
      "refinement_pass_complete",
      "claim_verdict",
      "claim_verdict",
      "document_verdict",
    ]);
  });

  it("reassembles frames split across chunk boundaries", async () => {
    const text = readFixture();
    // Split every 40 bytes to force multiple partial frames per chunk
    const chunks: string[] = [];
    for (let i = 0; i < text.length; i += 40) {
      chunks.push(text.slice(i, i + 40));
    }
    server.use(
      http.post(`${API_BASE}/verify/stream`, () =>
        chunkedResponse(chunks) as unknown as HttpResponse<string>,
      ),
    );

    const events: SseEvent["event"][] = [];
    await streamVerify(
      { text: "hello" },
      {
        onEvent: (e) => events.push(e.event),
        onError: () => {
          throw new Error("unexpected error");
        },
        onComplete: () => {},
      },
      new AbortController().signal,
    );
    expect(events.at(-1)).toBe("document_verdict");
    expect(events).toHaveLength(9);
  });

  it("emits OhiError via onError on non-2xx response", async () => {
    server.use(
      http.post(`${API_BASE}/verify/stream`, () =>
        HttpResponse.json(
          { status: "resting", reason: "origin-unreachable" },
          { status: 503, headers: { "Retry-After": "300" } },
        ),
      ),
    );
    const onError = vi.fn();
    await streamVerify(
      { text: "hello" },
      { onEvent: () => {}, onError, onComplete: () => {} },
      new AbortController().signal,
    );
    expect(onError).toHaveBeenCalled();
  });
});
