import { describe, it, expect } from "vitest";
import { verifyReducer, initialState } from "../verify-controller";
import type { ClaimVerdict, DocumentVerdict, SseEvent } from "../ohi-types";
import golden from "../../test/fixtures/document-verdict.golden.json";

const doc = golden as unknown as DocumentVerdict;
const [claim1, claim2] = doc.claims as [ClaimVerdict, ClaimVerdict];

describe("verifyReducer", () => {
  it("START transitions to streaming and resets prior state", () => {
    const dirty = { ...initialState, claims: [claim1], error: new Error("stale") };
    const next = verifyReducer(dirty, { type: "START", at: 100 });
    expect(next.status).toBe("streaming");
    expect(next.claims).toEqual([]);
    expect(next.error).toBeNull();
    expect(next.startedAt).toBe(100);
  });

  it("decomposition_complete populates progress.decomposition", () => {
    const evt: SseEvent = {
      event: "decomposition_complete",
      data: { claim_count: 2, estimated_total_ms: 60000 },
    };
    const next = verifyReducer(
      { ...initialState, status: "streaming" },
      { type: "SSE_EVENT", event: evt },
    );
    expect(next.progress.decomposition).toEqual(evt.data);
  });

  it("claim_routed accumulates claim_ids and dedupes", () => {
    const routed1: SseEvent = {
      event: "claim_routed",
      data: { claim_id: claim1.claim.id, domain: "general", weights: {} },
    };
    const routed2: SseEvent = {
      event: "claim_routed",
      data: { claim_id: claim2.claim.id, domain: "general", weights: {} },
    };
    let state = verifyReducer(initialState, { type: "SSE_EVENT", event: routed1 });
    state = verifyReducer(state, { type: "SSE_EVENT", event: routed2 });
    // duplicate
    state = verifyReducer(state, { type: "SSE_EVENT", event: routed1 });
    expect(state.progress.routed).toEqual([claim1.claim.id, claim2.claim.id]);
  });

  it("claim_verdict appends and dedupes by claim.id", () => {
    const cv1: SseEvent = { event: "claim_verdict", data: claim1 };
    const cv2: SseEvent = { event: "claim_verdict", data: claim2 };
    let state = verifyReducer(initialState, { type: "SSE_EVENT", event: cv1 });
    expect(state.status).toBe("partial");
    expect(state.claims).toHaveLength(1);
    state = verifyReducer(state, { type: "SSE_EVENT", event: cv2 });
    expect(state.claims).toHaveLength(2);
    // replay
    state = verifyReducer(state, { type: "SSE_EVENT", event: cv1 });
    expect(state.claims).toHaveLength(2);
  });

  it("document_verdict moves to complete and preserves accumulated claims when event has empty claims", () => {
    const cv1: SseEvent = { event: "claim_verdict", data: claim1 };
    const dv: SseEvent = {
      event: "document_verdict",
      data: { ...doc, claims: [] }, // backend emits the full doc with claims[] emptied
    };
    let state = verifyReducer(initialState, { type: "SSE_EVENT", event: cv1 });
    state = verifyReducer(state, { type: "SSE_EVENT", event: dv });
    expect(state.status).toBe("complete");
    expect(state.verdict).toEqual(dv.data);
    expect(state.claims).toHaveLength(1);
  });

  it("document_verdict uses its own claims if non-empty", () => {
    const dv: SseEvent = { event: "document_verdict", data: doc };
    const state = verifyReducer(initialState, { type: "SSE_EVENT", event: dv });
    expect(state.claims).toHaveLength(2);
  });

  it("error event transitions to error status", () => {
    const evt: SseEvent = {
      event: "error",
      data: { status: "resting", reason: "origin-unreachable" },
    };
    const state = verifyReducer(initialState, { type: "SSE_EVENT", event: evt });
    expect(state.status).toBe("error");
    expect(state.error).not.toBeNull();
  });

  it("COMPLETE_SYNC copies claims from the DocumentVerdict", () => {
    const state = verifyReducer(initialState, { type: "COMPLETE_SYNC", verdict: doc });
    expect(state.status).toBe("complete");
    expect(state.claims).toHaveLength(2);
  });

  it("RESET returns initialState", () => {
    const state = verifyReducer(
      { ...initialState, status: "complete", claims: [claim1] },
      { type: "RESET" },
    );
    expect(state).toEqual(initialState);
  });
});
