import { describe, it, expect } from "vitest";
import {
  initialState,
  verifyReducer,
  type VerifyState,
} from "../verify-controller";
import type { DocumentVerdict, ClaimVerdict } from "../ohi-types";
import type { JobStatus } from "../ohi-client";

/**
 * Tests for the Stream D2 polling reducer — SSE events replaced with
 * JobStatus updates from ``GET /api/v2/verify/status/{id}``. These pin
 * the two pieces the UI actually reads:
 *
 * 1. Phase transitions advance the legacy ProgressBag step fields so
 *    SseProgress continues to render progress without any component
 *    changes. Frontend overhaul stream can drop the legacy fields
 *    later.
 * 2. The terminal ``status==='done'`` path lands ``state.verdict`` +
 *    ``state.claims`` in one COMPLETE dispatch.
 * 3. The terminal ``status==='error'`` path routes to state.error with
 *    an OhiError 503 wrapper.
 */


function fakeEvidence(id: string) {
  return {
    id,
    source_uri: null,
    content: `evidence ${id}`,
    retrieved_at: "2026-01-01T00:00:00.000Z",
  };
}


function fakeClaim(id: string): ClaimVerdict {
  return {
    claim: { id, text: `claim ${id}` },
    p_true: 0.5,
    interval: [0, 1],
    coverage_target: null,
    domain: "general",
    domain_assignment_weights: { general: 1 },
    supporting_evidence: [fakeEvidence(`sup-${id}`)],
    refuting_evidence: [fakeEvidence(`ref-${id}`)],
    pcg_neighbors: [],
    nli_self_consistency_variance: 0,
    bp_validated: null,
    information_gain: 0,
    queued_for_review: false,
    calibration_set_id: null,
    calibration_n: 0,
    fallback_used: null,
  };
}


function fakeVerdict(): DocumentVerdict {
  return {
    request_id: "req-1",
    pipeline_version: "ohi-v2.0",
    model_versions: { decomposer: "stub" },
    document_score: 0.7,
    document_interval: [0.2, 0.9],
    internal_consistency: 1,
    decomposition_coverage: 1,
    processing_time_ms: 123,
    rigor: "balanced",
    refinement_passes_executed: 0,
    claims: [fakeClaim("c-1"), fakeClaim("c-2")],
  };
}


function pollStatus(phase: string, status: "pending" | "done" | "error" = "pending"): JobStatus {
  return {
    job_id: "job-1",
    status,
    phase,
    created_at: 100,
    updated_at: 150,
  };
}


describe("verifyReducer", () => {
  it("RESET returns to initial state", () => {
    const started: VerifyState = { ...initialState, status: "streaming" };
    const next = verifyReducer(started, { type: "RESET" });
    expect(next).toEqual(initialState);
  });

  it("START stamps startedAt and flips to streaming", () => {
    const next = verifyReducer(initialState, { type: "START", at: 42 });
    expect(next.status).toBe("streaming");
    expect(next.startedAt).toBe(42);
  });

  it("JOB_ACCEPTED binds jobId without changing other state", () => {
    const started = verifyReducer(initialState, { type: "START", at: 10 });
    const next = verifyReducer(started, { type: "JOB_ACCEPTED", jobId: "job-7" });
    expect(next.jobId).toBe("job-7");
    expect(next.status).toBe("streaming");
  });

  it("POLL_UPDATE on 'decomposing' leaves all legacy progress fields unset", () => {
    const started = verifyReducer(initialState, { type: "START", at: 0 });
    const next = verifyReducer(started, {
      type: "POLL_UPDATE",
      status: pollStatus("decomposing"),
    });
    expect(next.progress.currentPhase).toBe("decomposing");
    expect(next.progress.decomposition).toBeUndefined();
    expect(next.progress.nli).toBeUndefined();
  });

  it("POLL_UPDATE 'retrieving_evidence' marks decomposition done", () => {
    let state = verifyReducer(initialState, { type: "START", at: 0 });
    state = verifyReducer(state, {
      type: "POLL_UPDATE",
      status: pollStatus("retrieving_evidence"),
    });
    expect(state.progress.decomposition).toEqual({
      claim_count: 0,
      estimated_total_ms: 0,
    });
  });

  it("POLL_UPDATE 'classifying' stamps routed[] so the routing step shows progress", () => {
    let state = verifyReducer(initialState, { type: "START", at: 0 });
    state = verifyReducer(state, {
      type: "POLL_UPDATE",
      status: pollStatus("classifying"),
    });
    expect(state.progress.routed.length).toBeGreaterThan(0);
  });

  it("POLL_UPDATE 'assembling' stamps pcg + refinement so those steps show done", () => {
    let state = verifyReducer(initialState, { type: "START", at: 0 });
    state = verifyReducer(state, {
      type: "POLL_UPDATE",
      status: pollStatus("assembling"),
    });
    expect(state.progress.pcg).toBeDefined();
    expect(state.progress.refinement).toBeDefined();
  });

  it("phasesSeen tracks arrival order without duplicates", () => {
    let state = verifyReducer(initialState, { type: "START", at: 0 });
    for (const phase of ["decomposing", "retrieving_evidence", "classifying", "classifying"]) {
      state = verifyReducer(state, { type: "POLL_UPDATE", status: pollStatus(phase) });
    }
    expect(state.progress.phasesSeen).toEqual([
      "decomposing",
      "retrieving_evidence",
      "classifying",
    ]);
  });

  it("POLL_UPDATE with status=error flips to error state with a 503 OhiError", () => {
    let state = verifyReducer(initialState, { type: "START", at: 0 });
    const js: JobStatus = {
      ...pollStatus("classifying"),
      status: "error",
      error: "NliAdapter: timeout",
    };
    state = verifyReducer(state, { type: "POLL_UPDATE", status: js });
    expect(state.status).toBe("error");
    expect(state.error).toBeTruthy();
    expect((state.error as Error).message).toContain("NliAdapter: timeout");
  });

  it("COMPLETE lands verdict + claims + backfills progress.decomposition.claim_count", () => {
    let state = verifyReducer(initialState, { type: "START", at: 0 });
    state = verifyReducer(state, {
      type: "POLL_UPDATE",
      status: pollStatus("retrieving_evidence"),
    });
    const verdict = fakeVerdict();
    state = verifyReducer(state, { type: "COMPLETE", verdict });
    expect(state.status).toBe("complete");
    expect(state.verdict).toBe(verdict);
    expect(state.claims).toEqual(verdict.claims);
    expect(state.progress.decomposition?.claim_count).toBe(verdict.claims.length);
    expect(state.progress.routed).toEqual(verdict.claims.map((c) => c.claim.id));
    expect(state.progress.nli).toEqual({
      claim_evidence_pairs_scored: 4,
      claim_pair_pairs_scored: 0,
    });
  });
});
