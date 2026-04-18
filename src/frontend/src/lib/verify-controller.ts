import { useCallback, useReducer, useRef } from "react";
import { OhiError, ohi } from "./ohi-client";
import type { JobStatus } from "./ohi-client";
import type {
  ClaimVerdict,
  Domain,
  DocumentVerdict,
  VerifyRequest,
} from "./ohi-types";

/**
 * Stream D2: SSE was blocked on a free-tier Cloudflare + API Gateway
 * host-header quirk and a 30 s hard cap. The controller is now a
 * polling loop against ``GET /api/v2/verify/status/{job_id}``. The
 * public state shape is preserved so ``VerifyPage`` and ``SseProgress``
 * (naming-legacy-only) continue to work — status values that SSE
 * populated are now driven from phase transitions in the polled
 * record.
 *
 * Polling mechanics, calibrated with E1's observed latency (~5 s for a
 * 2-claim doc) and the 180 s Lambda timeout:
 * - 1 s interval by default (phases typically change every 2–10 s).
 * - Exponential backoff to 5 s on transient network errors.
 * - Hard 3 min absolute cap — well past the 180 s Lambda timeout so
 *   hung Lambdas surface as a terminal error in the UI rather than a
 *   perpetual spinner.
 */

export type VerifyStatus =
  | "idle"
  | "streaming"
  | "partial"
  | "complete"
  | "sync_fallback"
  | "error";

export interface ProgressBag {
  /**
   * The canonical progress signal under D2: the current phase string
   * from the polled record (one of the five pipeline boundary phases,
   * plus ``queued`` before the async handler picks up). SseProgress
   * does not yet consume this field; it is here for future UI work.
   */
  currentPhase?: string;
  /** Every phase seen so far, in arrival order. Useful for debug. */
  phasesSeen?: string[];

  // ── Legacy ProgressBag fields (SSE era) ──────────────────────────
  // SseProgress reads these to render the step list. They are now
  // populated *opportunistically* on phase transitions: we stamp each
  // legacy field the moment its logical phase finishes, so the UI
  // still ticks through steps even though we no longer receive the
  // per-event telemetry SSE provided. Exact counts (claim_count, nli
  // pair count, pcg iterations) are placeholder zeros until the final
  // document_verdict arrives, at which point we backfill from the
  // verdict itself.
  decomposition?: { claim_count: number; estimated_total_ms: number };
  routed: string[];
  nli?: { claim_evidence_pairs_scored: number; claim_pair_pairs_scored: number };
  pcg?: {
    iterations: number;
    converged: boolean;
    algorithm: string;
    internal_consistency: number;
    gibbs_validated: boolean | null;
  };
  refinement?: { pass: number; claims_re_retrieved: number; marginal_max_change: number };
}

export interface VerifyState {
  status: VerifyStatus;
  progress: ProgressBag;
  claims: ClaimVerdict[];
  verdict: DocumentVerdict | null;
  error: OhiError | Error | null;
  startedAt: number | null;
  jobId: string | null;
}

export const initialState: VerifyState = {
  status: "idle",
  progress: { routed: [] },
  claims: [],
  verdict: null,
  error: null,
  startedAt: null,
  jobId: null,
};

export type VerifyAction =
  | { type: "START"; at: number }
  | { type: "JOB_ACCEPTED"; jobId: string }
  | { type: "POLL_UPDATE"; status: JobStatus }
  | { type: "COMPLETE"; verdict: DocumentVerdict }
  | { type: "ERROR"; error: OhiError | Error }
  | { type: "RESET" };

function applyPhaseToProgress(progress: ProgressBag, phase: string): ProgressBag {
  const phasesSeen = [...(progress.phasesSeen ?? [])];
  if (phasesSeen[phasesSeen.length - 1] !== phase) {
    phasesSeen.push(phase);
  }
  const next: ProgressBag = { ...progress, currentPhase: phase, phasesSeen };

  // Opportunistic population of legacy SseProgress fields. Each phase
  // marks *its predecessor* as done — so "retrieving_evidence" arrives
  // the moment the pipeline finished decomposing, and so on. Counts
  // are placeholders until the verdict arrives; the goal is just to
  // drive the step-state machine in SseProgress.
  if (
    phase === "retrieving_evidence" ||
    phase === "classifying" ||
    phase === "calibrating" ||
    phase === "assembling"
  ) {
    if (!next.decomposition) {
      next.decomposition = { claim_count: 0, estimated_total_ms: 0 };
    }
  }
  if (phase === "classifying" || phase === "calibrating" || phase === "assembling") {
    // No real claim-routing phase exists in pipeline.py — domain routing
    // is a trivial dict lookup so it never gets its own boundary. We
    // fake a single-entry routed[] so SseProgress shows the routing
    // step as active, then done, at the right moment.
    if (next.routed.length === 0) next.routed = ["__placeholder__"];
  }
  if (phase === "calibrating" || phase === "assembling") {
    if (!next.nli) {
      next.nli = { claim_evidence_pairs_scored: 0, claim_pair_pairs_scored: 0 };
    }
  }
  if (phase === "assembling") {
    if (!next.pcg) {
      next.pcg = {
        iterations: 0,
        converged: true,
        algorithm: "beta-posterior-from-nli",
        internal_consistency: 1.0,
        gibbs_validated: null,
      };
    }
    if (!next.refinement) {
      next.refinement = { pass: 0, claims_re_retrieved: 0, marginal_max_change: 0 };
    }
  }
  return next;
}

export function verifyReducer(state: VerifyState, action: VerifyAction): VerifyState {
  switch (action.type) {
    case "RESET":
      return initialState;

    case "START":
      return {
        ...initialState,
        status: "streaming",
        startedAt: action.at,
      };

    case "JOB_ACCEPTED":
      return { ...state, jobId: action.jobId };

    case "POLL_UPDATE": {
      const js = action.status;
      if (js.status === "error") {
        const asErr = new OhiError(503, { detail: js.error ?? "async pipeline error" });
        return { ...state, status: "error", error: asErr };
      }
      return {
        ...state,
        status: js.status === "done" ? "complete" : "streaming",
        progress: applyPhaseToProgress(state.progress, js.phase),
      };
    }

    case "COMPLETE": {
      const v = action.verdict;
      // Backfill progress.decomposition.claim_count + routed[] from the
      // real verdict so the step telemetry shows accurate counts at
      // completion time.
      const progress: ProgressBag = {
        ...state.progress,
        currentPhase: "assembling",
        decomposition: { claim_count: v.claims.length, estimated_total_ms: v.processing_time_ms },
        routed: v.claims.map((c) => c.claim.id),
      };
      return {
        ...state,
        status: "complete",
        verdict: v,
        claims: v.claims,
        progress,
      };
    }

    case "ERROR":
      return { ...state, status: "error", error: action.error };
  }
}

export interface UseVerifyControllerApi {
  state: VerifyState;
  submit: (req: VerifyRequest) => Promise<void>;
  /**
   * Alias for ``submit`` — kept for back-compat with VerifyPage's
   * "Retry with sync fallback" button. Pre-D2 this bypassed SSE and
   * hit /verify directly; under D2 there is no SSE path at all, so
   * both entry points drive the same polling flow.
   */
  submitSync: (req: VerifyRequest) => Promise<void>;
  cancel: () => void;
  reset: () => void;
}

const POLL_INTERVAL_MS = 1000;
const POLL_MAX_INTERVAL_MS = 5000;
const POLL_HARD_CAP_MS = 3 * 60 * 1000;

async function _sleep(ms: number, signal: AbortSignal): Promise<void> {
  if (signal.aborted) return;
  await new Promise<void>((resolve) => {
    const onAbort = () => {
      clearTimeout(handle);
      resolve();
    };
    const handle = setTimeout(() => {
      signal.removeEventListener("abort", onAbort);
      resolve();
    }, ms);
    signal.addEventListener("abort", onAbort, { once: true });
  });
}

export function useVerifyController(options?: {
  domains?: Domain[]; // reserved for future hints
  firstByteTimeoutMs?: number; // kept for API compat; no longer used
}): UseVerifyControllerApi {
  // options.firstByteTimeoutMs is intentionally unused under D2 polling.
  // Kept in the props signature so VerifyPage and existing callers that
  // pass it do not need to change when they upgrade.
  void options?.firstByteTimeoutMs;

  const [state, dispatch] = useReducer(verifyReducer, initialState);
  const abortRef = useRef<AbortController | null>(null);

  const cancel = useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
  }, []);

  const reset = useCallback(() => {
    cancel();
    dispatch({ type: "RESET" });
  }, [cancel]);

  const runFlow = useCallback(async (req: VerifyRequest) => {
    cancel();
    const controller = new AbortController();
    abortRef.current = controller;
    const signal = controller.signal;
    const startedAt = Date.now();

    dispatch({ type: "START", at: startedAt });

    let jobId: string;
    try {
      const accepted = await ohi.verify(req, { signal });
      jobId = accepted.job_id;
      dispatch({ type: "JOB_ACCEPTED", jobId });
    } catch (err) {
      if (signal.aborted) return;
      dispatch({
        type: "ERROR",
        error: err instanceof Error ? err : new Error(String(err)),
      });
      return;
    }

    let currentInterval = POLL_INTERVAL_MS;

    while (!signal.aborted) {
      if (Date.now() - startedAt > POLL_HARD_CAP_MS) {
        dispatch({
          type: "ERROR",
          error: new Error(
            "Verification timed out after 3 minutes. The backend may still be " +
              "processing; check the poll endpoint directly if you need the result.",
          ),
        });
        return;
      }

      let js: JobStatus;
      try {
        js = await ohi.verifyStatus(jobId, { signal });
        // Success resets the backoff clock.
        currentInterval = POLL_INTERVAL_MS;
      } catch (err) {
        if (signal.aborted) return;
        if (err instanceof OhiError && err.status === 404) {
          // Job disappeared — either TTL reaped it (shouldn't happen
          // within 3 min) or a producer bug. Surface.
          dispatch({ type: "ERROR", error: err });
          return;
        }
        // Transient: exponential backoff capped at 5 s.
        currentInterval = Math.min(currentInterval * 2, POLL_MAX_INTERVAL_MS);
        await _sleep(currentInterval, signal);
        continue;
      }

      dispatch({ type: "POLL_UPDATE", status: js });

      if (js.status === "done") {
        if (js.result) {
          dispatch({ type: "COMPLETE", verdict: js.result });
        } else {
          dispatch({
            type: "ERROR",
            error: new Error("Backend reported done but returned no verdict payload."),
          });
        }
        return;
      }
      if (js.status === "error") {
        // POLL_UPDATE already dispatched the error; stop the loop.
        return;
      }

      await _sleep(currentInterval, signal);
    }
  }, [cancel]);

  const submit = useCallback(
    async (req: VerifyRequest) => runFlow(req),
    [runFlow],
  );

  // Retry button still calls submitSync; under polling it is just submit.
  const submitSync = submit;

  return { state, submit, submitSync, cancel, reset };
}
