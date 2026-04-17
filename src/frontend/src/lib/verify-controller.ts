import { useCallback, useReducer, useRef } from "react";
import { OhiError, ohi } from "./ohi-client";
import { streamVerify } from "./sse";
import type {
  ClaimVerdict,
  Domain,
  DocumentVerdict,
  SseEvent,
  VerifyRequest,
} from "./ohi-types";

export type VerifyStatus =
  | "idle"
  | "streaming"
  | "partial"
  | "complete"
  | "sync_fallback"
  | "error";

export interface ProgressBag {
  decomposition?: { claim_count: number; estimated_total_ms: number };
  routed: string[]; // claim_ids, use array for stable render (JSON.stringify-safe)
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
}

export const initialState: VerifyState = {
  status: "idle",
  progress: { routed: [] },
  claims: [],
  verdict: null,
  error: null,
  startedAt: null,
};

export type VerifyAction =
  | { type: "START"; at: number }
  | { type: "SSE_EVENT"; event: SseEvent }
  | { type: "COMPLETE_SYNC"; verdict: DocumentVerdict }
  | { type: "SWITCH_SYNC_FALLBACK" }
  | { type: "ERROR"; error: OhiError | Error }
  | { type: "RESET" };

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

    case "SWITCH_SYNC_FALLBACK":
      return { ...state, status: "sync_fallback" };

    case "ERROR":
      return { ...state, status: "error", error: action.error };

    case "COMPLETE_SYNC":
      return {
        ...state,
        status: "complete",
        verdict: action.verdict,
        claims: action.verdict.claims,
      };

    case "SSE_EVENT": {
      const { event } = action;
      switch (event.event) {
        case "decomposition_complete":
          return {
            ...state,
            status: "streaming",
            progress: { ...state.progress, decomposition: event.data },
          };

        case "claim_routed": {
          if (state.progress.routed.includes(event.data.claim_id)) return state;
          return {
            ...state,
            progress: {
              ...state.progress,
              routed: [...state.progress.routed, event.data.claim_id],
            },
          };
        }

        case "nli_complete":
          return { ...state, progress: { ...state.progress, nli: event.data } };

        case "pcg_propagation_complete":
          return { ...state, progress: { ...state.progress, pcg: event.data } };

        case "refinement_pass_complete":
          return {
            ...state,
            progress: { ...state.progress, refinement: event.data },
          };

        case "claim_verdict": {
          // De-dupe by claim.id for reconnect safety
          if (state.claims.some((c) => c.claim.id === event.data.claim.id)) {
            return state;
          }
          return {
            ...state,
            status: "partial",
            claims: [...state.claims, event.data],
          };
        }

        case "document_verdict":
          return {
            ...state,
            status: "complete",
            verdict: event.data,
            // Backend's final document_verdict event has an empty claims array
            // (spec §10 example); keep the accumulated per-claim list.
            claims: event.data.claims.length > 0 ? event.data.claims : state.claims,
          };

        case "error": {
          const body = event.data;
          const asErr = new OhiError(503, body as never);
          return { ...state, status: "error", error: asErr };
        }
      }
      return state;
    }
  }
}

export interface UseVerifyControllerApi {
  state: VerifyState;
  submit: (req: VerifyRequest) => Promise<void>;
  /**
   * Skip SSE entirely and go straight to the synchronous /verify endpoint.
   * Used by the "Retry with sync fallback" button in the error panel when
   * the stream endpoint is unavailable (e.g. 404 during Phase 1 when
   * /verify/stream isn't implemented yet).
   */
  submitSync: (req: VerifyRequest) => Promise<void>;
  cancel: () => void;
  reset: () => void;
}

/**
 * React hook owning the AbortController + reducer for a /verify session.
 * Implementation note: SYNC fallback (post /verify blocking) triggers when
 * no SSE bytes arrive within 8s (corporate proxy / broken streaming).
 */
export function useVerifyController(options?: {
  domains?: Domain[]; // reserved for future hints
  firstByteTimeoutMs?: number;
}): UseVerifyControllerApi {
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

  const submit = useCallback(
    async (req: VerifyRequest) => {
      cancel();
      const controller = new AbortController();
      abortRef.current = controller;

      dispatch({ type: "START", at: Date.now() });

      let sawAnyByte = false;
      const timeoutMs = options?.firstByteTimeoutMs ?? 8000;
      const timeoutId = setTimeout(() => {
        if (!sawAnyByte) {
          dispatch({ type: "SWITCH_SYNC_FALLBACK" });
          // Fire sync fallback concurrently; the stream will still be aborted below.
          controller.abort();
          ohi
            .verify(req, { signal: new AbortController().signal })
            .then((v) => dispatch({ type: "COMPLETE_SYNC", verdict: v }))
            .catch((err) =>
              dispatch({ type: "ERROR", error: err instanceof Error ? err : new Error(String(err)) }),
            );
        }
      }, timeoutMs);

      await streamVerify(
        req,
        {
          onEvent: (evt) => {
            sawAnyByte = true;
            dispatch({ type: "SSE_EVENT", event: evt });
          },
          onError: (err) => {
            clearTimeout(timeoutId);
            if (controller.signal.aborted) return; // swallowed by cancel or fallback
            dispatch({
              type: "ERROR",
              error: err instanceof Error ? err : new Error(String(err)),
            });
          },
          onComplete: () => {
            clearTimeout(timeoutId);
          },
        },
        controller.signal,
      );
    },
    [cancel, options?.firstByteTimeoutMs],
  );

  const submitSync = useCallback(
    async (req: VerifyRequest) => {
      cancel();
      const controller = new AbortController();
      abortRef.current = controller;

      dispatch({ type: "START", at: Date.now() });
      dispatch({ type: "SWITCH_SYNC_FALLBACK" });

      try {
        const verdict = await ohi.verify(req, { signal: controller.signal });
        dispatch({ type: "COMPLETE_SYNC", verdict });
      } catch (err) {
        if (controller.signal.aborted) return;
        dispatch({
          type: "ERROR",
          error: err instanceof Error ? err : new Error(String(err)),
        });
      }
    },
    [cancel],
  );

  return { state, submit, submitSync, cancel, reset };
}
