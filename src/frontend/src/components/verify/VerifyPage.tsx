"use client";

import { useRef, useState } from "react";
import type { ClaimVerdict, VerifyRequest } from "@/lib/ohi-types";
import { useVerifyController } from "@/lib/verify-controller";
import { OhiError } from "@/lib/ohi-client";
import { VerifyForm } from "./VerifyForm";
import { SseProgress } from "./SseProgress";
import { DocumentVerdictCard } from "./DocumentVerdictCard";
import { ClaimList } from "./ClaimList";
import { PcgGraph, type PcgGraphHandle } from "./PcgGraph";
import { FeedbackSheet } from "./FeedbackSheet";
import { RestingState } from "@/components/common/RestingState";
import { BudgetExhaustedState } from "@/components/common/BudgetExhaustedState";
import { LlmUnavailableState } from "@/components/common/LlmUnavailableState";
import { RateLimitedState } from "@/components/common/RateLimitedState";
import { NetworkErrorState } from "@/components/common/NetworkErrorState";
import { DegradedState } from "@/components/common/DegradedState";

function ErrorPanel({
  error,
  onRetry,
  onRetrySync,
}: {
  error: OhiError | Error;
  onRetry?: () => void;
  onRetrySync?: () => void;
}) {
  if (error instanceof OhiError) {
    if (error.isResting) {
      return <RestingState retryAfterSec={error.retryAfterSec ?? 300} onRetry={onRetry} />;
    }
    if (error.isBudgetExhausted) {
      const detail = (error.body as { detail?: string }).detail;
      return <BudgetExhaustedState retryAfterSec={error.retryAfterSec} detailMessage={detail} />;
    }
    if (error.isLlmDown) {
      return <LlmUnavailableState onRetry={onRetry} />;
    }
    if (error.isRateLimited) {
      return <RateLimitedState retryAfterSec={error.retryAfterSec} onRetry={onRetry} />;
    }
  }
  return <NetworkErrorState onRetrySync={onRetrySync} detail={error.message} />;
}

export function VerifyPage() {
  const { state, submit, submitSync, cancel, reset } = useVerifyController();
  const [lastRequest, setLastRequest] = useState<VerifyRequest | null>(null);
  const [flagged, setFlagged] = useState<ClaimVerdict | null>(null);
  const graphRef = useRef<PcgGraphHandle>(null);

  const streaming = state.status === "streaming" || state.status === "partial";

  async function onSubmit(req: VerifyRequest) {
    setLastRequest(req);
    await submit(req);
  }

  function onRetry() {
    if (!lastRequest) return;
    reset();
    void submit(lastRequest);
  }

  async function onRetrySync() {
    if (!lastRequest) return;
    await submitSync(lastRequest);
  }

  const degradedLayers =
    state.error instanceof OhiError && Array.isArray((state.error.body as { degraded_layers?: string[] }).degraded_layers)
      ? ((state.error.body as { degraded_layers?: string[] }).degraded_layers as string[])
      : [];
  const fallbackCount = state.claims.filter((c) => c.fallback_used !== null).length;

  return (
    <div className="mx-auto grid w-full max-w-7xl grid-cols-1 gap-6 px-4 py-8 lg:grid-cols-[minmax(0,32rem)_minmax(0,1fr)]">
      {/* LEFT — form + progress */}
      <div className="space-y-4 lg:sticky lg:top-24 lg:self-start">
        <VerifyForm onSubmit={onSubmit} onCancel={cancel} streaming={streaming} />
        <SseProgress
          status={state.status}
          progress={state.progress}
          claimsRenderedCount={state.claims.length}
        />
      </div>

      {/* RIGHT — results */}
      <div className="space-y-4">
        {state.status === "error" && state.error && (
          <ErrorPanel error={state.error} onRetry={onRetry} onRetrySync={onRetrySync} />
        )}

        {state.status === "sync_fallback" && (
          <NetworkErrorState
            detail="Switching to synchronous endpoint — SSE didn't deliver the first byte in time."
            onRetrySync={onRetrySync}
          />
        )}

        {state.status !== "idle" && state.status !== "error" && (
          <>
            <DocumentVerdictCard verdict={state.verdict} />
            <DegradedState degradedLayers={degradedLayers} fallbackCount={fallbackCount} />
            <ClaimList
              claims={state.claims}
              onShowInGraph={(id) => graphRef.current?.focusNode(id)}
              onFlag={setFlagged}
              emptyMessage={streaming ? "Claims will appear as they're verified…" : "No claims yet."}
            />
            {state.claims.length > 0 && (
              <PcgGraph ref={graphRef} claims={state.claims} height={360} />
            )}
          </>
        )}

        {state.status === "idle" && (
          <div className="rounded-xl border border-dashed border-white/10 p-12 text-center text-sm text-slate-500">
            Paste text in the form to start. Nothing is stored; only a hash of your input.
          </div>
        )}
      </div>

      {flagged && state.verdict && (
        <FeedbackSheet
          requestId={state.verdict.request_id}
          claim={flagged}
          open={flagged !== null}
          onClose={() => setFlagged(null)}
        />
      )}
    </div>
  );
}
