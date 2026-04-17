"use client";

import { useHealthDeep } from "@/lib/ohi-queries";
import { OhiError } from "@/lib/ohi-client";
import { HealthMatrix } from "@/components/status/HealthMatrix";
import { RestingState } from "@/components/common/RestingState";
import { NetworkErrorState } from "@/components/common/NetworkErrorState";

export function StatusClient() {
  const { data, error, isLoading, refetch } = useHealthDeep();

  return (
    <div className="mx-auto w-full max-w-4xl px-4 py-10">
      <header className="mb-6">
        <p className="text-[10px] font-semibold uppercase tracking-[0.25em] text-indigo-300">
          Operations
        </p>
        <h1 className="mt-2 text-3xl font-bold text-slate-50">System status</h1>
        <p className="mt-2 max-w-2xl text-sm text-slate-400">
          OHI runs on volunteer infrastructure. When the PC hosting the data layer is offline, the
          API returns <code className="font-mono">{"{\"status\":\"resting\"}"}</code> and this page
          reflects that directly. Polls every 30 s.
        </p>
      </header>

      {isLoading && (
        <div className="h-48 animate-pulse rounded-xl border border-white/10 bg-white/[0.03]" />
      )}

      {error instanceof OhiError && error.isResting && (
        <RestingState retryAfterSec={error.retryAfterSec ?? 300} onRetry={() => refetch()} />
      )}

      {error && !(error instanceof OhiError && error.isResting) && (
        <NetworkErrorState
          detail={error instanceof Error ? error.message : String(error)}
          onRetrySync={() => refetch()}
        />
      )}

      {data && <HealthMatrix data={data} />}
    </div>
  );
}
