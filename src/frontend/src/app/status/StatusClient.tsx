"use client";

import { useHealthDeep } from "@/lib/ohi-queries";
import { OhiError } from "@/lib/ohi-client";
import { HealthMatrix } from "@/components/status/HealthMatrix";
import { EndToEndHealth } from "@/components/status/EndToEndHealth";
import { RestingState } from "@/components/common/RestingState";
import { NetworkErrorState } from "@/components/common/NetworkErrorState";

export function StatusClient() {
  const { data, error, isLoading, refetch } = useHealthDeep();

  return (
    <div className="mx-auto w-full max-w-4xl px-4 py-10">
      <header className="mb-6">
        <p className="label-mono text-[color:var(--brand-indigo-strong)]">
          Operations
        </p>
        <h1 className="mt-2 text-3xl font-bold text-[color:var(--brand-ink)]">System status</h1>
        <p className="mt-2 max-w-2xl text-sm text-[color:var(--brand-muted)]">
          OHI runs on volunteer infrastructure. When the PC hosting the data layer is offline, the
          API returns{" "}
          <code className="rounded bg-[color:var(--brand-indigo-soft)] px-1.5 py-0.5 font-mono text-[color:var(--brand-indigo-strong)]">
            {"{\"status\":\"resting\"}"}
          </code>{" "}
          and this page reflects that directly. Polls every 30 s.
        </p>
      </header>

      <EndToEndHealth deepData={data} deepError={error} deepLoading={isLoading} onRefreshDeep={refetch} />

      {isLoading && (
        <div className="h-48 animate-pulse rounded-xl border border-[color:var(--border-subtle)] bg-[color:var(--surface-elevated)]/70 shadow-sm" />
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
