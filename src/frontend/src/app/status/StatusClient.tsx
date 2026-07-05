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
    <div className="bg-surface-base pt-[168px] md:pt-[204px]">
      <div className="sb-container max-w-5xl pb-28">
        <header className="mb-14">
          <p className="sb-kicker">Operations</p>
          <h1 className="mt-6">System status.</h1>
          <p className="mt-8 max-w-2xl text-[1.32rem] font-light leading-[1.625] text-brand-muted">
            OHI runs on volunteer infrastructure. When the PC hosting the data layer is offline, the
            API returns{" "}
            <code className="rounded bg-[color:var(--brand-secondary)] px-1.5 py-0.5 font-mono text-sm text-[color:var(--brand-accent)]">
              {"{\"status\":\"resting\"}"}
            </code>{" "}
            and this page reflects that directly. Polls every 30 s.
          </p>
        </header>

        <EndToEndHealth deepData={data} deepError={error} deepLoading={isLoading} onRefreshDeep={refetch} />

        {isLoading && (
          <div className="sb-panel h-48 animate-pulse" />
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
    </div>
  );
}
