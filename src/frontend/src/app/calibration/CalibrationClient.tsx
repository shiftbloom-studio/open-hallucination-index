"use client";

import { CalibrationTable } from "@/components/calibration/CalibrationTable";
import { RestingState } from "@/components/common/RestingState";
import { NetworkErrorState } from "@/components/common/NetworkErrorState";
import { useCalibration } from "@/lib/ohi-queries";
import { OhiError } from "@/lib/ohi-client";

export function CalibrationClient() {
  const { data, error, refetch, isLoading } = useCalibration();

  return (
    <div className="mx-auto w-full max-w-5xl px-4 py-10">
      <header className="mb-6">
        <p className="text-[10px] font-semibold uppercase tracking-[0.25em] text-indigo-300">
          Transparency
        </p>
        <h1 className="mt-2 text-3xl font-bold text-slate-50">Calibration report</h1>
        <p className="mt-2 max-w-2xl text-sm text-slate-400">
          Per-domain split conformal prediction targets 90% coverage. These are the empirical
          numbers from the most recent calibration set. Coverage within ±2% of target is on-target;
          ±2–5% amber; &gt;5% rose.
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

      {data && <CalibrationTable report={data} />}
    </div>
  );
}
