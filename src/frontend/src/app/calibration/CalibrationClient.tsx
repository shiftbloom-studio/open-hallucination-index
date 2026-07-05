"use client";

import { CalibrationTable } from "@/components/calibration/CalibrationTable";
import { RestingState } from "@/components/common/RestingState";
import { NetworkErrorState } from "@/components/common/NetworkErrorState";
import { useCalibration } from "@/lib/ohi-queries";
import { OhiError } from "@/lib/ohi-client";

export function CalibrationClient() {
  const { data, error, refetch, isLoading } = useCalibration();

  return (
    <div className="bg-surface-base pt-[168px] md:pt-[204px]">
      <div className="sb-container max-w-5xl pb-28">
        <header className="mb-14">
          <p className="sb-kicker">Transparency</p>
          <h1 className="mt-6">Calibration report.</h1>
          <p className="mt-8 max-w-2xl text-[1.32rem] font-light leading-[1.625] text-brand-muted">
            Per-domain split conformal prediction targets 90% coverage. These are the empirical
            numbers from the most recent calibration set. Coverage within +/-2% of target is on-target;
            +/-2-5% amber; &gt;5% rose.
          </p>
        </header>

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

        {data && <CalibrationTable report={data} />}
      </div>
    </div>
  );
}
