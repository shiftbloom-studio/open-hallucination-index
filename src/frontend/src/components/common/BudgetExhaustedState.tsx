import Link from "next/link";
import { cn } from "@/lib/utils";
import { RetryCountdown } from "./RetryCountdown";

export interface BudgetExhaustedStateProps {
  retryAfterSec?: number;
  detailMessage?: string;
  className?: string;
}

export function BudgetExhaustedState({
  retryAfterSec,
  detailMessage,
  className,
}: BudgetExhaustedStateProps) {
  return (
    <section
      className={cn(
        "rounded-xl border border-orange-400/25 bg-gradient-to-b from-orange-500/[0.05] to-transparent p-6",
        className,
      )}
      role="status"
      data-testid="budget-exhausted-state"
    >
      <div className="text-3xl">💰</div>
      <h2 className="mt-2 text-lg font-semibold text-slate-100">Daily budget exhausted</h2>
      <p className="mt-1 text-sm font-semibold text-slate-200">
        OHI&apos;s free LLM budget for today has been used up.
      </p>
      <p className="mt-2 max-w-prose text-sm text-slate-400">
        Verification resumes when the daily quota resets. Browsing existing verdicts and the
        calibration report still works.
      </p>
      {detailMessage && (
        <p className="mt-1 font-mono text-[11px] text-orange-300">{detailMessage}</p>
      )}
      <div className="mt-4 flex flex-wrap items-center gap-3">
        <RetryCountdown retryAfterSec={retryAfterSec} primaryLabel="Retry now" />
        <Link
          href="/calibration"
          className="rounded-md bg-white/[0.04] px-3 py-1.5 text-xs font-semibold text-slate-200 ring-1 ring-inset ring-white/10 hover:bg-white/[0.08]"
        >
          Browse calibration report
        </Link>
      </div>
    </section>
  );
}
