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
        "rounded-xl border border-[color:var(--brand-warning)]/30 bg-[color:var(--brand-warning-soft)]/50 p-6",
        className,
      )}
      role="status"
      data-testid="budget-exhausted-state"
    >
      <div className="text-3xl">💰</div>
      <h2 className="font-heading mt-2 text-lg font-semibold text-brand-ink">
        Daily budget exhausted
      </h2>
      <p className="mt-1 text-sm font-medium text-brand-ink">
        OHI&apos;s free LLM budget for today has been used up.
      </p>
      <p className="mt-2 max-w-prose text-sm text-brand-muted">
        Verification resumes when the daily quota resets. Browsing existing verdicts and the
        calibration report still works.
      </p>
      {detailMessage && (
        <p className="num-mono mt-1 text-[11px] text-[color:var(--brand-warning)]">
          {detailMessage}
        </p>
      )}
      <div className="mt-4 flex flex-wrap items-center gap-3">
        <RetryCountdown retryAfterSec={retryAfterSec} primaryLabel="Retry now" />
        <Link
          href="/calibration"
          className="rounded-md border border-[color:var(--border-default)] bg-surface-elevated px-3 py-1.5 text-xs font-semibold text-brand-ink hover:bg-[color:var(--surface-soft)]"
        >
          Browse calibration report
        </Link>
      </div>
    </section>
  );
}
