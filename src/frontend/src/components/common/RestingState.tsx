import Link from "next/link";
import { cn } from "@/lib/utils";
import { RetryCountdown } from "./RetryCountdown";

export interface RestingStateProps {
  retryAfterSec?: number;
  onRetry?: () => void;
  className?: string;
}

export function RestingState({ retryAfterSec = 300, onRetry, className }: RestingStateProps) {
  return (
    <section
      className={cn(
        "rounded-xl border border-[color:var(--brand-warning)]/30 bg-[color:var(--brand-warning-soft)]/40 p-6",
        className,
      )}
      role="status"
      data-testid="resting-state"
    >
      <div className="text-3xl">🌙</div>
      <h2 className="font-heading mt-2 text-lg font-semibold text-brand-ink">
        Service is resting
      </h2>
      <p className="mt-1 text-sm font-medium text-brand-ink">
        The PC that hosts OHI&apos;s data is currently offline.
      </p>
      <p className="mt-2 max-w-prose text-sm text-brand-muted">
        OHI runs on volunteer infrastructure. The service typically wakes back up within a few
        hours. Your request wasn&apos;t saved.
      </p>
      <div className="mt-4 flex flex-wrap items-center gap-3">
        <RetryCountdown retryAfterSec={retryAfterSec} onRetry={onRetry} />
        <Link
          href="/status"
          className="text-xs text-brand-muted underline-offset-2 hover:text-brand-ink hover:underline"
        >
          See /status
        </Link>
      </div>
    </section>
  );
}
