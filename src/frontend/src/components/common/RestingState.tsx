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
        "rounded-xl border border-amber-400/20 bg-gradient-to-b from-amber-500/[0.05] to-transparent p-6",
        className,
      )}
      role="status"
      data-testid="resting-state"
    >
      <div className="text-3xl">🌙</div>
      <h2 className="mt-2 text-lg font-semibold text-slate-100">Service is resting</h2>
      <p className="mt-1 text-sm font-semibold text-slate-200">
        The PC that hosts OHI&apos;s data is currently offline.
      </p>
      <p className="mt-2 max-w-prose text-sm text-slate-400">
        OHI runs on volunteer infrastructure. The service typically wakes back up within a few
        hours. Your request wasn&apos;t saved.
      </p>
      <div className="mt-4 flex flex-wrap items-center gap-3">
        <RetryCountdown retryAfterSec={retryAfterSec} onRetry={onRetry} />
        <Link href="/status" className="text-xs text-slate-500 underline-offset-2 hover:text-slate-300 hover:underline">
          See /status
        </Link>
      </div>
    </section>
  );
}
