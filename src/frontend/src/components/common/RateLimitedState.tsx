import { cn } from "@/lib/utils";
import { RetryCountdown } from "./RetryCountdown";

export interface RateLimitedStateProps {
  retryAfterSec?: number;
  onRetry?: () => void;
  className?: string;
}

export function RateLimitedState({ retryAfterSec, onRetry, className }: RateLimitedStateProps) {
  return (
    <section
      className={cn(
        "rounded-xl border border-indigo-400/25 bg-gradient-to-b from-indigo-500/[0.05] to-transparent p-6",
        className,
      )}
      role="status"
      data-testid="rate-limited-state"
    >
      <div className="text-3xl">⏳</div>
      <h2 className="mt-2 text-lg font-semibold text-slate-100">Rate limit reached</h2>
      <p className="mt-1 text-sm font-semibold text-slate-200">
        You&apos;ve hit the per-IP rate limit for <code className="font-mono">/verify</code>.
      </p>
      <p className="mt-2 max-w-prose text-sm text-slate-400">
        Rate limits keep OHI sustainable for everyone. No action needed — try again in a moment.
      </p>
      <div className="mt-4">
        <RetryCountdown retryAfterSec={retryAfterSec ?? 60} onRetry={onRetry} />
      </div>
    </section>
  );
}
