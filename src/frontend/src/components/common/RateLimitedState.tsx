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
        "rounded-xl border border-[color:var(--brand-indigo)]/30 bg-[color:var(--brand-indigo-soft)]/70 p-6",
        className,
      )}
      role="status"
      data-testid="rate-limited-state"
    >
      <div className="text-3xl">⏳</div>
      <h2 className="font-heading mt-2 text-lg font-semibold text-brand-ink">
        Rate limit reached
      </h2>
      <p className="mt-1 text-sm font-medium text-brand-ink">
        You&apos;ve hit the per-IP rate limit for{" "}
        <code className="font-mono text-[color:var(--brand-indigo-strong)]">/verify</code>.
      </p>
      <p className="mt-2 max-w-prose text-sm text-brand-muted">
        Rate limits keep OHI sustainable for everyone. No action needed — try again in a moment.
      </p>
      <div className="mt-4">
        <RetryCountdown retryAfterSec={retryAfterSec ?? 60} onRetry={onRetry} />
      </div>
    </section>
  );
}
