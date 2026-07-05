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
        "rounded-lg border border-[color:var(--brand-primary)]/30 bg-[color:var(--brand-secondary)]/70 p-6",
        className,
      )}
      role="status"
      data-testid="rate-limited-state"
    >
      <p className="sb-kicker text-[color:var(--brand-accent)]">Limit</p>
      <h2 className="mt-2 text-lg font-semibold text-brand-ink">
        Rate limit reached
      </h2>
      <p className="mt-1 text-sm font-medium text-brand-ink">
        You&apos;ve hit the per-IP rate limit for{" "}
        <code className="font-mono text-[color:var(--brand-accent)]">/verify</code>.
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
