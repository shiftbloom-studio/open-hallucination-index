import { cn } from "@/lib/utils";
import { RetryCountdown } from "./RetryCountdown";

export interface LlmUnavailableStateProps {
  onRetry?: () => void;
  className?: string;
}

export function LlmUnavailableState({ onRetry, className }: LlmUnavailableStateProps) {
  return (
    <section
      className={cn(
        "rounded-xl border border-[color:var(--brand-danger)]/30 bg-[color:var(--brand-danger-soft)]/50 p-6",
        className,
      )}
      role="status"
      data-testid="llm-unavailable-state"
    >
      <div className="text-3xl">⚡</div>
      <h2 className="font-heading mt-2 text-lg font-semibold text-brand-ink">
        Upstream LLM unavailable
      </h2>
      <p className="mt-1 text-sm font-medium text-brand-ink">
        Google Gemini — OHI&apos;s sole Phase-1 verification LLM — is not responding.
      </p>
      <p className="mt-2 max-w-prose text-sm text-brand-muted">
        This is outside our control. Retry in a moment; if it persists, please report it.
      </p>
      <div className="mt-4 flex flex-wrap items-center gap-3">
        <RetryCountdown onRetry={onRetry} />
        <a
          href="https://github.com/shiftbloom-studio/open-hallucination-index/issues"
          target="_blank"
          rel="noopener noreferrer"
          className="text-xs text-brand-muted underline-offset-2 hover:text-brand-ink hover:underline"
        >
          Report on GitHub
        </a>
      </div>
    </section>
  );
}
