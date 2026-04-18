import { cn } from "@/lib/utils";

export interface NetworkErrorStateProps {
  onRetrySync?: () => void;
  detail?: string;
  className?: string;
}

// The "Retry with sync fallback" button MUST dispatch submitSync — see
// phase2-handover §7.8. We only restyle; the callback wiring is preserved.
export function NetworkErrorState({ onRetrySync, detail, className }: NetworkErrorStateProps) {
  return (
    <section
      className={cn(
        "rounded-xl border border-[color:var(--brand-danger)]/30 bg-[color:var(--brand-danger-soft)]/50 p-6",
        className,
      )}
      role="status"
      data-testid="network-error-state"
    >
      <div className="text-3xl">🛰️</div>
      <h2 className="font-heading mt-2 text-lg font-semibold text-brand-ink">Network hiccup</h2>
      <p className="mt-1 text-sm font-medium text-brand-ink">
        The request couldn&apos;t reach OHI, or was cut off mid-stream.
      </p>
      <p className="mt-2 max-w-prose text-sm text-brand-muted">
        Usually a transient network issue. If this persists, try the synchronous endpoint as a
        fallback.
      </p>
      {detail && <p className="num-mono mt-1 text-[11px] text-[color:var(--brand-danger)]">{detail}</p>}
      <div className="mt-4 flex gap-3">
        <button
          type="button"
          onClick={onRetrySync}
          disabled={!onRetrySync}
          className="rounded-md bg-[color:var(--brand-indigo)] px-3 py-1.5 text-xs font-semibold text-white shadow-sm hover:bg-[color:var(--brand-indigo-strong)] disabled:cursor-not-allowed disabled:opacity-40"
        >
          Retry with sync fallback
        </button>
      </div>
    </section>
  );
}
