import { cn } from "@/lib/utils";

export interface NetworkErrorStateProps {
  onRetrySync?: () => void;
  detail?: string;
  className?: string;
}

export function NetworkErrorState({ onRetrySync, detail, className }: NetworkErrorStateProps) {
  return (
    <section
      className={cn(
        "rounded-xl border border-pink-400/25 bg-gradient-to-b from-pink-500/[0.04] to-transparent p-6",
        className,
      )}
      role="status"
      data-testid="network-error-state"
    >
      <div className="text-3xl">🛰️</div>
      <h2 className="mt-2 text-lg font-semibold text-slate-100">Network hiccup</h2>
      <p className="mt-1 text-sm font-semibold text-slate-200">
        The request couldn&apos;t reach OHI, or was cut off mid-stream.
      </p>
      <p className="mt-2 max-w-prose text-sm text-slate-400">
        Usually a transient network issue. If this persists, try the synchronous endpoint as a
        fallback.
      </p>
      {detail && <p className="mt-1 font-mono text-[11px] text-pink-300">{detail}</p>}
      <div className="mt-4 flex gap-3">
        <button
          type="button"
          onClick={onRetrySync}
          disabled={!onRetrySync}
          className="rounded-md bg-indigo-500/20 px-3 py-1.5 text-xs font-semibold text-indigo-100 ring-1 ring-inset ring-indigo-400/30 hover:bg-indigo-500/30 disabled:cursor-not-allowed disabled:opacity-40"
        >
          Retry with sync fallback
        </button>
      </div>
    </section>
  );
}
