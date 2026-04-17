"use client";

import { useEffect, useState } from "react";
import { cn } from "@/lib/utils";

export interface RetryCountdownProps {
  retryAfterSec?: number;
  onRetry?: () => void;
  primaryLabel?: string;
  className?: string;
}

function fmt(sec: number): string {
  if (sec >= 3600) {
    const h = Math.floor(sec / 3600);
    const m = Math.floor((sec % 3600) / 60);
    return `${h}h ${m.toString().padStart(2, "0")}m`;
  }
  const m = Math.floor(sec / 60);
  const s = sec % 60;
  return `${m.toString().padStart(2, "0")}:${s.toString().padStart(2, "0")}`;
}

export function RetryCountdown({
  retryAfterSec,
  onRetry,
  primaryLabel = "Retry now",
  className,
}: RetryCountdownProps) {
  const [remaining, setRemaining] = useState(retryAfterSec ?? 0);

  useEffect(() => {
    setRemaining(retryAfterSec ?? 0);
    if (!retryAfterSec || retryAfterSec <= 0) return;
    const id = setInterval(() => {
      setRemaining((s) => (s > 0 ? s - 1 : 0));
    }, 1000);
    return () => clearInterval(id);
  }, [retryAfterSec]);

  const canRetry = remaining <= 0;

  return (
    <div className={cn("flex items-center gap-3", className)}>
      <button
        type="button"
        onClick={onRetry}
        disabled={!canRetry || !onRetry}
        className={cn(
          "rounded-md px-3 py-1.5 text-xs font-semibold transition-colors",
          canRetry
            ? "bg-[color:var(--brand-indigo)] text-white shadow-sm hover:bg-[color:var(--brand-indigo-strong)]"
            : "cursor-not-allowed bg-[color:var(--border-subtle)] text-brand-subtle",
        )}
      >
        {canRetry ? primaryLabel : `Retry in ${fmt(remaining)}`}
      </button>
      {!canRetry && retryAfterSec && retryAfterSec > 0 && (
        <span
          className="num-mono text-[11px] text-[color:var(--brand-warning)]"
          aria-live="polite"
          data-testid="countdown"
        >
          {fmt(remaining)}
        </span>
      )}
    </div>
  );
}
