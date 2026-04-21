import type { HealthDeep, HealthLayer } from "@/lib/ohi-types";
import { cn } from "@/lib/utils";

function layerColor(status: HealthLayer["status"]): string {
  if (status === "ok" || status === "up") return "bg-emerald-400";
  if (status === "degraded") return "bg-amber-400";
  if (status === "skipped") return "bg-slate-500";
  return "bg-rose-500";
}

export interface HealthMatrixProps {
  data: HealthDeep;
  className?: string;
}

export function HealthMatrix({ data, className }: HealthMatrixProps) {
  const layers = Object.entries(data.layers);
  const overall = data.status ?? data.overall ?? "unknown";
  return (
    <section
      className={cn(
        "overflow-hidden rounded-xl border border-[color:var(--border-subtle)] bg-[color:var(--surface-elevated)] shadow-sm",
        className,
      )}
    >
      <header className="border-b border-[color:var(--border-subtle)] bg-[color:var(--surface-soft)]/55 px-5 py-3">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-semibold text-[color:var(--brand-ink)]">
            System health — <span className="font-mono text-[color:var(--brand-muted)]">{overall}</span>
          </h2>
          <span className="font-mono text-[10px] text-[color:var(--brand-subtle)]">{data.timestamp}</span>
        </div>
      </header>

      <ol className="divide-y divide-[color:var(--border-subtle)]" data-testid="health-matrix">
        {layers.map(([name, l]) => (
          <li key={name} className="flex items-center gap-3 px-5 py-2.5 text-xs text-[color:var(--brand-ink)]">
            <span className={cn("h-2 w-2 rounded-full", layerColor(l.status))} />
            <span className="font-mono text-[color:var(--brand-ink)]">{name}</span>
            <span className="ml-auto flex items-center gap-3 font-mono text-[10px] text-[color:var(--brand-muted)]">
              {typeof l.latency_p50_ms === "number" && (
                <span title="p50 latency">p50 {l.latency_p50_ms.toFixed(0)} ms</span>
              )}
              {typeof l.latency_p95_ms === "number" && (
                <span title="p95 latency">p95 {l.latency_p95_ms.toFixed(0)} ms</span>
              )}
              {typeof l.latency_ms === "number" && (
                <span title="latency">{l.latency_ms.toFixed(1)} ms</span>
              )}
              {l.last_check && <span>{l.last_check}</span>}
              {l.detail && (
                <span className="max-w-[28ch] truncate text-[color:var(--brand-subtle)]" title={l.detail}>
                  {l.detail}
                </span>
              )}
            </span>
          </li>
        ))}
      </ol>

      {data.calibration ? (
        <footer className="border-t border-[color:var(--border-subtle)] bg-[color:var(--surface-soft)]/35 px-5 py-2 text-[10px] text-[color:var(--brand-muted)]">
          Calibration: {data.calibration.domains_fresh} fresh, {data.calibration.domains_stale} stale
          · last updated{" "}
          <span className="font-mono text-[color:var(--brand-subtle)]">
            {data.calibration.last_updated}
          </span>
        </footer>
      ) : null}
    </section>
  );
}
