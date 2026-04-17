import type { HealthDeep } from "@/lib/ohi-types";
import { cn } from "@/lib/utils";

function layerColor(status: "up" | "degraded" | "down"): string {
  if (status === "up") return "bg-emerald-400";
  if (status === "degraded") return "bg-amber-400";
  return "bg-rose-500";
}

export interface HealthMatrixProps {
  data: HealthDeep;
  className?: string;
}

export function HealthMatrix({ data, className }: HealthMatrixProps) {
  const layers = Object.entries(data.layers);
  return (
    <section className={cn("rounded-xl border border-white/10 bg-white/[0.03]", className)}>
      <header className="border-b border-white/10 px-5 py-3">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-semibold text-slate-100">
            System health — <span className="font-mono text-slate-400">{data.overall}</span>
          </h2>
          <span className="font-mono text-[10px] text-slate-500">{data.timestamp}</span>
        </div>
      </header>

      <ol className="divide-y divide-white/5" data-testid="health-matrix">
        {layers.map(([name, l]) => (
          <li key={name} className="flex items-center gap-3 px-5 py-2.5 text-xs">
            <span className={cn("h-2 w-2 rounded-full", layerColor(l.status))} />
            <span className="font-mono text-slate-200">{name}</span>
            <span className="ml-auto flex items-center gap-3 font-mono text-[10px] text-slate-400">
              {typeof l.latency_p50_ms === "number" && (
                <span title="p50 latency">p50 {l.latency_p50_ms.toFixed(0)} ms</span>
              )}
              {typeof l.latency_p95_ms === "number" && (
                <span title="p95 latency">p95 {l.latency_p95_ms.toFixed(0)} ms</span>
              )}
              <span>{l.last_check}</span>
            </span>
          </li>
        ))}
      </ol>

      <footer className="border-t border-white/10 px-5 py-2 text-[10px] text-slate-500">
        Calibration: {data.calibration.domains_fresh} fresh, {data.calibration.domains_stale} stale
        · last updated <span className="font-mono text-slate-400">{data.calibration.last_updated}</span>
      </footer>
    </section>
  );
}
