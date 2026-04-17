import { cn } from "@/lib/utils";

export interface DegradedStateProps {
  degradedLayers: string[];
  fallbackCount?: number;
  className?: string;
}

export function DegradedState({ degradedLayers, fallbackCount, className }: DegradedStateProps) {
  if (degradedLayers.length === 0 && (fallbackCount ?? 0) === 0) return null;

  const parts: string[] = [];
  if (degradedLayers.length > 0) parts.push(`${degradedLayers.length} layer(s) degraded`);
  if (fallbackCount && fallbackCount > 0) parts.push(`${fallbackCount} claim(s) used a calibration fallback`);

  return (
    <aside
      className={cn(
        "rounded-md border border-yellow-400/25 bg-yellow-500/[0.06] p-3",
        className,
      )}
      role="status"
      data-testid="degraded-state"
    >
      <div className="flex items-center gap-2 text-xs text-yellow-200">
        <span className="text-base">⚠️</span>
        <span className="font-semibold">Partial result — {parts.join(" · ")}.</span>
      </div>
      <p className="mt-1 text-[11px] text-yellow-100/70">
        Claim intervals are wider than usual. Affected layers:
      </p>
      <div className="mt-1 flex flex-wrap gap-1">
        {degradedLayers.map((l) => (
          <span
            key={l}
            className="rounded-full bg-yellow-400/15 px-2 py-0.5 font-mono text-[10px] text-yellow-200 ring-1 ring-inset ring-yellow-400/25"
          >
            {l}
          </span>
        ))}
      </div>
    </aside>
  );
}
