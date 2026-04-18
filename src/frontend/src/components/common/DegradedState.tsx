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
  if (fallbackCount && fallbackCount > 0)
    parts.push(`${fallbackCount} claim(s) used a calibration fallback`);

  return (
    <aside
      className={cn(
        "rounded-md border border-[color:var(--brand-warning)]/30 bg-[color:var(--brand-warning-soft)]/60 p-3",
        className,
      )}
      role="status"
      data-testid="degraded-state"
    >
      <div className="flex items-center gap-2 text-xs" style={{ color: "#92400e" }}>
        <span className="text-base">⚠️</span>
        <span className="font-semibold">Partial result — {parts.join(" · ")}.</span>
      </div>
      <p className="mt-1 text-[11px]" style={{ color: "#b45309" }}>
        Claim intervals are wider than usual. Affected layers:
      </p>
      <div className="mt-1 flex flex-wrap gap-1">
        {degradedLayers.map((l) => (
          <span
            key={l}
            className="num-mono rounded-full px-2 py-0.5 text-[10px] ring-1 ring-inset"
            style={{
              backgroundColor: "var(--brand-warning-soft)",
              color: "#92400e",
              boxShadow: "inset 0 0 0 1px rgba(217,119,6,0.3)",
            }}
          >
            {l}
          </span>
        ))}
      </div>
    </aside>
  );
}
