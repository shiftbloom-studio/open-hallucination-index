import { cn } from "@/lib/utils";

export type IntervalBarSize = "lg" | "sm";

export interface IntervalBarProps {
  pTrue: number;
  interval: [number, number];
  size?: IntervalBarSize;
  className?: string;
  ariaLabel?: string;
}

function band(pTrue: number): "high" | "mid" | "low" {
  if (pTrue >= 0.8) return "high";
  if (pTrue >= 0.5) return "mid";
  return "low";
}

const fillClass: Record<ReturnType<typeof band>, string> = {
  high: "bg-emerald-400/45",
  mid: "bg-amber-400/45",
  low: "bg-rose-500/45",
};

const tickClass: Record<ReturnType<typeof band>, string> = {
  high: "bg-emerald-200",
  mid: "bg-amber-200",
  low: "bg-rose-200",
};

const sizeClass: Record<IntervalBarSize, string> = {
  lg: "h-2",
  sm: "h-1",
};

function clamp(n: number): number {
  if (Number.isNaN(n)) return 0;
  if (n < 0) return 0;
  if (n > 1) return 1;
  return n;
}

export function IntervalBar({
  pTrue,
  interval,
  size = "lg",
  className,
  ariaLabel,
}: IntervalBarProps) {
  const [lower, upper] = interval;
  const low = clamp(Math.min(lower, upper));
  const up = clamp(Math.max(lower, upper));
  const p = clamp(pTrue);
  const b = band(p);

  const left = low * 100;
  const width = Math.max(0, (up - low) * 100);
  const tickLeft = p * 100;

  return (
    <div
      className={cn("relative w-full rounded-full bg-white/10", sizeClass[size], className)}
      role="progressbar"
      aria-label={ariaLabel ?? "probability interval"}
      aria-valuenow={Math.round(p * 100)}
      aria-valuemin={0}
      aria-valuemax={100}
      data-band={b}
    >
      <div
        className={cn("absolute inset-y-0 rounded-full", fillClass[b])}
        data-testid="interval-fill"
        style={{ left: `${left}%`, width: `${width}%` }}
      />
      <div
        className={cn(
          "absolute top-1/2 h-3 w-0.5 -translate-y-1/2 rounded-full",
          tickClass[b],
          size === "sm" && "h-2",
        )}
        data-testid="interval-tick"
        style={{ left: `calc(${tickLeft}% - 1px)` }}
      />
    </div>
  );
}
