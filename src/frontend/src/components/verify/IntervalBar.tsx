import { cn } from "@/lib/utils";

export type IntervalBarSize = "lg" | "sm";

export interface IntervalBarProps {
  pTrue: number;
  interval: [number, number];
  size?: IntervalBarSize;
  className?: string;
  ariaLabel?: string;
}

type Band = "high" | "mid" | "low";

function band(pTrue: number): Band {
  if (pTrue >= 0.8) return "high";
  if (pTrue >= 0.5) return "mid";
  return "low";
}

// Semantic brand tokens; each band keeps a soft fill + a saturated tick so the
// point estimate stays readable on the fill.
const fillStyle: Record<Band, { background: string }> = {
  high: { background: "rgba(5,150,105,0.22)" }, // brand-success @ 22%
  mid: { background: "rgba(217,119,6,0.22)" }, // brand-warning @ 22%
  low: { background: "rgba(230,57,70,0.22)" }, // brand-danger @ 22%
};

const tickStyle: Record<Band, { background: string }> = {
  high: { background: "var(--brand-success)" },
  mid: { background: "var(--brand-warning)" },
  low: { background: "var(--brand-danger)" },
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
      className={cn(
        "relative w-full rounded-full bg-[color:var(--border-subtle)]",
        sizeClass[size],
        className,
      )}
      role="progressbar"
      aria-label={ariaLabel ?? "probability interval"}
      aria-valuenow={Math.round(p * 100)}
      aria-valuemin={0}
      aria-valuemax={100}
      data-band={b}
    >
      <div
        className="absolute inset-y-0 rounded-full"
        style={{ ...fillStyle[b], left: `${left}%`, width: `${width}%` }}
        data-testid="interval-fill"
      />
      <div
        className={cn(
          "absolute top-1/2 h-3 w-0.5 -translate-y-1/2 rounded-full",
          size === "sm" && "h-2",
        )}
        style={{ ...tickStyle[b], left: `calc(${tickLeft}% - 1px)` }}
        data-testid="interval-tick"
      />
    </div>
  );
}
