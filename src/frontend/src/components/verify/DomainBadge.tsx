import type { Domain } from "@/lib/ohi-types";
import { cn } from "@/lib/utils";

const palette: Record<
  Domain,
  { bg: string; fg: string; ring: string }
> = {
  general: {
    bg: "rgba(99,102,241,0.1)",
    fg: "var(--brand-indigo-strong)",
    ring: "rgba(99,102,241,0.3)",
  },
  biomedical: {
    bg: "rgba(13,148,136,0.1)",
    fg: "#0f766e",
    ring: "rgba(13,148,136,0.3)",
  },
  legal: {
    bg: "rgba(217,119,6,0.1)",
    fg: "var(--brand-warning)",
    ring: "rgba(217,119,6,0.3)",
  },
  code: {
    bg: "rgba(139,92,246,0.1)",
    fg: "#6d28d9",
    ring: "rgba(139,92,246,0.3)",
  },
  social: {
    bg: "rgba(219,39,119,0.1)",
    fg: "#be185d",
    ring: "rgba(219,39,119,0.3)",
  },
};

export interface DomainBadgeProps {
  domain: Domain;
  weight?: number;
  className?: string;
}

export function DomainBadge({ domain, weight, className }: DomainBadgeProps) {
  const showWeight = typeof weight === "number" && weight < 1 && weight > 0;
  const p = palette[domain];
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wider ring-1 ring-inset",
        className,
      )}
      style={{ backgroundColor: p.bg, color: p.fg, boxShadow: `inset 0 0 0 1px ${p.ring}` }}
      data-domain={domain}
    >
      {domain}
      {showWeight && (
        <span className="num-mono text-[9px] opacity-75">·{weight!.toFixed(2)}</span>
      )}
    </span>
  );
}
