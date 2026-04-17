import type { Domain } from "@/lib/ohi-types";
import { cn } from "@/lib/utils";

const palette: Record<Domain, string> = {
  general: "bg-indigo-500/15 text-indigo-200 ring-indigo-400/25",
  biomedical: "bg-teal-500/15 text-teal-200 ring-teal-400/25",
  legal: "bg-amber-500/15 text-amber-200 ring-amber-400/25",
  code: "bg-violet-500/15 text-violet-200 ring-violet-400/25",
  social: "bg-pink-500/15 text-pink-200 ring-pink-400/25",
};

export interface DomainBadgeProps {
  domain: Domain;
  weight?: number;
  className?: string;
}

export function DomainBadge({ domain, weight, className }: DomainBadgeProps) {
  const showWeight = typeof weight === "number" && weight < 1 && weight > 0;
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wider ring-1 ring-inset",
        palette[domain],
        className,
      )}
      data-domain={domain}
    >
      {domain}
      {showWeight && <span className="font-mono text-[9px] opacity-70">·{weight!.toFixed(2)}</span>}
    </span>
  );
}
