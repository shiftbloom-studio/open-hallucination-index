import type { FallbackKind } from "@/lib/ohi-types";
import { cn } from "@/lib/utils";

const label: Record<FallbackKind, string> = {
  domain: "domain fallback",
  general: "general fallback",
  non_converged: "non-converged",
};

export interface FallbackBadgeProps {
  kind: FallbackKind | null | undefined;
  className?: string;
}

export function FallbackBadge({ kind, className }: FallbackBadgeProps) {
  if (!kind) return null;
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 rounded-full bg-amber-500/15 px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wider text-amber-300 ring-1 ring-inset ring-amber-400/25",
        className,
      )}
      data-fallback={kind}
      title={`calibration used ${label[kind]} — interval widened`}
    >
      ⚠ {label[kind]}
    </span>
  );
}
