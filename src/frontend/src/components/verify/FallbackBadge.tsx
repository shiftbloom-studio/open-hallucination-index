import type { FallbackKind } from "@/lib/ohi-types";
import { cn } from "@/lib/utils";

// Labels are display strings — DO NOT rename or reorder.
// "general fallback" is load-bearing (Fabian's hard-rule: signals calibration
// state to operators reading the UI).
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
        "inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wider ring-1 ring-inset",
        className,
      )}
      style={{
        backgroundColor: "var(--brand-warning-soft)",
        color: "#92400e",
        boxShadow: "inset 0 0 0 1px rgba(217,119,6,0.35)",
      }}
      data-fallback={kind}
      title={`calibration used ${label[kind]} — interval widened`}
    >
      <span aria-hidden>⚠</span>
      {label[kind]}
    </span>
  );
}
