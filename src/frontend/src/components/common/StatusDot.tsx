"use client";

import Link from "next/link";
import { useHealthDeep } from "@/lib/ohi-queries";
import { OhiError } from "@/lib/ohi-client";
import { cn } from "@/lib/utils";

type DotStatus = "green" | "amber" | "red" | "unknown";

function dotClass(status: DotStatus): string {
  if (status === "green")
    return "bg-brand-success shadow-[0_0_6px_rgba(5,150,105,0.55)]";
  if (status === "amber")
    return "bg-brand-warning shadow-[0_0_6px_rgba(217,119,6,0.55)]";
  if (status === "red")
    return "bg-brand-danger shadow-[0_0_6px_rgba(230,57,70,0.55)]";
  return "bg-brand-subtle";
}

function evaluate(
  data: ReturnType<typeof useHealthDeep>["data"],
  error: unknown,
): { status: DotStatus; label: string } {
  if (error) {
    if (error instanceof OhiError && error.isResting) return { status: "red", label: "resting" };
    return { status: "red", label: "offline" };
  }
  if (!data) return { status: "unknown", label: "checking…" };
  // Backend canonical field is `status`; legacy builds emit `overall`. Accept both.
  const overall = data.status ?? data.overall;
  if (overall === "healthy") return { status: "green", label: "healthy" };
  if (overall === "degraded") return { status: "amber", label: "degraded" };
  return { status: "red", label: "unhealthy" };
}

export interface StatusDotProps {
  className?: string;
  showLabel?: boolean;
}

export function StatusDot({ className, showLabel = false }: StatusDotProps) {
  const { data, error } = useHealthDeep();
  const { status, label } = evaluate(data, error);

  return (
    <Link
      href="/status"
      className={cn(
        "inline-flex items-center gap-1.5 text-[10px] text-brand-muted hover:text-brand-ink transition-colors",
        className,
      )}
      aria-label={`Service status: ${label}`}
      title={`Service status: ${label}`}
    >
      <span className={cn("h-2 w-2 rounded-full", dotClass(status))} />
      {showLabel && (
        <span className="font-mono uppercase tracking-wider">{label}</span>
      )}
    </Link>
  );
}
