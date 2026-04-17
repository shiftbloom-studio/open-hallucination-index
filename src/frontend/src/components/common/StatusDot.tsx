"use client";

import Link from "next/link";
import { useHealthDeep } from "@/lib/ohi-queries";
import { OhiError } from "@/lib/ohi-client";
import { cn } from "@/lib/utils";

function colorFor(status: "green" | "amber" | "red" | "unknown"): string {
  if (status === "green") return "bg-emerald-400 shadow-[0_0_8px_theme(colors.emerald.400)]";
  if (status === "amber") return "bg-amber-400 shadow-[0_0_8px_theme(colors.amber.400)]";
  if (status === "red") return "bg-rose-500 shadow-[0_0_8px_theme(colors.rose.500)]";
  return "bg-slate-500";
}

function evaluate(
  data: ReturnType<typeof useHealthDeep>["data"],
  error: unknown,
): { status: "green" | "amber" | "red" | "unknown"; label: string } {
  if (error) {
    if (error instanceof OhiError && error.isResting) return { status: "red", label: "resting" };
    return { status: "red", label: "offline" };
  }
  if (!data) return { status: "unknown", label: "checking…" };
  if (data.overall === "healthy") return { status: "green", label: "healthy" };
  if (data.overall === "degraded") return { status: "amber", label: "degraded" };
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
      className={cn("inline-flex items-center gap-1.5 text-[10px] text-slate-400 hover:text-slate-200", className)}
      aria-label={`Service status: ${label}`}
      title={`Service status: ${label}`}
    >
      <span className={cn("h-2 w-2 rounded-full", colorFor(status))} />
      {showLabel && <span className="uppercase tracking-wider">{label}</span>}
    </Link>
  );
}
