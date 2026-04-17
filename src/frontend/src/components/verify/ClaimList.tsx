"use client";

import { useMemo, useState } from "react";
import type { ClaimVerdict } from "@/lib/ohi-types";
import { cn } from "@/lib/utils";
import { ClaimCard } from "./ClaimCard";

export type ClaimSortKey = "emitted" | "p_true_asc" | "information_gain_desc";

export interface ClaimListProps {
  claims: ClaimVerdict[];
  onShowInGraph?: (claimId: string) => void;
  onFlag?: (verdict: ClaimVerdict) => void;
  className?: string;
  emptyMessage?: string;
}

const sortLabels: Record<ClaimSortKey, string> = {
  emitted: "emitted order",
  p_true_asc: "lowest p_true first",
  information_gain_desc: "most informative first",
};

function sorted(claims: ClaimVerdict[], key: ClaimSortKey): ClaimVerdict[] {
  if (key === "emitted") return claims;
  const copy = [...claims];
  if (key === "p_true_asc") copy.sort((a, b) => a.p_true - b.p_true);
  else if (key === "information_gain_desc")
    copy.sort((a, b) => b.information_gain - a.information_gain);
  return copy;
}

export function ClaimList({
  claims,
  onShowInGraph,
  onFlag,
  className,
  emptyMessage = "No claims yet.",
}: ClaimListProps) {
  const [sortKey, setSortKey] = useState<ClaimSortKey>("emitted");
  const visible = useMemo(() => sorted(claims, sortKey), [claims, sortKey]);

  if (claims.length === 0) {
    return (
      <div
        className={cn(
          "rounded-xl border border-dashed border-[color:var(--border-default)] bg-surface-elevated p-6 text-center text-sm text-brand-subtle",
          className,
        )}
      >
        {emptyMessage}
      </div>
    );
  }

  return (
    <div className={cn("space-y-3", className)}>
      <div className="flex items-center justify-between text-brand-subtle">
        <span className="label-mono">Claims ({claims.length})</span>
        <label className="flex items-center gap-2 text-[10px] uppercase tracking-wider">
          <span>Sort</span>
          <select
            className="num-mono rounded border border-[color:var(--border-subtle)] bg-surface-elevated px-2 py-0.5 text-[10px] text-brand-ink focus:border-[color:var(--brand-indigo)] focus:outline-none"
            value={sortKey}
            onChange={(e) => setSortKey(e.target.value as ClaimSortKey)}
            aria-label="Sort claims"
          >
            {(Object.keys(sortLabels) as ClaimSortKey[]).map((k) => (
              <option key={k} value={k}>
                {sortLabels[k]}
              </option>
            ))}
          </select>
        </label>
      </div>
      {visible.map((v) => (
        <ClaimCard
          key={v.claim.id}
          verdict={v}
          onShowInGraph={onShowInGraph}
          onFlag={onFlag}
        />
      ))}
    </div>
  );
}
