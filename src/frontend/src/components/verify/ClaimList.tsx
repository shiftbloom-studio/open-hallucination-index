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
        className={cn("rounded-xl border border-dashed border-white/10 p-6 text-center text-sm text-slate-500", className)}
      >
        {emptyMessage}
      </div>
    );
  }

  return (
    <div className={cn("space-y-3", className)}>
      <div className="flex items-center justify-between text-[10px] uppercase tracking-wider text-slate-500">
        <span>Claims ({claims.length})</span>
        <label className="flex items-center gap-2">
          <span>Sort</span>
          <select
            className="rounded border border-white/10 bg-slate-900/60 px-2 py-0.5 font-mono text-[10px] text-slate-200 focus:border-indigo-400 focus:outline-none"
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
