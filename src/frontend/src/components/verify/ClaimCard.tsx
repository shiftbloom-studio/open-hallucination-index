"use client";

import { useState } from "react";
import type { ClaimVerdict } from "@/lib/ohi-types";
import { cn } from "@/lib/utils";
import { IntervalBar } from "./IntervalBar";
import { DomainBadge } from "./DomainBadge";
import { FallbackBadge } from "./FallbackBadge";
import { EvidenceDrawer } from "./EvidenceDrawer";

export interface ClaimCardProps {
  verdict: ClaimVerdict;
  onShowInGraph?: (claimId: string) => void;
  onFlag?: (verdict: ClaimVerdict) => void;
  className?: string;
}

function scoreColor(p: number): string {
  if (p >= 0.8) return "text-emerald-400";
  if (p >= 0.5) return "text-amber-400";
  return "text-rose-400";
}

export function ClaimCard({ verdict, onShowInGraph, onFlag, className }: ClaimCardProps) {
  const [expanded, setExpanded] = useState(false);
  const v = verdict;
  const neighborCount = v.pcg_neighbors.length;
  const supportN = v.supporting_evidence.length;
  const refuteN = v.refuting_evidence.length;

  return (
    <article
      className={cn(
        "rounded-xl border border-white/10 bg-white/[0.03] p-4 transition-colors hover:border-white/20",
        className,
      )}
      data-testid="claim-card"
      data-claim-id={v.claim.id}
    >
      <div className="flex items-baseline justify-between gap-3">
        <div className={cn("font-mono text-xl font-bold", scoreColor(v.p_true))}>
          {v.p_true.toFixed(2)}
        </div>
        <div className="font-mono text-[10px] text-slate-500">
          [{v.interval[0].toFixed(2)}, {v.interval[1].toFixed(2)}]
          {v.coverage_target !== null && (
            <span className="ml-1">@ {Math.round(v.coverage_target * 100)}%</span>
          )}
        </div>
      </div>

      <p className="mt-2 text-sm leading-relaxed text-slate-100">&ldquo;{v.claim.text}&rdquo;</p>

      <div className="mt-3">
        <IntervalBar pTrue={v.p_true} interval={v.interval} size="sm" />
      </div>

      <div className="mt-3 flex flex-wrap items-center gap-2 text-[10px] text-slate-400">
        <DomainBadge domain={v.domain} weight={v.domain_assignment_weights[v.domain]} />
        <FallbackBadge kind={v.fallback_used} />
        {v.queued_for_review && (
          <span className="rounded-full bg-sky-500/15 px-2 py-0.5 font-semibold uppercase tracking-wider text-sky-300 ring-1 ring-inset ring-sky-400/25">
            review queued
          </span>
        )}
        <span className="font-mono">
          {supportN} support · {refuteN} refute
          {neighborCount > 0 && ` · ${neighborCount} PCG neighbor${neighborCount > 1 ? "s" : ""}`}
        </span>
      </div>

      <div className="mt-3 flex gap-3 text-xs">
        <button
          type="button"
          className="text-slate-400 hover:text-slate-200"
          onClick={() => setExpanded((v) => !v)}
          aria-expanded={expanded}
          aria-controls={`evidence-${v.claim.id}`}
        >
          {expanded ? "Hide evidence" : "Expand evidence"}
        </button>
        {neighborCount > 0 && onShowInGraph && (
          <button
            type="button"
            className="text-slate-400 hover:text-slate-200"
            onClick={() => onShowInGraph(v.claim.id)}
          >
            Show in graph
          </button>
        )}
        {onFlag && (
          <button
            type="button"
            className="ml-auto text-slate-500 hover:text-amber-300"
            onClick={() => onFlag(v)}
            aria-label="Flag this claim for review"
          >
            🚩 Flag
          </button>
        )}
      </div>

      <div id={`evidence-${v.claim.id}`}>
        <EvidenceDrawer
          supporting={v.supporting_evidence}
          refuting={v.refuting_evidence}
          open={expanded}
        />
      </div>
    </article>
  );
}
