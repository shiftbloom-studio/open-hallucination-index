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
  if (p >= 0.8) return "var(--brand-success)";
  if (p >= 0.5) return "var(--brand-warning)";
  return "var(--brand-danger)";
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
        "rounded-xl border border-[color:var(--border-subtle)] bg-surface-elevated p-4 shadow-sm transition-colors hover:border-[color:var(--border-default)]",
        className,
      )}
      data-testid="claim-card"
      data-claim-id={v.claim.id}
    >
      <div className="flex items-baseline justify-between gap-3">
        <div
          className="num-mono text-xl font-semibold"
          style={{ color: scoreColor(v.p_true) }}
        >
          {v.p_true.toFixed(2)}
        </div>
        <div className="num-mono text-[10px] text-brand-subtle">
          [{v.interval[0].toFixed(2)}, {v.interval[1].toFixed(2)}]
          {v.coverage_target !== null && (
            <span className="ml-1">@ {Math.round(v.coverage_target * 100)}%</span>
          )}
        </div>
      </div>

      <p className="mt-2 text-sm leading-relaxed text-brand-ink">&ldquo;{v.claim.text}&rdquo;</p>

      <div className="mt-3">
        <IntervalBar pTrue={v.p_true} interval={v.interval} size="sm" />
      </div>

      <div className="mt-3 flex flex-wrap items-center gap-2 text-[10px] text-brand-muted">
        <DomainBadge domain={v.domain} weight={v.domain_assignment_weights[v.domain]} />
        <FallbackBadge kind={v.fallback_used} />
        {v.queued_for_review && (
          <span
            className="rounded-full px-2 py-0.5 font-semibold uppercase tracking-wider ring-1 ring-inset"
            style={{
              backgroundColor: "var(--brand-info-soft)",
              color: "var(--brand-info)",
              boxShadow: "inset 0 0 0 1px rgba(2,132,199,0.3)",
            }}
          >
            review queued
          </span>
        )}
        <span className="num-mono">
          {supportN} support · {refuteN} refute
          {neighborCount > 0 && ` · ${neighborCount} PCG neighbor${neighborCount > 1 ? "s" : ""}`}
        </span>
      </div>

      <div className="mt-3 flex gap-3 text-xs">
        <button
          type="button"
          className="text-brand-muted hover:text-brand-ink"
          onClick={() => setExpanded((v) => !v)}
          aria-expanded={expanded}
          aria-controls={`evidence-${v.claim.id}`}
        >
          {expanded ? "Hide evidence" : "Expand evidence"}
        </button>
        {neighborCount > 0 && onShowInGraph && (
          <button
            type="button"
            className="text-brand-muted hover:text-brand-ink"
            onClick={() => onShowInGraph(v.claim.id)}
          >
            Show in graph
          </button>
        )}
        {onFlag && (
          <button
            type="button"
            className="ml-auto text-brand-subtle hover:text-[color:var(--brand-warning)]"
            onClick={() => onFlag(v)}
            aria-label="Flag this claim for review"
          >
            <span aria-hidden>🚩</span> Flag
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
