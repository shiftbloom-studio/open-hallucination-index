import type { ProgressBag, VerifyStatus } from "@/lib/verify-controller";
import { cn } from "@/lib/utils";

export interface SseProgressProps {
  status: VerifyStatus;
  progress: ProgressBag;
  claimsRenderedCount: number;
  className?: string;
}

type StepState = "pending" | "active" | "done";

interface Step {
  key: string;
  label: string;
  state: StepState;
  detail?: string;
}

function computeSteps(
  status: VerifyStatus,
  p: ProgressBag,
  claimsCount: number,
): Step[] {
  const steps: Step[] = [
    {
      key: "decomposition",
      label: "Decomposition",
      state: p.decomposition ? "done" : status === "streaming" ? "active" : "pending",
      detail: p.decomposition ? `${p.decomposition.claim_count} claims` : undefined,
    },
    {
      key: "routing",
      label: "Domain routing",
      state: p.decomposition && p.routed.length >= p.decomposition.claim_count
        ? "done"
        : p.decomposition
          ? "active"
          : "pending",
      detail: p.decomposition ? `${p.routed.length}/${p.decomposition.claim_count}` : undefined,
    },
    {
      key: "nli",
      label: "NLI cross-encoder",
      state: p.nli ? "done" : p.routed.length > 0 ? "active" : "pending",
      detail: p.nli
        ? `${p.nli.claim_evidence_pairs_scored} ev · ${p.nli.claim_pair_pairs_scored} pairs`
        : undefined,
    },
    {
      key: "pcg",
      label: "PCG propagation",
      state: p.pcg ? "done" : p.nli ? "active" : "pending",
      detail: p.pcg
        ? `${p.pcg.algorithm} · iter ${p.pcg.iterations} · ic ${p.pcg.internal_consistency.toFixed(2)}`
        : undefined,
    },
    {
      key: "refinement",
      label: "Refinement",
      state: p.refinement ? "done" : p.pcg ? "active" : "pending",
      detail: p.refinement
        ? `pass ${p.refinement.pass} · Δmax ${p.refinement.marginal_max_change.toFixed(2)}`
        : undefined,
    },
    {
      key: "verdicts",
      label: "Claim verdicts",
      state:
        status === "complete" ? "done" : claimsCount > 0 ? "active" : "pending",
      detail:
        p.decomposition && claimsCount > 0
          ? `${claimsCount}/${p.decomposition.claim_count}`
          : undefined,
    },
    {
      key: "document",
      label: "Document verdict",
      state: status === "complete" ? "done" : "pending",
    },
  ];

  if (status === "error" || status === "sync_fallback") {
    // Freeze currently active step as-is.
    return steps;
  }
  return steps;
}

function dotClass(state: StepState): string {
  if (state === "done") return "bg-emerald-400";
  if (state === "active") return "bg-amber-400 shadow-[0_0_8px_theme(colors.amber.300)]";
  return "bg-white/15";
}

export function SseProgress({ status, progress, claimsRenderedCount, className }: SseProgressProps) {
  const steps = computeSteps(status, progress, claimsRenderedCount);

  return (
    <section
      className={cn("rounded-xl border border-white/10 bg-white/[0.03] p-4", className)}
      aria-label="Pipeline progress"
      data-testid="sse-progress"
      data-status={status}
    >
      <div className="mb-2 text-[10px] font-semibold uppercase tracking-wider text-slate-400">
        Pipeline progress
      </div>
      <ol className="space-y-1.5">
        {steps.map((s) => (
          <li
            key={s.key}
            className="flex items-center gap-2.5 text-[11px]"
            data-step={s.key}
            data-state={s.state}
          >
            <span className={cn("h-2 w-2 flex-none rounded-full", dotClass(s.state))} />
            <span
              className={cn(
                s.state === "done" && "text-slate-200",
                s.state === "active" && "text-amber-200",
                s.state === "pending" && "text-slate-500",
              )}
            >
              {s.label}
            </span>
            {s.detail && (
              <span className="ml-auto font-mono text-[10px] text-slate-500">{s.detail}</span>
            )}
          </li>
        ))}
      </ol>
    </section>
  );
}
