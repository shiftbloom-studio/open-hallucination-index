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
      state:
        p.decomposition && p.routed.length >= p.decomposition.claim_count
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
      state: status === "complete" ? "done" : claimsCount > 0 ? "active" : "pending",
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

  return steps;
}

function dotStyle(state: StepState): { className: string; style: React.CSSProperties } {
  if (state === "done") {
    return {
      className: "",
      style: { background: "var(--brand-success)" },
    };
  }
  if (state === "active") {
    return {
      className: "",
      style: {
        background: "var(--brand-indigo)",
        boxShadow: "0 0 8px rgba(99,102,241,0.55)",
      },
    };
  }
  return {
    className: "bg-[color:var(--border-default)]",
    style: {},
  };
}

export function SseProgress({ status, progress, claimsRenderedCount, className }: SseProgressProps) {
  const steps = computeSteps(status, progress, claimsRenderedCount);

  return (
    <section
      className={cn(
        "rounded-xl border border-[color:var(--border-subtle)] bg-surface-elevated p-4 shadow-sm",
        className,
      )}
      aria-label="Pipeline progress"
      data-testid="sse-progress"
      data-status={status}
    >
      <div className="label-mono mb-2">Pipeline progress</div>
      <ol className="space-y-1.5">
        {steps.map((s) => {
          const d = dotStyle(s.state);
          return (
            <li
              key={s.key}
              className="flex items-center gap-2.5 text-[11px]"
              data-step={s.key}
              data-state={s.state}
            >
              <span className={cn("h-2 w-2 flex-none rounded-full", d.className)} style={d.style} />
              <span
                className={cn(
                  s.state === "done" && "text-brand-ink",
                  s.state === "active" && "text-[color:var(--brand-indigo-strong)] font-medium",
                  s.state === "pending" && "text-brand-subtle",
                )}
              >
                {s.label}
              </span>
              {s.detail && (
                <span className="num-mono ml-auto text-[10px] text-brand-subtle">{s.detail}</span>
              )}
            </li>
          );
        })}
      </ol>
    </section>
  );
}
