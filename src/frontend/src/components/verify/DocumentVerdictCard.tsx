import type { DocumentVerdict } from "@/lib/ohi-types";
import { cn } from "@/lib/utils";
import { IntervalBar } from "./IntervalBar";

export interface DocumentVerdictCardProps {
  verdict: DocumentVerdict | null;
  className?: string;
}

function scoreColor(score: number): string {
  if (score >= 0.8) return "var(--brand-success)";
  if (score >= 0.5) return "var(--brand-warning)";
  return "var(--brand-danger)";
}

function fmtSeconds(ms: number): string {
  if (ms < 1000) return `${Math.round(ms)} ms`;
  return `${(ms / 1000).toFixed(1)} s`;
}

export function DocumentVerdictCard({ verdict, className }: DocumentVerdictCardProps) {
  if (!verdict) {
    return (
      <div
        className={cn(
          "rounded-xl border border-[color:var(--border-subtle)] bg-surface-elevated p-5 shadow-sm animate-pulse",
          className,
        )}
        data-testid="document-verdict-skeleton"
      >
        <div className="mb-3 h-2 w-20 rounded bg-[color:var(--border-subtle)]" />
        <div className="mb-2 h-8 w-24 rounded bg-[color:var(--border-subtle)]" />
        <div className="h-1.5 w-full rounded bg-[color:var(--border-subtle)]" />
      </div>
    );
  }

  const {
    document_score,
    document_interval,
    internal_consistency,
    rigor,
    processing_time_ms,
    refinement_passes_executed,
    claims,
  } = verdict;

  return (
    <section
      className={cn(
        "rounded-xl border border-[color:var(--border-subtle)] bg-surface-elevated p-5 shadow-sm",
        className,
      )}
      aria-label="Document verdict"
      data-testid="document-verdict"
    >
      <div className="mb-2 flex items-center justify-between">
        <span className="label-mono">Document verdict</span>
        <span className="num-mono text-[10px] text-brand-subtle">{verdict.pipeline_version}</span>
      </div>

      <div className="flex items-end justify-between gap-6">
        <div>
          <div
            className="num-mono font-display text-4xl font-semibold"
            style={{ color: scoreColor(document_score) }}
          >
            {document_score.toFixed(2)}
          </div>
          <div className="mt-1 text-xs text-brand-muted">
            interval [{document_interval[0].toFixed(2)}, {document_interval[1].toFixed(2)}]
          </div>
        </div>

        <dl className="grid grid-cols-2 gap-x-5 gap-y-1 text-right text-[11px]">
          <dt className="text-brand-subtle">internal consistency</dt>
          <dd className="num-mono text-brand-ink">{internal_consistency.toFixed(2)}</dd>
          <dt className="text-brand-subtle">claims</dt>
          <dd className="num-mono text-brand-ink">{claims.length}</dd>
          <dt className="text-brand-subtle">rigor</dt>
          <dd className="num-mono text-brand-ink">{rigor}</dd>
          <dt className="text-brand-subtle">pipeline</dt>
          <dd className="num-mono text-brand-ink">
            {fmtSeconds(processing_time_ms)}
            {refinement_passes_executed > 0 && (
              <span className="ml-1 text-brand-subtle">· {refinement_passes_executed} pass</span>
            )}
          </dd>
        </dl>
      </div>

      <div className="mt-4">
        <IntervalBar
          pTrue={document_score}
          interval={document_interval}
          size="lg"
          ariaLabel="document probability interval"
        />
        <div className="mt-1 flex justify-between text-[9px] text-brand-subtle">
          <span>0</span>
          <span>0.5</span>
          <span>1</span>
        </div>
      </div>
    </section>
  );
}
