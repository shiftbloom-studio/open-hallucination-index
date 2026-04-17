import type { DocumentVerdict } from "@/lib/ohi-types";
import { cn } from "@/lib/utils";
import { IntervalBar } from "./IntervalBar";

export interface DocumentVerdictCardProps {
  verdict: DocumentVerdict | null;
  className?: string;
}

function scoreColor(score: number): string {
  if (score >= 0.8) return "text-emerald-400";
  if (score >= 0.5) return "text-amber-400";
  return "text-rose-400";
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
          "rounded-xl border border-white/10 bg-white/[0.04] p-5 animate-pulse",
          className,
        )}
        data-testid="document-verdict-skeleton"
      >
        <div className="mb-3 h-2 w-20 rounded bg-white/10" />
        <div className="mb-2 h-8 w-24 rounded bg-white/10" />
        <div className="h-1.5 w-full rounded bg-white/10" />
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
        "rounded-xl border border-white/10 bg-white/[0.04] p-5 backdrop-blur-sm",
        className,
      )}
      aria-label="Document verdict"
      data-testid="document-verdict"
    >
      <div className="mb-2 flex items-center justify-between text-[10px] uppercase tracking-wider text-slate-400">
        <span>Document verdict</span>
        <span className="font-mono">{verdict.pipeline_version}</span>
      </div>

      <div className="flex items-end justify-between gap-6">
        <div>
          <div className={cn("font-mono text-4xl font-bold", scoreColor(document_score))}>
            {document_score.toFixed(2)}
          </div>
          <div className="mt-1 text-xs text-slate-400">
            interval [{document_interval[0].toFixed(2)}, {document_interval[1].toFixed(2)}]
          </div>
        </div>

        <dl className="grid grid-cols-2 gap-x-5 gap-y-1 text-right text-[11px] text-slate-300">
          <dt className="text-slate-500">internal consistency</dt>
          <dd className="font-mono text-slate-100">{internal_consistency.toFixed(2)}</dd>
          <dt className="text-slate-500">claims</dt>
          <dd className="font-mono text-slate-100">{claims.length}</dd>
          <dt className="text-slate-500">rigor</dt>
          <dd className="font-mono text-slate-100">{rigor}</dd>
          <dt className="text-slate-500">pipeline</dt>
          <dd className="font-mono text-slate-100">
            {fmtSeconds(processing_time_ms)}
            {refinement_passes_executed > 0 && (
              <span className="ml-1 text-slate-500">· {refinement_passes_executed} pass</span>
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
        <div className="mt-1 flex justify-between text-[9px] text-slate-500">
          <span>0</span>
          <span>0.5</span>
          <span>1</span>
        </div>
      </div>
    </section>
  );
}
