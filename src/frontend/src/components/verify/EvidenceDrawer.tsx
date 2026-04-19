import type { Evidence } from "@/lib/ohi-types";
import { cn } from "@/lib/utils";

export interface EvidenceDrawerProps {
  supporting: Evidence[];
  refuting: Evidence[];
  open: boolean;
  className?: string;
}

function formatScore(value: number | null | undefined): string | null {
  return typeof value === "number" ? value.toFixed(2) : null;
}

function hostnameFor(url: string | null): string | null {
  if (!url) return null;
  try {
    return new URL(url).hostname;
  } catch {
    return null;
  }
}

function EvidenceItem({ e, kind }: { e: Evidence; kind: "support" | "refute" }) {
  const dotColor =
    kind === "support" ? "var(--brand-success)" : "var(--brand-danger)";
  const title = typeof e.structured_data?.title === "string" ? e.structured_data.title : null;
  const reasoning =
    typeof e.structured_data?.nli_reasoning === "string" ? e.structured_data.nli_reasoning : null;
  const bucketScore = formatScore(
    typeof e.structured_data?.bucket_score === "number"
      ? e.structured_data.bucket_score
      : e.classification_confidence,
  );
  const relevance = formatScore(e.similarity_score);
  const cred = formatScore(e.source_credibility);
  const host = hostnameFor(e.source_uri);
  return (
    <li className="group rounded-md border border-[color:var(--border-subtle)] bg-surface-base p-2 text-xs">
      <div className="flex items-start gap-2">
        <span
          className="mt-1.5 h-1.5 w-1.5 flex-none rounded-full"
          style={{ background: dotColor }}
        />
        <div className="min-w-0 flex-1">
          {title && (
            <p className="mb-1 text-[10px] font-semibold uppercase tracking-wider text-brand-subtle">
              {title}
            </p>
          )}
          <p className="text-brand-ink">{e.content}</p>
          {reasoning && (
            <p className="mt-2 text-[11px] italic text-brand-muted">
              {reasoning}
            </p>
          )}
          <div className="mt-1 flex flex-wrap items-center gap-2 text-[10px] text-brand-subtle">
            {host && e.source_uri && (
              <a
                href={e.source_uri}
                target="_blank"
                rel="noopener noreferrer"
                className="text-[color:var(--brand-indigo-strong)] hover:underline"
              >
                {host}
              </a>
            )}
            {bucketScore && (
              <span className="num-mono" title={`${kind} score`}>
                {kind} {bucketScore}
              </span>
            )}
            {relevance && (
              <span className="num-mono" title="retrieval relevance">
                rel {relevance}
              </span>
            )}
            {cred && (
              <span className="num-mono" title="source credibility">
                cred {cred}
              </span>
            )}
          </div>
        </div>
      </div>
    </li>
  );
}

export function EvidenceDrawer({ supporting, refuting, open, className }: EvidenceDrawerProps) {
  if (!open) return null;
  return (
    <div
      className={cn(
        "mt-3 grid grid-cols-1 gap-3 rounded-md border border-[color:var(--border-subtle)] bg-[color:var(--surface-soft)]/40 p-3 md:grid-cols-2",
        className,
      )}
      data-testid="evidence-drawer"
    >
      <div>
        <div
          className="mb-2 text-[10px] font-semibold uppercase tracking-wider"
          style={{ color: "var(--brand-success)" }}
        >
          Supporting ({supporting.length})
        </div>
        {supporting.length === 0 ? (
          <p className="text-xs text-brand-subtle">No supporting evidence retrieved.</p>
        ) : (
          <ul className="space-y-2">
            {supporting.map((e) => (
              <EvidenceItem key={e.id} e={e} kind="support" />
            ))}
          </ul>
        )}
      </div>
      <div>
        <div
          className="mb-2 text-[10px] font-semibold uppercase tracking-wider"
          style={{ color: "var(--brand-danger)" }}
        >
          Refuting ({refuting.length})
        </div>
        {refuting.length === 0 ? (
          <p className="text-xs text-brand-subtle">No refuting evidence retrieved.</p>
        ) : (
          <ul className="space-y-2">
            {refuting.map((e) => (
              <EvidenceItem key={e.id} e={e} kind="refute" />
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}
