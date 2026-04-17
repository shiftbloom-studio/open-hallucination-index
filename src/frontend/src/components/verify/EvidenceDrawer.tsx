import type { Evidence } from "@/lib/ohi-types";
import { cn } from "@/lib/utils";

export interface EvidenceDrawerProps {
  supporting: Evidence[];
  refuting: Evidence[];
  open: boolean;
  className?: string;
}

function EvidenceItem({ e, kind }: { e: Evidence; kind: "support" | "refute" }) {
  const dot =
    kind === "support" ? "bg-emerald-400" : "bg-rose-400";
  const cred = typeof e.source_credibility === "number" ? e.source_credibility : null;
  return (
    <li className="group rounded-md border border-white/5 bg-slate-950/40 p-2 text-xs">
      <div className="flex items-start gap-2">
        <span className={cn("mt-1.5 h-1.5 w-1.5 flex-none rounded-full", dot)} />
        <div className="min-w-0 flex-1">
          <p className="text-slate-200 line-clamp-3">{e.content}</p>
          <div className="mt-1 flex flex-wrap items-center gap-2 text-[10px] text-slate-500">
            {e.source_uri && (
              <a
                href={e.source_uri}
                target="_blank"
                rel="noopener noreferrer"
                className="text-indigo-300 hover:text-indigo-200 hover:underline"
              >
                {new URL(e.source_uri).hostname}
              </a>
            )}
            {cred !== null && (
              <span className="font-mono" title="source credibility">
                cred {cred.toFixed(2)}
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
        "mt-3 grid grid-cols-1 gap-3 rounded-md border border-white/5 bg-slate-950/30 p-3 md:grid-cols-2",
        className,
      )}
      data-testid="evidence-drawer"
    >
      <div>
        <div className="mb-2 text-[10px] font-semibold uppercase tracking-wider text-emerald-400">
          Supporting ({supporting.length})
        </div>
        {supporting.length === 0 ? (
          <p className="text-xs text-slate-500">No supporting evidence retrieved.</p>
        ) : (
          <ul className="space-y-2">
            {supporting.map((e) => (
              <EvidenceItem key={e.id} e={e} kind="support" />
            ))}
          </ul>
        )}
      </div>
      <div>
        <div className="mb-2 text-[10px] font-semibold uppercase tracking-wider text-rose-400">
          Refuting ({refuting.length})
        </div>
        {refuting.length === 0 ? (
          <p className="text-xs text-slate-500">No refuting evidence retrieved.</p>
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
