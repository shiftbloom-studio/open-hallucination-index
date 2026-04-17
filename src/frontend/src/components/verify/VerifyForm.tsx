"use client";

import { useState } from "react";
import type { Domain, Rigor, VerifyRequest } from "@/lib/ohi-types";
import { cn } from "@/lib/utils";

const MAX_CHARS = 50_000;

const RIGORS: Rigor[] = ["fast", "balanced", "maximum"];
const DOMAINS: ("auto" | Domain)[] = ["auto", "general", "biomedical", "legal", "code", "social"];

export interface VerifyFormProps {
  onSubmit: (req: VerifyRequest) => void;
  onCancel?: () => void;
  streaming?: boolean;
  className?: string;
}

function Chip({
  label,
  active,
  onClick,
  disabled,
}: {
  label: string;
  active: boolean;
  onClick: () => void;
  disabled?: boolean;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      className={cn(
        "rounded-full border px-2.5 py-0.5 text-[10px] font-semibold uppercase tracking-wider transition-colors",
        active
          ? "border-indigo-400/60 bg-indigo-500/20 text-indigo-100"
          : "border-white/10 bg-white/[0.03] text-slate-400 hover:border-white/20 hover:text-slate-200",
        disabled && "cursor-not-allowed opacity-50",
      )}
      data-active={active}
    >
      {label}
    </button>
  );
}

export function VerifyForm({ onSubmit, onCancel, streaming = false, className }: VerifyFormProps) {
  const [text, setText] = useState("");
  const [rigor, setRigor] = useState<Rigor>("balanced");
  const [domainHint, setDomainHint] = useState<"auto" | Domain>("auto");
  const [coverage, setCoverage] = useState(0.9);

  const chars = text.length;
  const overLimit = chars > MAX_CHARS;
  const canSubmit = !streaming && text.trim().length > 0 && !overLimit;

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!canSubmit) return;
    const req: VerifyRequest = {
      text,
      domain_hint: domainHint === "auto" ? null : domainHint,
      options: {
        rigor,
        coverage_target: coverage,
      },
    };
    onSubmit(req);
  }

  return (
    <form
      onSubmit={handleSubmit}
      className={cn(
        "space-y-3 rounded-xl border border-white/10 bg-white/[0.03] p-4",
        className,
      )}
      aria-label="Verify text"
    >
      <label className="block">
        <span className="mb-1.5 block text-[10px] font-semibold uppercase tracking-wider text-slate-400">
          Input text
        </span>
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          rows={8}
          disabled={streaming}
          maxLength={MAX_CHARS + 1000}
          placeholder={'Paste an AI-generated paragraph here. e.g. "In 1905 Einstein published four papers…"'}
          className="w-full resize-y rounded-md border border-white/10 bg-slate-950/60 p-2.5 font-mono text-xs leading-relaxed text-slate-100 focus:border-indigo-400 focus:outline-none disabled:opacity-60"
          aria-invalid={overLimit}
          aria-describedby="char-counter"
        />
        <div
          id="char-counter"
          className={cn(
            "mt-1 flex justify-between text-[10px]",
            overLimit ? "text-rose-400" : "text-slate-500",
          )}
        >
          <span>{overLimit && "Too long. Max 50,000 characters."}</span>
          <span className="font-mono">
            {chars.toLocaleString()} / {MAX_CHARS.toLocaleString()}
          </span>
        </div>
      </label>

      <div className="flex flex-wrap items-center gap-2">
        <span className="text-[10px] font-semibold uppercase tracking-wider text-slate-500">
          Rigor
        </span>
        {RIGORS.map((r) => (
          <Chip
            key={r}
            label={r}
            active={rigor === r}
            onClick={() => setRigor(r)}
            disabled={streaming}
          />
        ))}
      </div>

      <div className="flex flex-wrap items-center gap-2">
        <span className="text-[10px] font-semibold uppercase tracking-wider text-slate-500">
          Domain
        </span>
        {DOMAINS.map((d) => (
          <Chip
            key={d}
            label={d}
            active={domainHint === d}
            onClick={() => setDomainHint(d)}
            disabled={streaming}
          />
        ))}
      </div>

      <div className="flex flex-wrap items-center gap-2">
        <span className="text-[10px] font-semibold uppercase tracking-wider text-slate-500">
          Coverage target
        </span>
        {[0.8, 0.9, 0.95].map((c) => (
          <Chip
            key={c}
            label={`${Math.round(c * 100)}%`}
            active={Math.abs(coverage - c) < 1e-6}
            onClick={() => setCoverage(c)}
            disabled={streaming}
          />
        ))}
      </div>

      <div className="flex items-center justify-between pt-1">
        <p className="text-[10px] text-slate-500">
          Submitted text is not retained. Hashed for caching only.
        </p>
        {streaming ? (
          <button
            type="button"
            onClick={onCancel}
            className="rounded-md bg-rose-500/20 px-4 py-1.5 text-xs font-semibold text-rose-200 ring-1 ring-inset ring-rose-400/30 hover:bg-rose-500/30"
          >
            Cancel
          </button>
        ) : (
          <button
            type="submit"
            disabled={!canSubmit}
            className="rounded-md bg-gradient-to-r from-sky-500 to-indigo-500 px-4 py-1.5 text-xs font-semibold text-white shadow-md hover:brightness-110 disabled:cursor-not-allowed disabled:opacity-40"
          >
            Verify →
          </button>
        )}
      </div>
    </form>
  );
}
