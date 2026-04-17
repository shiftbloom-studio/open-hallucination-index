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
          ? "border-[color:var(--brand-indigo)] bg-[color:var(--brand-indigo-soft)] text-[color:var(--brand-indigo-strong)]"
          : "border-[color:var(--border-subtle)] bg-surface-elevated text-brand-muted hover:border-[color:var(--border-default)] hover:text-brand-ink",
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
        "space-y-3 rounded-xl border border-[color:var(--border-subtle)] bg-surface-elevated p-4 shadow-sm",
        className,
      )}
      aria-label="Verify text"
    >
      <label className="block">
        <span className="label-mono mb-1.5 block">Input text</span>
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          rows={8}
          disabled={streaming}
          maxLength={MAX_CHARS + 1000}
          placeholder={'Paste an AI-generated paragraph here. e.g. "In 1905 Einstein published four papers…"'}
          className="w-full resize-y rounded-md border border-[color:var(--border-default)] bg-surface-base p-2.5 font-mono text-xs leading-relaxed text-brand-ink focus:border-[color:var(--brand-indigo)] focus:outline-none focus:ring-2 focus:ring-[color:var(--brand-indigo)]/25 disabled:opacity-60"
          aria-invalid={overLimit}
          aria-describedby="char-counter"
        />
        <div
          id="char-counter"
          className={cn(
            "mt-1 flex justify-between text-[10px]",
            overLimit ? "text-[color:var(--brand-danger)]" : "text-brand-subtle",
          )}
        >
          <span>{overLimit && "Too long. Max 50,000 characters."}</span>
          <span className="num-mono">
            {chars.toLocaleString()} / {MAX_CHARS.toLocaleString()}
          </span>
        </div>
      </label>

      <div className="flex flex-wrap items-center gap-2">
        <span className="label-mono">Rigor</span>
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
        <span className="label-mono">Domain</span>
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
        <span className="label-mono">Coverage target</span>
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
        <p className="text-[10px] text-brand-subtle">
          Submitted text is not retained. Hashed for caching only.
        </p>
        {streaming ? (
          <button
            type="button"
            onClick={onCancel}
            className="rounded-md border border-[color:var(--brand-danger)]/40 bg-[color:var(--brand-danger-soft)] px-4 py-1.5 text-xs font-semibold text-[color:var(--brand-danger)] hover:bg-[color:var(--brand-danger-soft)]/80"
          >
            Cancel
          </button>
        ) : (
          <button
            type="submit"
            disabled={!canSubmit}
            className="rounded-md bg-[color:var(--brand-indigo)] px-4 py-1.5 text-xs font-semibold text-white shadow-sm transition-colors hover:bg-[color:var(--brand-indigo-strong)] disabled:cursor-not-allowed disabled:opacity-40"
          >
            Verify
            <span className="ml-1" aria-hidden>→</span>
          </button>
        )}
      </div>
    </form>
  );
}
