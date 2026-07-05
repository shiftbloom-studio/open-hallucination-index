"use client";

import Script from "next/script";
import { useEffect } from "react";
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
        "rounded-full border px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.12em] transition-colors",
        active
          ? "border-[color:var(--brand-primary)] bg-[color:var(--brand-secondary)] text-[color:var(--brand-accent)]"
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
  const [turnstileToken, setTurnstileToken] = useState("");
  const turnstileSiteKey = process.env.NEXT_PUBLIC_TURNSTILE_SITE_KEY;
  const turnstileRequired = Boolean(turnstileSiteKey);

  const chars = text.length;
  const overLimit = chars > MAX_CHARS;
  const canSubmit = !streaming && text.trim().length > 0 && !overLimit && (!turnstileRequired || turnstileToken.length > 0);

  useEffect(() => {
    const target = window as typeof window & {
      __ohiTurnstileSuccess?: (token: string) => void;
      __ohiTurnstileExpired?: () => void;
      __ohiTurnstileError?: () => void;
    };
    target.__ohiTurnstileSuccess = (token: string) => setTurnstileToken(token);
    target.__ohiTurnstileExpired = () => setTurnstileToken("");
    target.__ohiTurnstileError = () => setTurnstileToken("");
    return () => {
      delete target.__ohiTurnstileSuccess;
      delete target.__ohiTurnstileExpired;
      delete target.__ohiTurnstileError;
    };
  }, []);

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
    if (turnstileToken) {
      req.turnstile_token = turnstileToken;
    }
    onSubmit(req);
  }

  return (
    <form
      onSubmit={handleSubmit}
      className={cn(
        "sb-panel space-y-5 p-5 md:p-6",
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
          placeholder={'Paste an AI-generated paragraph here. e.g. "In 1905 Einstein published four papers..."'}
          className="min-h-[260px] w-full resize-y rounded-md border border-[color:var(--border-default)] bg-surface-base p-3 font-mono text-sm leading-relaxed text-brand-ink focus:border-[color:var(--brand-primary)] focus:outline-none focus:ring-2 focus:ring-[color:var(--brand-primary)]/20 disabled:opacity-60"
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

      {turnstileSiteKey && (
        <>
          <Script src="https://challenges.cloudflare.com/turnstile/v0/api.js" async defer />
          <div
            className="cf-turnstile min-h-16"
            data-sitekey={turnstileSiteKey}
            data-action="verify"
            data-callback="__ohiTurnstileSuccess"
            data-expired-callback="__ohiTurnstileExpired"
            data-error-callback="__ohiTurnstileError"
          />
        </>
      )}

      <div className="flex flex-col gap-4 border-t border-[color:var(--border-subtle)] pt-5 sm:flex-row sm:items-center sm:justify-between">
        <p className="text-[10px] text-brand-subtle">
          Submitted text is not retained. Hashed for caching only.
        </p>
        {streaming ? (
          <button
            type="button"
            onClick={onCancel}
            className="tertiary-btn min-h-0 px-5 py-2 text-sm text-[color:var(--brand-danger)]"
          >
            Cancel
          </button>
        ) : (
          <button
            type="submit"
            disabled={!canSubmit}
            className="modern-btn min-h-0 px-6 py-2 text-sm disabled:cursor-not-allowed disabled:opacity-40"
          >
            Verify
          </button>
        )}
      </div>
    </form>
  );
}
