"use client";

import { useEffect, useState } from "react";
import type { ClaimVerdict, FeedbackLabel, FeedbackRequest } from "@/lib/ohi-types";
import { cn } from "@/lib/utils";
import { useFeedbackMutation } from "@/lib/ohi-queries";

const LABELER_ID_KEY = "ohi:labeler-id";

function getOrCreateLabelerId(): string {
  if (typeof window === "undefined") return "ssr-placeholder";
  const existing = window.localStorage.getItem(LABELER_ID_KEY);
  if (existing) return existing;
  const next = crypto.randomUUID();
  window.localStorage.setItem(LABELER_ID_KEY, next);
  return next;
}

export interface FeedbackSheetProps {
  requestId: string;
  claim: ClaimVerdict;
  open: boolean;
  onClose: () => void;
}

const LABEL_OPTIONS: { value: FeedbackLabel; title: string; hint: string }[] = [
  { value: "true", title: "True", hint: "The claim matches the evidence and is factually correct." },
  { value: "false", title: "False", hint: "The claim contradicts the evidence." },
  { value: "unverifiable", title: "Unverifiable", hint: "The evidence is insufficient to decide." },
  { value: "abstain", title: "Abstain", hint: "I'd rather not say." },
];

export function FeedbackSheet({ requestId, claim, open, onClose }: FeedbackSheetProps) {
  const [label, setLabel] = useState<FeedbackLabel>("false");
  const [rationale, setRationale] = useState("");
  const mutation = useFeedbackMutation();

  useEffect(() => {
    if (!open) return;
    setLabel("false");
    setRationale("");
    mutation.reset();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, claim.claim.id]);

  useEffect(() => {
    if (!open) return;
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") onClose();
    }
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  if (!open) return null;

  const rationaleChars = rationale.length;
  const rationaleTooLong = rationaleChars > 2000;

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (rationaleTooLong) return;
    const payload: FeedbackRequest = {
      request_id: requestId,
      claim_id: claim.claim.id,
      label,
      labeler: { kind: "user", id: getOrCreateLabelerId(), credential_level: 0 },
      rationale: rationale || undefined,
    };
    try {
      await mutation.mutateAsync(payload);
      setTimeout(() => onClose(), 1000);
    } catch {
      /* UI shows mutation.error */
    }
  }

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="feedback-title"
      className="fixed inset-0 z-50 flex items-center justify-center p-4"
      data-testid="feedback-sheet"
    >
      <div
        className="absolute inset-0 bg-[color:var(--surface-inverse)]/40 backdrop-blur-sm"
        onClick={onClose}
        aria-hidden="true"
      />
      <form
        onSubmit={handleSubmit}
        className="relative w-full max-w-lg rounded-2xl border border-[color:var(--border-subtle)] bg-surface-elevated p-5 shadow-xl"
      >
        <header className="mb-3 flex items-start justify-between gap-4">
          <div>
            <h2 id="feedback-title" className="font-heading text-sm font-semibold text-brand-ink">
              Flag this claim
            </h2>
            <p className="mt-1 text-xs text-brand-muted">
              Your label enters the untrusted-consensus queue. 3 concordant distinct labelers
              promote it into the calibration set.
            </p>
          </div>
          <button
            type="button"
            className="text-brand-subtle hover:text-brand-ink"
            onClick={onClose}
            aria-label="Close"
          >
            ✕
          </button>
        </header>

        <blockquote className="mb-3 rounded-md border border-[color:var(--border-subtle)] bg-surface-base p-3 text-xs italic text-brand-ink">
          &ldquo;{claim.claim.text}&rdquo;
        </blockquote>

        <fieldset className="mb-3 space-y-1">
          <legend className="label-mono mb-1">Your label</legend>
          {LABEL_OPTIONS.map((opt) => (
            <label
              key={opt.value}
              className={cn(
                "flex cursor-pointer items-start gap-2 rounded-md border border-transparent p-2 hover:bg-[color:var(--surface-soft)]/60",
                label === opt.value &&
                  "border-[color:var(--brand-indigo)]/40 bg-[color:var(--brand-indigo-soft)]",
              )}
            >
              <input
                type="radio"
                name="label"
                value={opt.value}
                checked={label === opt.value}
                onChange={() => setLabel(opt.value)}
                className="mt-1 accent-[color:var(--brand-indigo)]"
              />
              <span>
                <span className="block text-sm text-brand-ink">{opt.title}</span>
                <span className="block text-[11px] text-brand-muted">{opt.hint}</span>
              </span>
            </label>
          ))}
        </fieldset>

        <label className="mb-3 block text-xs">
          <span className="label-mono mb-1 block">Rationale (optional)</span>
          <textarea
            value={rationale}
            onChange={(e) => setRationale(e.target.value)}
            rows={4}
            maxLength={2500}
            className="w-full rounded-md border border-[color:var(--border-default)] bg-surface-base p-2 text-sm text-brand-ink focus:border-[color:var(--brand-indigo)] focus:outline-none focus:ring-2 focus:ring-[color:var(--brand-indigo)]/25"
            placeholder="Cite the source that contradicts / corroborates this claim…"
          />
          <span
            className={cn(
              "mt-1 block text-right text-[10px]",
              rationaleTooLong ? "text-[color:var(--brand-danger)]" : "text-brand-subtle",
            )}
          >
            {rationaleChars} / 2000
          </span>
        </label>

        {mutation.isError && (
          <p className="mb-2 text-xs text-[color:var(--brand-danger)]">
            Submit failed. Please retry.
          </p>
        )}
        {mutation.isSuccess && (
          <p className="mb-2 text-xs text-[color:var(--brand-success)]">Thanks — queued.</p>
        )}

        <div className="flex justify-end gap-2">
          <button
            type="button"
            onClick={onClose}
            className="rounded-md px-3 py-1.5 text-xs text-brand-muted hover:text-brand-ink"
          >
            Cancel
          </button>
          <button
            type="submit"
            disabled={rationaleTooLong || mutation.isPending || mutation.isSuccess}
            className="rounded-md bg-[color:var(--brand-indigo)] px-3 py-1.5 text-xs font-semibold text-white hover:bg-[color:var(--brand-indigo-strong)] disabled:cursor-not-allowed disabled:bg-[color:var(--border-default)] disabled:text-brand-subtle"
          >
            {mutation.isPending ? "Submitting…" : "Submit"}
          </button>
        </div>
      </form>
    </div>
  );
}
