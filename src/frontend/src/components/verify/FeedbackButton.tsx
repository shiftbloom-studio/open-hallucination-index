"use client";

import { useState } from "react";
import type { ClaimVerdict } from "@/lib/ohi-types";
import { FeedbackSheet } from "./FeedbackSheet";

export interface FeedbackButtonProps {
  requestId: string;
  claim: ClaimVerdict;
  className?: string;
}

/**
 * Convenience wrapper: renders the flag button and owns its own open state.
 * Use directly if you want each claim card to manage its own sheet; for a
 * single shared sheet controlled by a parent, mount FeedbackSheet directly.
 */
export function FeedbackButton({ requestId, claim, className }: FeedbackButtonProps) {
  const [open, setOpen] = useState(false);
  return (
    <>
      <button
        type="button"
        onClick={() => setOpen(true)}
        className={className ?? "text-slate-500 hover:text-amber-300"}
        aria-label="Flag this claim for review"
      >
        🚩 Flag
      </button>
      <FeedbackSheet
        requestId={requestId}
        claim={claim}
        open={open}
        onClose={() => setOpen(false)}
      />
    </>
  );
}
