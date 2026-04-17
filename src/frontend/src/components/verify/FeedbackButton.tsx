"use client";

import { useState } from "react";
import type { ClaimVerdict } from "@/lib/ohi-types";
import { FeedbackSheet } from "./FeedbackSheet";

export interface FeedbackButtonProps {
  requestId: string;
  claim: ClaimVerdict;
  className?: string;
}

export function FeedbackButton({ requestId, claim, className }: FeedbackButtonProps) {
  const [open, setOpen] = useState(false);
  return (
    <>
      <button
        type="button"
        onClick={() => setOpen(true)}
        className={className ?? "text-brand-subtle hover:text-[color:var(--brand-warning)]"}
        aria-label="Flag this claim for review"
      >
        <span aria-hidden>🚩</span> Flag
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
