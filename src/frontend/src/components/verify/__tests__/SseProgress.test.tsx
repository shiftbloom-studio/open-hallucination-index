import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { SseProgress } from "../SseProgress";
import type { ProgressBag } from "@/lib/verify-controller";

function step(el: HTMLElement, name: string): HTMLElement | null {
  return el.querySelector(`[data-step="${name}"]`);
}

describe("SseProgress", () => {
  it("shows all pending when idle and no progress", () => {
    const { container } = render(
      <SseProgress status="idle" progress={{ routed: [] }} claimsRenderedCount={0} />,
    );
    const routing = step(container, "routing") as HTMLElement;
    expect(routing.getAttribute("data-state")).toBe("pending");
  });

  it("marks decomposition done after decomposition_complete and routing active mid-stream", () => {
    const progress: ProgressBag = {
      decomposition: { claim_count: 2, estimated_total_ms: 60000 },
      routed: ["c1"],
    };
    const { container } = render(
      <SseProgress status="streaming" progress={progress} claimsRenderedCount={0} />,
    );
    expect((step(container, "decomposition") as HTMLElement).getAttribute("data-state")).toBe(
      "done",
    );
    expect((step(container, "routing") as HTMLElement).getAttribute("data-state")).toBe("active");
    expect(screen.getByText("1/2")).toBeInTheDocument();
  });

  it("completes when status=complete", () => {
    const { container } = render(
      <SseProgress
        status="complete"
        progress={{
          decomposition: { claim_count: 2, estimated_total_ms: 0 },
          routed: ["c1", "c2"],
          nli: { claim_evidence_pairs_scored: 24, claim_pair_pairs_scored: 2 },
          pcg: {
            iterations: 4,
            converged: true,
            algorithm: "TRW-BP",
            internal_consistency: 0.83,
            gibbs_validated: null,
          },
          refinement: { pass: 1, claims_re_retrieved: 0, marginal_max_change: 0 },
        }}
        claimsRenderedCount={2}
      />,
    );
    expect((step(container, "document") as HTMLElement).getAttribute("data-state")).toBe("done");
  });
});
