import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { ClaimCard } from "../ClaimCard";
import type { ClaimVerdict, DocumentVerdict } from "@/lib/ohi-types";
import golden from "@/test/fixtures/document-verdict.golden.json";

const doc = golden as unknown as DocumentVerdict;
const [c1, c2] = doc.claims as [ClaimVerdict, ClaimVerdict];

describe("ClaimCard", () => {
  it("renders p_true, interval, claim text, and domain badge", () => {
    render(<ClaimCard verdict={c1} />);
    expect(screen.getByText("0.96")).toBeInTheDocument();
    expect(screen.getByText(/\[0\.91, 0\.99\]/)).toBeInTheDocument();
    expect(screen.getByText(/Einstein was born in 1879/)).toBeInTheDocument();
    expect(screen.getByText("general")).toBeInTheDocument();
  });

  it("shows FallbackBadge when fallback_used is set", () => {
    render(<ClaimCard verdict={c2} />);
    expect(screen.getByText(/general fallback/)).toBeInTheDocument();
  });

  it("shows 'review queued' chip when queued_for_review is true", () => {
    render(<ClaimCard verdict={c2} />);
    expect(screen.getByText(/review queued/i)).toBeInTheDocument();
  });

  it("'Expand evidence' toggles EvidenceDrawer", () => {
    render(<ClaimCard verdict={c1} />);
    expect(screen.queryByTestId("evidence-drawer")).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /expand evidence/i }));
    expect(screen.getByTestId("evidence-drawer")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /hide evidence/i }));
    expect(screen.queryByTestId("evidence-drawer")).not.toBeInTheDocument();
  });

  it("emits onShowInGraph(claim.id) when 'Show in graph' clicked", () => {
    const onShow = vi.fn();
    render(<ClaimCard verdict={c1} onShowInGraph={onShow} />);
    fireEvent.click(screen.getByRole("button", { name: /show in graph/i }));
    expect(onShow).toHaveBeenCalledWith(c1.claim.id);
  });

  it("emits onFlag(verdict) when flag clicked", () => {
    const onFlag = vi.fn();
    render(<ClaimCard verdict={c1} onFlag={onFlag} />);
    fireEvent.click(screen.getByRole("button", { name: /flag this claim/i }));
    expect(onFlag).toHaveBeenCalledWith(c1);
  });

  it("shows '@ 90%' coverage target when set", () => {
    render(<ClaimCard verdict={c1} />);
    expect(screen.getByText(/@ 90%/)).toBeInTheDocument();
  });

  it("does not render 'Show in graph' when no neighbors or handler", () => {
    const noNeighbors: ClaimVerdict = { ...c1, pcg_neighbors: [] };
    const { queryByRole } = render(
      <ClaimCard verdict={noNeighbors} onShowInGraph={() => {}} />,
    );
    expect(queryByRole("button", { name: /show in graph/i })).not.toBeInTheDocument();
  });
});
