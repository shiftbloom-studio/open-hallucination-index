import { describe, it, expect } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { ClaimList } from "../ClaimList";
import type { ClaimVerdict, DocumentVerdict } from "@/lib/ohi-types";
import golden from "@/test/fixtures/document-verdict.golden.json";

const doc = golden as unknown as DocumentVerdict;
const claims = doc.claims as ClaimVerdict[];

describe("ClaimList", () => {
  it("renders empty message when no claims", () => {
    render(<ClaimList claims={[]} emptyMessage="Nothing yet" />);
    expect(screen.getByText("Nothing yet")).toBeInTheDocument();
  });

  it("renders one ClaimCard per claim", () => {
    render(<ClaimList claims={claims} />);
    expect(screen.getAllByTestId("claim-card")).toHaveLength(2);
  });

  it("reorders by p_true ascending when sort changes", () => {
    render(<ClaimList claims={claims} />);
    const cardsBefore = screen.getAllByTestId("claim-card");
    // emitted order: c1 (0.96) then c2 (0.62)
    expect(cardsBefore[0].getAttribute("data-claim-id")).toContain("c1");

    fireEvent.change(screen.getByLabelText(/sort claims/i), {
      target: { value: "p_true_asc" },
    });
    const cardsAfter = screen.getAllByTestId("claim-card");
    expect(cardsAfter[0].getAttribute("data-claim-id")).toContain("c2");
  });
});
