import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { DocumentVerdictCard } from "../DocumentVerdictCard";
import golden from "@/test/fixtures/document-verdict.golden.json";
import type { DocumentVerdict } from "@/lib/ohi-types";

const doc = golden as unknown as DocumentVerdict;

describe("DocumentVerdictCard", () => {
  it("renders skeleton when verdict is null", () => {
    render(<DocumentVerdictCard verdict={null} />);
    expect(screen.getByTestId("document-verdict-skeleton")).toBeInTheDocument();
  });

  it("renders the document_score, interval, and metadata", () => {
    render(<DocumentVerdictCard verdict={doc} />);
    expect(screen.getByTestId("document-verdict")).toBeInTheDocument();
    expect(screen.getByText("0.74")).toBeInTheDocument();
    expect(screen.getByText(/interval \[0\.61, 0\.84\]/)).toBeInTheDocument();
    expect(screen.getByText(/balanced/i)).toBeInTheDocument();
    expect(screen.getByText("0.83")).toBeInTheDocument();
    expect(screen.getByText("2")).toBeInTheDocument(); // 2 claims
  });

  it("shows refinement pass count when > 0", () => {
    render(<DocumentVerdictCard verdict={doc} />);
    expect(screen.getByText(/1 pass/)).toBeInTheDocument();
  });
});
