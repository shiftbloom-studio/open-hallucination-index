import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { EvidenceDrawer } from "../EvidenceDrawer";
import type { Evidence } from "@/lib/ohi-types";

const supportingEvidence: Evidence = {
  id: "support-1",
  source_uri: "https://en.wikipedia.org/wiki/Albert_Einstein",
  content: "Albert Einstein was born in Ulm, Germany, in 1879.",
  source_credibility: 0.82,
  similarity_score: 0.91,
  classification_confidence: 0.94,
  retrieved_at: "2026-04-19T08:00:00Z",
  structured_data: {
    title: "Albert Einstein",
    nli_reasoning: "The evidence gives Germany as the birthplace, which contradicts the claim.",
    bucket_score: 0.94,
  },
};

const refutingEvidence: Evidence = {
  id: "refute-1",
  source_uri: "https://www.wikidata.org/wiki/Q937",
  content: "country of citizenship: Germany",
  source_credibility: 0.88,
  similarity_score: 0.73,
  classification_confidence: 0.81,
  retrieved_at: "2026-04-19T08:00:01Z",
  structured_data: {
    title: "Q937",
    nli_reasoning: "The structured fact conflicts with the claimed country.",
    bucket_score: 0.81,
  },
};

describe("EvidenceDrawer", () => {
  it("renders title, reasoning, and evidence scores for both buckets", () => {
    render(
      <EvidenceDrawer
        open
        supporting={[supportingEvidence]}
        refuting={[refutingEvidence]}
      />,
    );

    expect(screen.getByText("Albert Einstein")).toBeInTheDocument();
    expect(
      screen.getByText(/Germany as the birthplace, which contradicts the claim/i),
    ).toBeInTheDocument();
    expect(screen.getByText("support 0.94")).toBeInTheDocument();
    expect(screen.getByText("rel 0.91")).toBeInTheDocument();
    expect(screen.getByText("cred 0.82")).toBeInTheDocument();

    expect(screen.getByText("Q937")).toBeInTheDocument();
    expect(screen.getByText(/structured fact conflicts/i)).toBeInTheDocument();
    expect(screen.getByText("refute 0.81")).toBeInTheDocument();
  });

  it("returns null when closed", () => {
    const { container } = render(
      <EvidenceDrawer open={false} supporting={[supportingEvidence]} refuting={[]} />,
    );
    expect(container).toBeEmptyDOMElement();
  });
});
