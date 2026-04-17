import { describe, it, expect } from "vitest";
import { buildGraphData } from "../PcgGraph";
import type { ClaimVerdict, DocumentVerdict } from "@/lib/ohi-types";
import golden from "@/test/fixtures/document-verdict.golden.json";

const doc = golden as unknown as DocumentVerdict;
const claims = doc.claims as ClaimVerdict[];

describe("PcgGraph buildGraphData", () => {
  it("produces one node per claim", () => {
    const data = buildGraphData(claims, true);
    expect(data.nodes).toHaveLength(2);
    expect(data.nodes.map((n) => n.id).sort()).toEqual(
      [claims[0].claim.id, claims[1].claim.id].sort(),
    );
  });

  it("dedupes mutual edges (c1↔c2 same edge_type) to a single link", () => {
    const data = buildGraphData(claims, true);
    expect(data.links).toHaveLength(1);
    expect(data.links[0].edgeType).toBe("entail");
  });

  it("drops neutral edges when hideNeutral is true", () => {
    const withNeutral: ClaimVerdict[] = [
      { ...claims[0], pcg_neighbors: [{ neighbor_claim_id: claims[1].claim.id, edge_type: "neutral", edge_strength: 0.1 }] },
      { ...claims[1], pcg_neighbors: [] },
    ];
    expect(buildGraphData(withNeutral, true).links).toHaveLength(0);
    expect(buildGraphData(withNeutral, false).links).toHaveLength(1);
  });

  it("keeps distinct edge_types between the same pair", () => {
    const dualType: ClaimVerdict[] = [
      {
        ...claims[0],
        pcg_neighbors: [
          { neighbor_claim_id: claims[1].claim.id, edge_type: "entail", edge_strength: 0.4 },
          { neighbor_claim_id: claims[1].claim.id, edge_type: "contradict", edge_strength: 0.3 },
        ],
      },
      { ...claims[1], pcg_neighbors: [] },
    ];
    expect(buildGraphData(dualType, false).links).toHaveLength(2);
  });
});
