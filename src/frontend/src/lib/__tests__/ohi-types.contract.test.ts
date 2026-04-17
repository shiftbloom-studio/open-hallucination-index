import { describe, it, expect } from "vitest";
import golden from "../../test/fixtures/document-verdict.golden.json";
import { DocumentVerdictSchema } from "../ohi-types";

describe("ohi-types contract", () => {
  it("DocumentVerdictSchema accepts the spec §10 golden fixture", () => {
    const parsed = DocumentVerdictSchema.safeParse(golden);
    if (!parsed.success) {
      throw new Error(
        `Golden fixture invalid:\n${JSON.stringify(parsed.error.issues, null, 2)}`,
      );
    }
    expect(parsed.success).toBe(true);
  });

  it("rejects payloads missing document_score", () => {
    const { document_score: _drop, ...rest } = golden as Record<string, unknown>;
    const parsed = DocumentVerdictSchema.safeParse(rest);
    expect(parsed.success).toBe(false);
  });

  it("accepts additive fields (forward compatibility per spec §10 versioning)", () => {
    const augmented = { ...golden, future_field: "new in v2.1" };
    const parsed = DocumentVerdictSchema.safeParse(augmented);
    expect(parsed.success).toBe(true);
  });
});
