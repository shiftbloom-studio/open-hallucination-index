import { z } from "zod";

// ── Enums ────────────────────────────────────────────────────────
export const DomainSchema = z.enum([
  "general",
  "biomedical",
  "legal",
  "code",
  "social",
]);
export type Domain = z.infer<typeof DomainSchema>;

export const RigorSchema = z.enum(["fast", "balanced", "maximum"]);
export type Rigor = z.infer<typeof RigorSchema>;

export const TierSchema = z.enum(["local", "default", "max"]);
export type Tier = z.infer<typeof TierSchema>;

export const FallbackKindSchema = z.enum(["domain", "general", "non_converged"]);
export type FallbackKind = z.infer<typeof FallbackKindSchema>;

export const EdgeTypeSchema = z.enum(["entail", "contradict", "neutral"]);
export type EdgeType = z.infer<typeof EdgeTypeSchema>;

export const LabelerKindSchema = z.enum(["user", "expert", "adjudicator"]);
export type LabelerKind = z.infer<typeof LabelerKindSchema>;

export const FeedbackLabelSchema = z.enum(["true", "false", "unverifiable", "abstain"]);
export type FeedbackLabel = z.infer<typeof FeedbackLabelSchema>;

// ── Core models (algorithm spec §9) ──────────────────────────────
export const ClaimSchema = z.object({
  id: z.string(),
  text: z.string(),
  claim_type: z.string().nullable().optional(),
  span: z.tuple([z.number(), z.number()]).nullable().optional(),
});
export type Claim = z.infer<typeof ClaimSchema>;

export const EvidenceSchema = z.object({
  id: z.string(),
  source_uri: z.string().nullable(),
  content: z.string(),
  snippet: z.string().nullable().optional(),
  source_credibility: z.number().nullable().optional(),
  retrieved_at: z.string(),
});
export type Evidence = z.infer<typeof EvidenceSchema>;

export const ClaimEdgeSchema = z.object({
  neighbor_claim_id: z.string(),
  edge_type: EdgeTypeSchema,
  edge_strength: z.number(),
});
export type ClaimEdge = z.infer<typeof ClaimEdgeSchema>;

export const ClaimVerdictSchema = z.object({
  claim: ClaimSchema,
  p_true: z.number(),
  interval: z.tuple([z.number(), z.number()]),
  coverage_target: z.number().nullable(),
  domain: DomainSchema,
  domain_assignment_weights: z.record(z.string(), z.number()),
  supporting_evidence: z.array(EvidenceSchema),
  refuting_evidence: z.array(EvidenceSchema),
  pcg_neighbors: z.array(ClaimEdgeSchema),
  nli_self_consistency_variance: z.number(),
  bp_validated: z.boolean().nullable(),
  information_gain: z.number(),
  queued_for_review: z.boolean(),
  calibration_set_id: z.string().nullable(),
  calibration_n: z.number(),
  fallback_used: FallbackKindSchema.nullable(),
});
export type ClaimVerdict = z.infer<typeof ClaimVerdictSchema>;

export const DocumentVerdictSchema = z.object({
  request_id: z.string(),
  pipeline_version: z.string(),
  model_versions: z.record(z.string(), z.string()),
  document_score: z.number(),
  document_interval: z.tuple([z.number(), z.number()]),
  internal_consistency: z.number(),
  decomposition_coverage: z.number(),
  processing_time_ms: z.number(),
  rigor: RigorSchema,
  refinement_passes_executed: z.number(),
  claims: z.array(ClaimVerdictSchema),
});
export type DocumentVerdict = z.infer<typeof DocumentVerdictSchema>;

// ── Requests ──────────────────────────────────────────────────────
export const VerifyOptionsSchema = z.object({
  rigor: RigorSchema.optional(),
  tier: TierSchema.optional(),
  max_claims: z.number().optional(),
  include_pcg_neighbors: z.boolean().optional(),
  include_full_provenance: z.boolean().optional(),
  self_consistency_k: z.number().nullable().optional(),
  coverage_target: z.number().optional(),
});
export type VerifyOptions = z.infer<typeof VerifyOptionsSchema>;

export const VerifyRequestSchema = z.object({
  text: z.string(),
  context: z.string().nullable().optional(),
  domain_hint: DomainSchema.nullable().optional(),
  options: VerifyOptionsSchema.optional(),
  request_id: z.string().nullable().optional(),
});
export type VerifyRequest = z.infer<typeof VerifyRequestSchema>;

export const FeedbackRequestSchema = z.object({
  request_id: z.string(),
  claim_id: z.string(),
  label: FeedbackLabelSchema,
  labeler: z.object({
    kind: LabelerKindSchema,
    id: z.string(),
    credential_level: z.number(),
  }),
  rationale: z.string().max(2000).optional(),
  evidence_corrections: z
    .array(
      z.object({
        evidence_id: z.string(),
        correct_classification: z.enum(["supports", "refutes", "irrelevant"]),
      }),
    )
    .optional(),
});
export type FeedbackRequest = z.infer<typeof FeedbackRequestSchema>;

// ── Ancillary responses (spec §10) ───────────────────────────────
export interface CalibrationStratum {
  calibration_n: number;
  empirical_coverage: number;
  interval_width_p50: number;
  interval_width_p95: number;
}

export interface CalibrationDomain extends CalibrationStratum {
  strata?: Record<string, CalibrationStratum>;
}

export interface CalibrationReport {
  report_date: string;
  global_coverage_target: number;
  domains: Record<string, CalibrationDomain>;
}

export interface HealthLayer {
  status: "up" | "degraded" | "down";
  latency_p50_ms?: number;
  latency_p95_ms?: number;
  last_check: string;
  note?: string;
}

export interface HealthDeep {
  overall: "healthy" | "degraded" | "unhealthy";
  timestamp: string;
  pipeline_version: string;
  layers: Record<string, HealthLayer>;
  calibration: {
    last_updated: string;
    domains_fresh: number;
    domains_stale: number;
  };
  model_versions: Record<string, string>;
}

// ── Error payloads ────────────────────────────────────────────────
export type ApiErrorBody =
  | { status: "resting"; reason: string }
  | { status: "llm_unavailable" }
  | { detail: string; degraded_layers?: string[] }
  | { degraded_layers: string[]; detail?: string }
  | Record<string, unknown>;

// ── SSE events (spec §10 /verify/stream) ─────────────────────────
export type SseEvent =
  | {
      event: "decomposition_complete";
      data: { claim_count: number; estimated_total_ms: number };
    }
  | {
      event: "claim_routed";
      data: {
        claim_id: string;
        domain: Domain;
        weights: Partial<Record<Domain, number>>;
      };
    }
  | {
      event: "nli_complete";
      data: { claim_evidence_pairs_scored: number; claim_pair_pairs_scored: number };
    }
  | {
      event: "pcg_propagation_complete";
      data: {
        iterations: number;
        converged: boolean;
        algorithm: string;
        internal_consistency: number;
        gibbs_validated: boolean | null;
      };
    }
  | {
      event: "refinement_pass_complete";
      data: { pass: number; claims_re_retrieved: number; marginal_max_change: number };
    }
  | { event: "claim_verdict"; data: ClaimVerdict }
  | { event: "document_verdict"; data: DocumentVerdict }
  | { event: "error"; data: ApiErrorBody };

export type SseEventName = SseEvent["event"];

export const SSE_EVENT_NAMES: readonly SseEventName[] = [
  "decomposition_complete",
  "claim_routed",
  "nli_complete",
  "pcg_propagation_complete",
  "refinement_pass_complete",
  "claim_verdict",
  "document_verdict",
  "error",
] as const;
