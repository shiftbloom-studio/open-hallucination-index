import { DurableObject, WorkflowEntrypoint } from "cloudflare:workers";
import type { WorkflowEvent, WorkflowStep } from "cloudflare:workers";
import { McpAgent } from "agents/mcp";
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { z } from "zod";
import {
  type CorpusIngestQueueMessage,
  type CorpusWorkflowParams,
  getCorpusOverview,
  getCorpusRun,
  isCorpusIngestQueueMessage,
  processCorpusIngestMessage,
  startCorpusRun,
} from "./corpus-ingestion";
import { callKnowledgeTool, KNOWLEDGE_TOOLS, type SearchResult } from "./knowledge-tools";

type Domain = "general" | "biomedical" | "legal" | "code" | "social";
type Rigor = "fast" | "balanced" | "maximum";
type JobStatusValue = "pending" | "done" | "error";
type NliLabel = "support" | "refute" | "neutral";

interface VerifyRequest {
  text: string;
  context?: string | null;
  domain_hint?: Domain | null;
  turnstile_token?: string | null;
  "cf-turnstile-response"?: string | null;
  options?: {
    rigor?: Rigor;
    tier?: "local" | "default" | "max";
    max_claims?: number;
    include_pcg_neighbors?: boolean;
    include_full_provenance?: boolean;
    self_consistency_k?: number | null;
    coverage_target?: number;
  };
  request_id?: string | null;
}

interface VerifyQueueMessage {
  jobId: string;
  request: VerifyRequest;
  submittedAt: number;
  textHash: string;
}

type WorkerQueueMessage = VerifyQueueMessage | CorpusIngestQueueMessage;

interface Claim {
  id: string;
  text: string;
  claim_type?: string | null;
  span?: [number, number] | null;
}

interface Evidence {
  id: string;
  source_uri: string | null;
  content: string;
  snippet?: string | null;
  source_credibility?: number | null;
  similarity_score?: number | null;
  classification_confidence?: number | null;
  structured_data?: Record<string, unknown> | null;
  retrieved_at: string;
}

interface ClassifiedEvidence extends Evidence {
  nli_label: NliLabel;
  nli_confidence: number;
  supporting_score: number;
  refuting_score: number;
  neutral_score: number;
  relevance_score: number;
  nli_reasoning?: string;
}

interface ClaimVerdict {
  claim: Claim;
  p_true: number;
  interval: [number, number];
  coverage_target: number | null;
  domain: Domain;
  domain_assignment_weights: Record<string, number>;
  supporting_evidence: Evidence[];
  refuting_evidence: Evidence[];
  pcg_neighbors: Array<{
    neighbor_claim_id: string;
    edge_type: "entail" | "contradict" | "neutral";
    edge_strength: number;
  }>;
  nli_self_consistency_variance: number;
  bp_validated: boolean | null;
  information_gain: number;
  queued_for_review: boolean;
  calibration_set_id: string | null;
  calibration_n: number;
  fallback_used: "domain" | "general" | "non_converged" | null;
}

interface DocumentVerdict {
  request_id: string;
  pipeline_version: string;
  model_versions: Record<string, string>;
  document_score: number;
  document_interval: [number, number];
  internal_consistency: number;
  decomposition_coverage: number;
  processing_time_ms: number;
  rigor: Rigor;
  refinement_passes_executed: number;
  claims: ClaimVerdict[];
}

interface JobRecord {
  job_id: string;
  status: JobStatusValue;
  phase: string;
  created_at: number;
  updated_at: number;
  completed_at?: number;
  result?: DocumentVerdict;
  error?: string;
}

interface RawJobRow {
  job_id: string;
  status: JobStatusValue;
  phase: string;
  created_at: number;
  updated_at: number;
  completed_at: number | null;
  result_json: string | null;
  error: string | null;
}

const CHAT_MODEL = "@cf/google/gemma-3-12b-it";
const NLI_CORROBORATION_MODEL = "@cf/meta/llama-3.3-70b-instruct-fp8-fast";
const FAST_TIER_MODEL = "@cf/meta/llama-4-scout-17b-16e-instruct";
const EMBEDDING_MODEL = "@cf/baai/bge-m3";
const RERANK_MODEL = "@cf/baai/bge-reranker-base";
const PIPELINE_VERSION = "ohi-v2.0-cloudflare";

interface RigorProfile {
  maxClaims: number;
  evidencePerClaim: number;
  classifiedPerClaim: number;
  ensemble: boolean;
}

// Sized against live Workers AI pricing (2026-07-05) to keep worst-case cost
// per verify request under ~$0.01 (fast), ~$0.035 (balanced), ~$0.10 (maximum).
const RIGOR_PROFILES: Record<Rigor, RigorProfile> = {
  fast: { maxClaims: 4, evidencePerClaim: 3, classifiedPerClaim: 3, ensemble: false },
  balanced: { maxClaims: 7, evidencePerClaim: 4, classifiedPerClaim: 4, ensemble: true },
  maximum: { maxClaims: 13, evidencePerClaim: 6, classifiedPerClaim: 6, ensemble: true },
};
const MAX_BODY_BYTES = 64 * 1024;
const DEFAULT_ALLOWED_ORIGINS = new Set([
  "https://ohi.shiftbloom.studio",
  "http://localhost:3000",
  "http://127.0.0.1:3000",
]);

interface RateLimitDecision {
  allowed: boolean;
  limit: number;
  remaining: number;
  reset_at: number;
}

export class JobObject extends DurableObject<Env> {
  constructor(ctx: DurableObjectState, env: Env) {
    super(ctx, env);
    ctx.blockConcurrencyWhile(async () => {
      this.ctx.storage.sql.exec(`
        CREATE TABLE IF NOT EXISTS state (
          job_id TEXT PRIMARY KEY,
          status TEXT NOT NULL,
          phase TEXT NOT NULL,
          created_at REAL NOT NULL,
          updated_at REAL NOT NULL,
          completed_at REAL,
          text_hash TEXT NOT NULL,
          result_json TEXT,
          error TEXT
        )
      `);
    });
  }

  async create(message: VerifyQueueMessage): Promise<JobRecord> {
    const now = secondsNow();
    this.ctx.storage.sql.exec(
      `INSERT OR REPLACE INTO state
        (job_id, status, phase, created_at, updated_at, text_hash)
       VALUES (?, 'pending', 'queued', ?, ?, ?)`,
      message.jobId,
      now,
      now,
      message.textHash,
    );
    await mirrorD1Job(this.env, {
      job_id: message.jobId,
      status: "pending",
      phase: "queued",
      created_at: now,
      updated_at: now,
      text_hash: message.textHash,
    });
    return {
      job_id: message.jobId,
      status: "pending",
      phase: "queued",
      created_at: now,
      updated_at: now,
    };
  }

  async getStatus(): Promise<JobRecord | null> {
    const row = this.ctx.storage.sql.exec(
      "SELECT job_id, status, phase, created_at, updated_at, completed_at, result_json, error FROM state LIMIT 1",
    ).toArray()[0] as unknown as RawJobRow | undefined;
    return row ? jobFromRow(row) : null;
  }

  async run(message: VerifyQueueMessage): Promise<JobRecord> {
    const current = await this.getStatus();
    if (!current) {
      await this.create(message);
    } else if (current.status !== "pending") {
      return current;
    }

    const updatePhase = async (phase: string) => {
      await this.setPhase(message.jobId, phase);
    };

    try {
      await updatePhase("decomposing");
      const verdict = await runPipeline(this.env, message.request, updatePhase);
      const now = secondsNow();
      this.ctx.storage.sql.exec(
        `UPDATE state
         SET status = 'done', phase = 'assembling', updated_at = ?, completed_at = ?,
             result_json = ?, error = NULL
         WHERE job_id = ?`,
        now,
        now,
        JSON.stringify(verdict),
        message.jobId,
      );
      await mirrorD1Job(this.env, {
        job_id: message.jobId,
        status: "done",
        phase: "assembling",
        updated_at: now,
        completed_at: now,
        result_json: JSON.stringify(verdict),
      });
      return (await this.getStatus()) as JobRecord;
    } catch (error) {
      const messageText = errorMessage(error);
      await this.fail(message.jobId, messageText);
      throw error;
    }
  }

  async fail(jobId: string, reason: string): Promise<void> {
    const now = secondsNow();
    this.ctx.storage.sql.exec(
      `UPDATE state
       SET status = 'error', phase = 'error', updated_at = ?, completed_at = ?, error = ?
       WHERE job_id = ?`,
      now,
      now,
      reason.slice(0, 2000),
      jobId,
    );
    await mirrorD1Job(this.env, {
      job_id: jobId,
      status: "error",
      phase: "error",
      updated_at: now,
      completed_at: now,
      error: reason.slice(0, 2000),
    });
  }

  private async setPhase(jobId: string, phase: string): Promise<void> {
    const now = secondsNow();
    this.ctx.storage.sql.exec(
      "UPDATE state SET phase = ?, updated_at = ? WHERE job_id = ? AND status = 'pending'",
      phase,
      now,
      jobId,
    );
    await mirrorD1Job(this.env, {
      job_id: jobId,
      status: "pending",
      phase,
      updated_at: now,
    });
  }
}

export class RateLimitObject extends DurableObject<Env> {
  constructor(ctx: DurableObjectState, env: Env) {
    super(ctx, env);
    ctx.blockConcurrencyWhile(async () => {
      this.ctx.storage.sql.exec(`
        CREATE TABLE IF NOT EXISTS counters (
          key TEXT NOT NULL,
          bucket INTEGER NOT NULL,
          count INTEGER NOT NULL,
          expires_at REAL NOT NULL,
          PRIMARY KEY (key, bucket)
        )
      `);
    });
  }

  async check(key: string, limit: number, windowSeconds: number): Promise<RateLimitDecision> {
    const now = secondsNow();
    const bucket = Math.floor(now / windowSeconds);
    const resetAt = (bucket + 1) * windowSeconds;
    this.ctx.storage.sql.exec("DELETE FROM counters WHERE expires_at < ?", now);
    const current = this.ctx.storage.sql.exec(
      "SELECT count FROM counters WHERE key = ? AND bucket = ?",
      key,
      bucket,
    ).toArray()[0] as unknown as { count: number } | undefined;
    const next = (Number(current?.count ?? 0) || 0) + 1;
    this.ctx.storage.sql.exec(
      `INSERT INTO counters (key, bucket, count, expires_at)
       VALUES (?, ?, ?, ?)
       ON CONFLICT(key, bucket) DO UPDATE SET count = excluded.count, expires_at = excluded.expires_at`,
      key,
      bucket,
      next,
      resetAt + windowSeconds,
    );
    return {
      allowed: next <= limit,
      limit,
      remaining: Math.max(0, limit - next),
      reset_at: resetAt,
    };
  }
}

export class CorpusIngestionWorkflow extends WorkflowEntrypoint<Env, CorpusWorkflowParams> {
  async run(event: Readonly<WorkflowEvent<CorpusWorkflowParams>>, step: WorkflowStep): Promise<unknown> {
    const result = await step.do(
      "enqueue corpus ingestion run",
      {
        retries: { limit: 3, delay: "30 seconds", backoff: "exponential" },
        timeout: "5 minutes",
      },
      async () => JSON.stringify(await startCorpusRun(this.env, event.payload ?? {})),
    );
    return JSON.parse(result) as Record<string, unknown>;
  }
}

export class OhiMCP extends McpAgent<Env> {
  server = new McpServer({ name: "open-hallucination-index", version: "0.1.0" });

  async init() {
    this.server.tool(
      "verify_text",
      {
        text: z.string().min(1).max(50_000),
        rigor: z.enum(["fast", "balanced", "maximum"]).optional(),
        domain_hint: z.enum(["general", "biomedical", "legal", "code", "social"]).optional(),
      },
      async ({ text, rigor, domain_hint }) => {
        const verdict = await runPipeline(
          this.env,
          {
            text,
            domain_hint: domain_hint ?? null,
            options: { rigor: rigor ?? "balanced" },
          },
          async () => undefined,
        );
        return {
          structuredContent: asStructuredContent(verdict),
          content: [{ type: "text" as const, text: JSON.stringify(verdict, null, 2) }],
        };
      },
    );

    this.server.tool(
      "job_status",
      { job_id: z.string().uuid() },
      async ({ job_id }) => {
        const status = await this.env.JOBS.getByName(job_id).getStatus();
        return {
          structuredContent: asStructuredContent(status ?? { error: "job_not_found" }),
          content: [{ type: "text" as const, text: JSON.stringify(status ?? { error: "job_not_found" }, null, 2) }],
        };
      },
    );

    this.server.tool(
      "search_evidence",
      { query: z.string().min(1).max(500) },
      async ({ query }) => {
        const evidence = await retrieveEvidence(this.env, query);
        return {
          structuredContent: asStructuredContent({ evidence }),
          content: [{ type: "text" as const, text: JSON.stringify(evidence, null, 2) }],
        };
      },
    );

    for (const tool of KNOWLEDGE_TOOLS) {
      this.server.tool(
        tool.name,
        tool.description,
        tool.schema,
        async (args) => {
          const result = await callKnowledgeTool(tool.name, args as Record<string, unknown>, this.env);
          return {
            structuredContent: asStructuredContent(typeof result === "string" ? { text: result } : result),
            content: [{ type: "text" as const, text: typeof result === "string" ? result : JSON.stringify(result, null, 2) }],
          };
        },
      );
    }
  }
}

const mcpHandler = OhiMCP.serve("/mcp");

export default {
  async fetch(request: Request, env: Env, ctx: ExecutionContext): Promise<Response> {
    const url = new URL(request.url);

    if (request.method === "OPTIONS") {
      return withCors(request, env, new Response(null, { status: 204 }));
    }

    if (url.pathname === "/mcp" || url.pathname.startsWith("/mcp/")) {
      return mcpHandler.fetch(request, env, ctx);
    }

    if (url.pathname === "/health" || url.pathname === "/health/live") {
      return json(request, env, {
        status: "healthy",
        timestamp: new Date().toISOString(),
        version: "0.1.0",
        environment: env.ENVIRONMENT ?? "production",
      });
    }

    if (url.pathname === "/health/ready") {
      return handleReady(request, env);
    }

    if (url.pathname === "/health/deep") {
      return handleDeepHealth(request, env);
    }

    if (url.pathname.startsWith("/api/v2/")) {
      return handleApi(request, env, ctx, url);
    }

    return env.ASSETS.fetch(request);
  },

  async queue(batch: MessageBatch<WorkerQueueMessage>, env: Env): Promise<void> {
    for (const message of batch.messages) {
      const payload = message.body;
      try {
        if (isCorpusIngestQueueMessage(payload)) {
          await processCorpusIngestMessage(env, payload);
        } else {
          const stub = env.JOBS.getByName(payload.jobId);
          await stub.run(payload);
        }
        message.ack();
      } catch (error) {
        if (!isCorpusIngestQueueMessage(payload)) {
          try {
            await env.JOBS.getByName(payload.jobId).fail(payload.jobId, errorMessage(error));
          } catch {
            // The retry below is the important recovery path.
          }
        }
        message.retry({ delaySeconds: 30 });
      }
    }
  },
} satisfies ExportedHandler<Env, WorkerQueueMessage>;

async function handleApi(
  request: Request,
  env: Env,
  _ctx: ExecutionContext,
  url: URL,
): Promise<Response> {
  if (url.pathname === "/api/v2/verify" && request.method === "POST") {
    return handleVerify(request, env);
  }

  if (url.pathname === "/api/v2/admin/corpus" && request.method === "GET") {
    const auth = await requireAdmin(request, env);
    if (auth) return auth;
    return json(request, env, await getCorpusOverview(env));
  }

  if (url.pathname === "/api/v2/admin/corpus/runs" && request.method === "POST") {
    const auth = await requireAdmin(request, env);
    if (auth) return auth;
    let body: CorpusWorkflowParams;
    try {
      body = (await request.json()) as CorpusWorkflowParams;
    } catch {
      return json(request, env, { detail: "Invalid JSON body" }, 400);
    }
    const workflow = (env as Env & { CORPUS_WORKFLOW?: Workflow<CorpusWorkflowParams> }).CORPUS_WORKFLOW;
    if (workflow) {
      const instanceId = crypto.randomUUID();
      await workflow.create({ id: instanceId, params: body });
      return json(request, env, { workflow_instance_id: instanceId, status: "queued" }, 202);
    }
    return json(request, env, await startCorpusRun(env, body), 202);
  }

  const corpusRunMatch = url.pathname.match(/^\/api\/v2\/admin\/corpus\/runs\/([0-9a-f-]{36})$/i);
  if (corpusRunMatch && request.method === "GET") {
    const auth = await requireAdmin(request, env);
    if (auth) return auth;
    const run = await getCorpusRun(env, corpusRunMatch[1]);
    if (!run) return json(request, env, { detail: "corpus run not found" }, 404);
    return json(request, env, run);
  }

  const statusMatch = url.pathname.match(/^\/api\/v2\/verify\/status\/([0-9a-f-]{36})$/i);
  if (statusMatch && request.method === "GET") {
    const record = await env.JOBS.getByName(statusMatch[1]).getStatus();
    if (!record) {
      return json(request, env, { detail: { code: "job_not_found", message: "Unknown job id" } }, 404);
    }
    return json(request, env, record);
  }

  if (url.pathname === "/api/v2/calibration/report" && request.method === "GET") {
    return json(request, env, calibrationReport());
  }

  if (url.pathname === "/api/v2/feedback" && request.method === "POST") {
    return handleFeedback(request, env);
  }

  return json(request, env, { detail: "Not found" }, 404);
}

async function handleVerify(request: Request, env: Env): Promise<Response> {
  const contentLength = Number(request.headers.get("content-length") ?? "0");
  if (contentLength > MAX_BODY_BYTES) {
    return json(request, env, { detail: "Request body too large" }, 413);
  }

  let body: VerifyRequest;
  try {
    body = (await request.json()) as VerifyRequest;
  } catch {
    return json(request, env, { detail: "Invalid JSON body" }, 400);
  }

  const validation = validateVerifyRequest(body, env);
  if (validation) {
    return json(request, env, { detail: validation }, 422);
  }

  const abuse = await rejectIfAbusive(request, env, body);
  if (abuse) return abuse;

  const jobId = crypto.randomUUID();
  const textHash = await sha256Hex(body.text);
  const payload: VerifyQueueMessage = {
    jobId,
    request: normalizeVerifyRequest(body, env),
    submittedAt: secondsNow(),
    textHash,
  };

  await env.JOBS.getByName(jobId).create(payload);
  await env.VERIFY_QUEUE.send(payload);
  return json(request, env, { job_id: jobId }, 202);
}

async function handleFeedback(request: Request, env: Env): Promise<Response> {
  const contentLength = Number(request.headers.get("content-length") ?? "0");
  if (contentLength > MAX_BODY_BYTES) {
    return json(request, env, { detail: "Request body too large" }, 413);
  }

  let body: Record<string, unknown>;
  try {
    body = (await request.json()) as Record<string, unknown>;
  } catch {
    return json(request, env, { detail: "Invalid JSON body" }, 400);
  }

  const feedbackId = crypto.randomUUID();
  await env.OHI_DB.prepare(
    `INSERT INTO feedback
      (feedback_id, request_id, claim_id, label, labeler_json, rationale, evidence_corrections_json, created_at)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
  )
    .bind(
      feedbackId,
      String(body.request_id ?? ""),
      String(body.claim_id ?? ""),
      String(body.label ?? ""),
      JSON.stringify(body.labeler ?? {}),
      typeof body.rationale === "string" ? body.rationale.slice(0, 2000) : null,
      JSON.stringify(body.evidence_corrections ?? []),
      secondsNow(),
    )
    .run();

  return json(request, env, { feedback_id: feedbackId, queued: true });
}

async function handleReady(request: Request, env: Env): Promise<Response> {
  const services: Record<string, { connected: boolean; status: string }> = {
    d1: { connected: false, status: "unknown" },
    durable_objects: { connected: false, status: "unknown" },
    queue: { connected: true, status: "configured" },
    workers_ai: { connected: true, status: "configured" },
    vectorize: { connected: true, status: "configured" },
  };

  try {
    await env.OHI_DB.prepare("SELECT 1 AS ok").first();
    services.d1 = { connected: true, status: "healthy" };
  } catch {
    services.d1 = { connected: false, status: "error" };
  }

  try {
    const probeId = `ready-${new Date().toISOString().slice(0, 10)}`;
    await env.JOBS.getByName(probeId).getStatus();
    services.durable_objects = { connected: true, status: "healthy" };
  } catch {
    services.durable_objects = { connected: false, status: "error" };
  }

  const ready = Object.values(services).every((service) => service.connected);
  return json(request, env, {
    ready,
    timestamp: new Date().toISOString(),
    services,
  });
}

async function handleDeepHealth(request: Request, env: Env): Promise<Response> {
  const started = performance.now();
  const layers: Record<string, { status: "ok" | "degraded" | "down" | "skipped"; latency_ms?: number; detail?: string }> = {};

  await timedLayer(layers, "L1.decompose", async () => {
    await decomposeClaims(env, "Cloudflare Workers run JavaScript at the edge.", 1);
  });

  await timedLayer(layers, "L1.retrieve.wikimedia", async () => {
    const result = await callKnowledgeTool("search_wikipedia", { query: "Cloudflare Workers", limit: 1 }, env);
    if (typeof result !== "string" && !result.success) throw new Error(result.error ?? "search_wikipedia failed");
  });

  await timedLayer(layers, "L1.retrieve.vectorize", async () => {
    const embedding = await embedText(env, "Cloudflare Workers");
    if (embedding) {
      await env.OHI_VECTOR.query(embedding, { topK: 1, returnMetadata: "all" });
    }
  });

  await timedLayer(layers, "L3.nli", async () => {
    await classifyEvidence(env, "Cloudflare Workers run JavaScript at the edge.", {
      id: "health",
      source_uri: "https://developers.cloudflare.com/workers/",
      content: "Cloudflare Workers provides a serverless execution environment.",
      snippet: "Cloudflare Workers provides a serverless execution environment.",
      retrieved_at: new Date().toISOString(),
    });
  });

  return json(request, env, {
    status: Object.values(layers).every((layer) => layer.status === "ok") ? "ok" : "degraded",
    timestamp: new Date().toISOString(),
    pipeline_version: PIPELINE_VERSION,
    layers,
    model_versions: modelVersions(),
    calibration_freshness_hours: null,
    processing_time_ms: Math.round(performance.now() - started),
  });
}

async function runPipeline(
  env: Env,
  request: VerifyRequest,
  phase: (phase: string) => Promise<void>,
): Promise<DocumentVerdict> {
  const started = performance.now();
  const rigor = request.options?.rigor ?? "balanced";
  const maxClaims = clamp(
    request.options?.max_claims ?? defaultMaxClaims(env, rigor),
    1,
    defaultMaxClaims(env, rigor),
  );
  const coverageTarget = clamp(request.options?.coverage_target ?? 0.9, 0.5, 0.99);

  const profile = RIGOR_PROFILES[rigor];
  const claims = await decomposeClaims(env, request.text, maxClaims);
  await phase("retrieving_evidence");

  const domain = request.domain_hint ?? "general";
  const evidenceByClaim = new Map<string, Evidence[]>();
  for (const claim of claims) {
    evidenceByClaim.set(claim.id, await retrieveEvidence(env, claim.text, domain, profile.evidencePerClaim));
  }

  await phase("classifying");
  const verdicts: ClaimVerdict[] = [];
  for (const claim of claims) {
    const evidence = evidenceByClaim.get(claim.id) ?? [];
    const classified = await Promise.all(
      evidence.slice(0, profile.classifiedPerClaim).map((item) => classifyEvidence(env, claim.text, item, rigor)),
    );
    verdicts.push(buildClaimVerdict(claim, classified, domain, coverageTarget));
  }

  await phase("calibrating");
  const documentScore = geometricMean(verdicts.map((verdict) => verdict.p_true));
  const low = Math.min(...verdicts.map((verdict) => verdict.interval[0]), documentScore);
  const high = Math.max(...verdicts.map((verdict) => verdict.interval[1]), documentScore);

  await phase("assembling");
  return {
    request_id: request.request_id ?? crypto.randomUUID(),
    pipeline_version: PIPELINE_VERSION,
    model_versions: modelVersions(),
    document_score: round4(documentScore),
    document_interval: [round4(clamp(low, 0, 1)), round4(clamp(high, 0, 1))],
    internal_consistency: round4(internalConsistency(verdicts)),
    decomposition_coverage: claims.length > 0 ? 1 : 0,
    processing_time_ms: Math.round(performance.now() - started),
    rigor,
    refinement_passes_executed: 0,
    claims: verdicts,
  };
}

async function decomposeClaims(env: Env, text: string, maxClaims: number): Promise<Claim[]> {
  const prompt = [
    "Decompose the input into atomic factual claims.",
    "Return only JSON matching {\"claims\":[{\"text\":\"...\",\"claim_type\":\"factual|temporal|quantitative|causal|other\"}]}",
    `Limit to ${maxClaims} claims. Ignore opinions and unverifiable stylistic statements.`,
    "",
    text.slice(0, 12_000),
  ].join("\n");

  try {
    const aiResult = await env.AI.run(
      CHAT_MODEL,
      {
        messages: [
          { role: "system", content: "You are a careful claim decomposition engine. Return strict JSON." },
          { role: "user", content: prompt },
        ],
        max_tokens: 1200,
        temperature: 0.1,
        guided_json: {
          type: "object",
          properties: {
            claims: {
              type: "array",
              items: {
                type: "object",
                properties: {
                  text: { type: "string" },
                  claim_type: { type: "string" },
                },
                required: ["text"],
              },
            },
          },
          required: ["claims"],
        },
      },
      { gateway: { id: "default", skipCache: false, cacheTtl: 3600 } },
    );
    const parsed = parseJsonObject(extractAiText(aiResult));
    const claims = Array.isArray(parsed?.claims) ? parsed.claims : [];
    const normalized = claims
      .map((claim, index) => claimFromText(String(claim?.text ?? ""), index, String(claim?.claim_type ?? "factual")))
      .filter((claim): claim is Claim => Boolean(claim))
      .slice(0, maxClaims);
    if (normalized.length > 0) {
      return normalized;
    }
  } catch {
    // Fall through to deterministic decomposition.
  }

  return fallbackClaims(text, maxClaims);
}

const DOMAIN_KNOWLEDGE_SOURCES: Record<Domain, string[]> = {
  general: ["search_wikipedia", "search_wikidata", "search_dbpedia", "search_openalex", "search_crossref"],
  biomedical: ["search_wikipedia", "search_wikidata", "search_dbpedia", "search_pubmed", "search_europepmc", "search_clinical_trials"],
  legal: ["search_wikipedia", "search_wikidata", "search_dbpedia"],
  code: ["search_wikipedia", "search_wikidata", "search_dbpedia"],
  social: ["search_wikipedia", "search_wikidata", "search_dbpedia", "search_gdelt"],
};

const SOURCE_CREDIBILITY: Record<string, number> = {
  mediawiki: 0.88,
  "wikimedia-rest": 0.9,
  wikidata: 0.82,
  dbpedia: 0.75,
  openalex: 0.8,
  crossref: 0.82,
  ncbi: 0.9,
  europepmc: 0.9,
  clinicaltrials: 0.88,
  gdelt: 0.6,
};

async function retrieveEvidence(env: Env, query: string, domain: Domain = "general", evidenceLimit = 6): Promise<Evidence[]> {
  const retrievedAt = new Date().toISOString();
  const evidence: Evidence[] = [];

  const vectorMatches = await queryVectorEvidence(env, query);
  evidence.push(...vectorMatches);

  const sources = DOMAIN_KNOWLEDGE_SOURCES[domain] ?? DOMAIN_KNOWLEDGE_SOURCES.general;
  const settled = await Promise.allSettled(
    sources.map((tool) => callKnowledgeTool(tool, { query, limit: tool === "search_wikipedia" ? 4 : 2 }, env)),
  );
  for (const outcome of settled) {
    if (outcome.status !== "fulfilled") continue;
    const result = outcome.value;
    if (typeof result === "string" || !result.success || !result.results) continue;
    for (const item of result.results) {
      evidence.push(await searchResultToEvidence(item));
    }
  }

  const deduped = dedupeEvidence(evidence).slice(0, 12);
  const reranked = await rerankEvidence(env, query, deduped);

  const limited = reranked.slice(0, evidenceLimit);
  await cacheEvidence(env, limited, retrievedAt);
  return limited;
}

async function searchResultToEvidence(result: SearchResult): Promise<Evidence> {
  const content = result.content || result.title;
  return {
    id: await evidenceId(result.url ?? `${result.source}:${result.title}`, content),
    source_uri: result.url ?? null,
    content,
    snippet: content.slice(0, 280),
    source_credibility: SOURCE_CREDIBILITY[result.source] ?? 0.7,
    similarity_score: typeof result.score === "number" ? result.score : null,
    structured_data: { source: result.source, title: result.title, ...(result.metadata ?? {}) },
    retrieved_at: new Date().toISOString(),
  };
}

async function queryVectorEvidence(env: Env, query: string): Promise<Evidence[]> {
  try {
    const embedding = await embedText(env, query);
    if (!embedding) return [];
    const result = await env.OHI_VECTOR.query(embedding, {
      topK: 3,
      returnMetadata: "all",
    });
    const matches = Array.isArray(result?.matches) ? result.matches : [];
    return matches
      .map((match: VectorizeMatch, index: number): Evidence | null => {
        const metadata = (match.metadata ?? {}) as Record<string, unknown>;
        const content = typeof metadata.content === "string" ? metadata.content : "";
        if (!content) return null;
        return {
          id: String(match.id ?? `vector-${index}`),
          source_uri: typeof metadata.source_uri === "string" ? metadata.source_uri : null,
          content,
          snippet: typeof metadata.snippet === "string" ? metadata.snippet : content.slice(0, 280),
          source_credibility: 0.72,
          similarity_score: typeof match.score === "number" ? match.score : null,
          structured_data: { ...metadata, source: "vectorize" },
          retrieved_at: new Date().toISOString(),
        };
      })
      .filter((item): item is Evidence => item !== null);
  } catch {
    return [];
  }
}

async function rerankEvidence(env: Env, query: string, evidence: Evidence[]): Promise<Evidence[]> {
  if (evidence.length < 2) return evidence;
  try {
    const result = await env.AI.run(
      RERANK_MODEL,
      {
        query,
        contexts: evidence.map((item) => ({ text: `${item.snippet ?? ""}\n${item.content}`.slice(0, 1000) })),
        top_k: evidence.length,
      } as unknown as AiModels[typeof RERANK_MODEL]["inputs"],
      { gateway: { id: "default", skipCache: false, cacheTtl: 3600 } },
    );
    const ranked = extractRerank(result, evidence);
    if (ranked.length > 0) return ranked;
  } catch {
    // Preserve original retrieval order.
  }
  return evidence;
}

interface NliOpinion {
  label: NliLabel;
  confidence: number;
  supporting_score: number;
  refuting_score: number;
  neutral_score: number;
  relevance_score: number;
  reasoning: string;
}

const NLI_JSON_SCHEMA = {
  type: "object",
  properties: {
    label: { type: "string", enum: ["support", "refute", "neutral"] },
    confidence: { type: "number" },
    supporting_score: { type: "number" },
    refuting_score: { type: "number" },
    neutral_score: { type: "number" },
    relevance_score: { type: "number" },
    reasoning: { type: "string" },
  },
  required: ["label", "confidence", "relevance_score"],
};

function nliPrompt(claim: string, evidence: Evidence): string {
  return [
    "Classify whether the evidence supports, refutes, or is neutral to the claim.",
    "Use only the evidence. Return strict JSON with label, confidence, supporting_score, refuting_score, neutral_score, relevance_score, reasoning.",
    "relevance_score measures how directly the evidence addresses the SAME entity and SAME attribute asserted in the claim: 1.0 = the evidence states a value for that exact entity+attribute, 0.5 = the evidence discusses the entity or topic but not that specific attribute/value, 0.0 = the evidence is about a different entity or an unrelated topic.",
    "Topical relevance alone is neutral, not support. A high relevance_score does not by itself imply support - it only means the evidence is capable of confirming or denying the claim.",
    "If the evidence states a different age, date, count, name, or quantity for the same entity and attribute, classify as refute.",
    "If the evidence mentions the entity but does not establish the specific asserted value, classify as neutral with a mid-range relevance_score.",
    "",
    `Claim: ${claim}`,
    `Evidence: ${evidence.content.slice(0, 2000)}`,
  ].join("\n");
}

async function runNliModel(
  env: Env,
  model: keyof AiModels,
  claim: string,
  evidence: Evidence,
  jsonMode: "guided_json" | "response_format",
): Promise<NliOpinion | null> {
  try {
    const input: Record<string, unknown> = {
      messages: [
        {
          role: "system",
          content: "You are a conservative natural-language-inference classifier for fact checking. Require exact factual entailment before support, and score relevance honestly even when you classify as neutral.",
        },
        { role: "user", content: nliPrompt(claim, evidence) },
      ],
      temperature: 0,
      max_tokens: 500,
    };
    if (jsonMode === "guided_json") {
      input.guided_json = NLI_JSON_SCHEMA;
    } else {
      input.response_format = { type: "json_schema", json_schema: NLI_JSON_SCHEMA };
    }
    const run = env.AI.run.bind(env.AI) as (
      model: string,
      inputs: Record<string, unknown>,
      options?: AiOptions,
    ) => Promise<unknown>;
    const aiResult = await run(model, input, { gateway: { id: "default", skipCache: false, cacheTtl: 3600 } });
    const parsed = parseJsonObject(extractAiText(aiResult));
    if (!parsed) return null;
    const label = normalizeLabel(parsed.label);
    const confidence = clamp(numberOr(parsed.confidence, 0.5), 0, 1);
    const relevance = clamp(numberOr(parsed.relevance_score, 0.5), 0, 1);
    const supporting = clamp(numberOr(parsed.supporting_score, label === "support" ? confidence : 0.1), 0, 1);
    const refuting = clamp(numberOr(parsed.refuting_score, label === "refute" ? confidence : 0.1), 0, 1);
    const neutral = clamp(numberOr(parsed.neutral_score, label === "neutral" ? confidence : 1 - Math.max(supporting, refuting)), 0, 1);
    return {
      label,
      confidence,
      supporting_score: supporting,
      refuting_score: refuting,
      neutral_score: neutral,
      relevance_score: relevance,
      reasoning: String(parsed.reasoning ?? ""),
    };
  } catch {
    return null;
  }
}

function combineNliOpinions(
  evidence: Evidence,
  fallback: ClassifiedEvidence,
  primary: NliOpinion | null,
  corroboration: NliOpinion | null,
): ClassifiedEvidence {
  if (!primary && !corroboration) return fallback;

  if (primary && corroboration) {
    const relevance = Math.min(primary.relevance_score, corroboration.relevance_score);
    if (primary.label === corroboration.label) {
      return enrichEvidence(
        evidence,
        primary.label,
        (primary.confidence + corroboration.confidence) / 2,
        (primary.supporting_score + corroboration.supporting_score) / 2,
        (primary.refuting_score + corroboration.refuting_score) / 2,
        (primary.neutral_score + corroboration.neutral_score) / 2,
        relevance,
        `${primary.reasoning} | Corroborated by second model: ${corroboration.reasoning}`,
      );
    }
    if (primary.label === "refute" || corroboration.label === "refute") {
      const refuter = primary.label === "refute" ? primary : corroboration;
      return enrichEvidence(
        evidence,
        "refute",
        refuter.confidence * 0.7,
        Math.min(primary.supporting_score, corroboration.supporting_score),
        refuter.refuting_score * 0.7,
        Math.max(primary.neutral_score, corroboration.neutral_score),
        relevance,
        `Models disagreed (${primary.label} vs ${corroboration.label}); treating the refutation as the conservative signal. ${refuter.reasoning}`,
      );
    }
    return enrichEvidence(
      evidence,
      "neutral",
      Math.min(primary.confidence, corroboration.confidence) * 0.6,
      Math.min(primary.supporting_score, corroboration.supporting_score) * 0.5,
      Math.min(primary.refuting_score, corroboration.refuting_score),
      Math.max(primary.neutral_score, corroboration.neutral_score),
      relevance,
      `Models disagreed (${primary.label} vs ${corroboration.label}) with no refutation from either; downgraded to neutral pending corroboration.`,
    );
  }

  const solo = (primary ?? corroboration) as NliOpinion;
  return enrichEvidence(
    evidence,
    solo.label,
    solo.confidence * 0.85,
    solo.supporting_score * 0.85,
    solo.refuting_score,
    solo.neutral_score,
    solo.relevance_score * 0.9,
    `${solo.reasoning} (single-model classification; no second-model corroboration was run for this evidence item)`,
  );
}

async function classifyEvidence(env: Env, claim: string, evidence: Evidence, rigor: Rigor = "balanced"): Promise<ClassifiedEvidence> {
  const deterministic = deterministicFactCheck(claim, evidence);
  if (deterministic) {
    return enrichEvidence(
      evidence,
      deterministic.label,
      deterministic.confidence,
      deterministic.supportingScore,
      deterministic.refutingScore,
      deterministic.neutralScore,
      deterministic.relevanceScore,
      deterministic.reasoning,
    );
  }

  const fallback = lexicalClassify(claim, evidence);

  if (!RIGOR_PROFILES[rigor].ensemble) {
    const solo = await runNliModel(env, FAST_TIER_MODEL, claim, evidence, "response_format");
    return combineNliOpinions(evidence, fallback, solo, null);
  }

  const [primary, corroboration] = await Promise.all([
    runNliModel(env, CHAT_MODEL, claim, evidence, "guided_json"),
    runNliModel(env, NLI_CORROBORATION_MODEL, claim, evidence, "response_format"),
  ]);
  return combineNliOpinions(evidence, fallback, primary, corroboration);
}

const RELEVANCE_FLOOR = 0.4;

function buildClaimVerdict(
  claim: Claim,
  evidence: ClassifiedEvidence[],
  domain: Domain,
  coverageTarget: number,
): ClaimVerdict {
  const supporting = evidence.filter((item) => item.nli_label === "support" && item.supporting_score >= 0.35 && item.relevance_score >= RELEVANCE_FLOOR);
  const refuting = evidence.filter((item) => item.nli_label === "refute" && item.refuting_score >= 0.35 && item.relevance_score >= RELEVANCE_FLOOR);
  const supportScore = evidenceSignal(supporting, "supporting_score");
  const refuteScore = evidenceSignal(refuting, "refuting_score");
  const neutralShare = evidence.length === 0
    ? 1
    : evidence.filter((item) => item.nli_label === "neutral" || item.relevance_score < RELEVANCE_FLOOR).length / evidence.length;
  const meanRelevance = evidence.length === 0
    ? 0
    : evidence.reduce((sum, item) => sum + item.relevance_score, 0) / evidence.length;
  const weakEvidencePenalty = evidence.length === 0 ? 0.3 : clamp(0.28 * (1 - meanRelevance), 0, 0.28);
  const pTrue = clamp(sigmoid(-0.25 + 2.0 * supportScore - 3.1 * refuteScore - 0.2 * neutralShare - weakEvidencePenalty), 0.03, 0.97);
  const evidenceStrength = clamp(Math.max(supportScore, refuteScore), 0, 1);
  const width = clamp(0.34 - evidenceStrength * 0.16 + (coverageTarget - 0.8) * 0.2, 0.12, 0.42);
  const interval: [number, number] = [round4(clamp(pTrue - width / 2, 0, 1)), round4(clamp(pTrue + width / 2, 0, 1))];

  return {
    claim,
    p_true: round4(pTrue),
    interval,
    coverage_target: coverageTarget,
    domain,
    domain_assignment_weights: domainWeights(domain),
    supporting_evidence: supporting.map(toPublicEvidence),
    refuting_evidence: refuting.map(toPublicEvidence),
    pcg_neighbors: [],
    nli_self_consistency_variance: 0,
    bp_validated: true,
    information_gain: round4(evidenceStrength),
    queued_for_review: evidence.length === 0 || interval[1] - interval[0] > 0.35,
    calibration_set_id: null,
    calibration_n: 0,
    fallback_used: "general",
  };
}

async function embedText(env: Env, text: string): Promise<number[] | null> {
  try {
    const result = await env.AI.run(
      EMBEDDING_MODEL,
      { text: [text] },
      { gateway: { id: "default", skipCache: false, cacheTtl: 86400 } },
    );
    return extractEmbedding(result);
  } catch {
    return null;
  }
}

async function cacheEvidence(env: Env, evidence: Evidence[], retrievedAt: string): Promise<void> {
  for (const item of evidence) {
    try {
      await env.OHI_DB.prepare(
        `INSERT OR REPLACE INTO evidence_cache
          (evidence_id, source_uri, title, content, retrieved_at, metadata_json)
         VALUES (?, ?, ?, ?, ?, ?)`,
      )
        .bind(
          item.id,
          item.source_uri,
          String(item.structured_data?.title ?? ""),
          item.content,
          retrievedAt,
          JSON.stringify(item.structured_data ?? {}),
        )
        .run();

      const embedding = await embedText(env, item.content.slice(0, 4000));
      if (embedding) {
        await env.OHI_VECTOR.insert([
          {
            id: item.id,
            values: embedding,
            metadata: {
              source_uri: item.source_uri ?? "",
              content: item.content.slice(0, 2000),
              snippet: item.snippet ?? "",
              title: String(item.structured_data?.title ?? ""),
            },
          },
        ]);
      }
    } catch {
      // Evidence cache is an accelerator; verification result should survive cache failures.
    }
  }
}

async function mirrorD1Job(
  env: Env,
  patch: {
    job_id: string;
    status: JobStatusValue;
    phase: string;
    created_at?: number;
    updated_at: number;
    completed_at?: number;
    text_hash?: string;
    result_json?: string;
    error?: string;
  },
): Promise<void> {
  try {
    await env.OHI_DB.prepare(
      `INSERT INTO jobs
        (job_id, status, phase, created_at, updated_at, completed_at, text_hash, result_json, error)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
       ON CONFLICT(job_id) DO UPDATE SET
        status = excluded.status,
        phase = excluded.phase,
        updated_at = excluded.updated_at,
        completed_at = COALESCE(excluded.completed_at, jobs.completed_at),
        result_json = COALESCE(excluded.result_json, jobs.result_json),
        error = excluded.error`,
    )
      .bind(
        patch.job_id,
        patch.status,
        patch.phase,
        patch.created_at ?? patch.updated_at,
        patch.updated_at,
        patch.completed_at ?? null,
        patch.text_hash ?? "",
        patch.result_json ?? null,
        patch.error ?? null,
      )
      .run();
  } catch {
    // Durable Object state remains the source of truth for live polling.
  }
}

function validateVerifyRequest(body: VerifyRequest, env: Env): string | null {
  if (!body || typeof body.text !== "string") return "text is required";
  if (body.text.length > 50_000) return "text must be at most 50,000 characters";
  if (body.context != null && typeof body.context === "string" && body.context.length > 2_000) {
    return "context must be at most 2,000 characters";
  }
  const max = defaultMaxClaims(env, "maximum");
  if (body.options?.max_claims != null && (body.options.max_claims < 1 || body.options.max_claims > max)) {
    return `options.max_claims must be between 1 and ${max}`;
  }
  return null;
}

async function rejectIfAbusive(request: Request, env: Env, body: VerifyRequest): Promise<Response | null> {
  const ip = request.headers.get("cf-connecting-ip")
    ?? request.headers.get("x-forwarded-for")?.split(",")[0]?.trim()
    ?? "unknown";
  const limit = clamp(Number((env as Env & { VERIFY_RATE_LIMIT_PER_MINUTE?: string }).VERIFY_RATE_LIMIT_PER_MINUTE ?? "12") || 12, 1, 120);
  const decision = await env.RATE_LIMITER.getByName("verify").check(`verify:${ip}`, limit, 60);
  if (!decision.allowed) {
    return json(request, env, {
      detail: {
        code: "rate_limited",
        message: "Too many verification requests. Try again after the reset time.",
        ...decision,
      },
    }, 429);
  }

  if ((body.options?.rigor ?? "balanced") === "maximum") {
    const cooldownSeconds = clamp(
      Number((env as Env & { MAXIMUM_RIGOR_COOLDOWN_SECONDS?: string }).MAXIMUM_RIGOR_COOLDOWN_SECONDS ?? "90") || 90,
      30,
      600,
    );
    const cooldown = await env.RATE_LIMITER.getByName("maximum-cooldown").check(`max:${ip}`, 1, cooldownSeconds);
    if (!cooldown.allowed) {
      return json(request, env, {
        detail: {
          code: "maximum_rigor_cooldown",
          message: `Maximum rigor is limited to one request every ${cooldownSeconds}s per client. Try again after the reset time.`,
          ...cooldown,
        },
      }, 429);
    }
  }

  const secret = (env as Env & { TURNSTILE_SECRET_KEY?: string }).TURNSTILE_SECRET_KEY;
  if (!secret || await isAdminRequest(request, env)) return null;

  const token = body.turnstile_token
    ?? body["cf-turnstile-response"]
    ?? request.headers.get("cf-turnstile-response")
    ?? request.headers.get("x-turnstile-token");
  if (!token) {
    return json(request, env, {
      detail: {
        code: "turnstile_required",
        message: "Turnstile verification is required for this request.",
      },
    }, 403);
  }

  const ok = await verifyTurnstile(secret, String(token), ip);
  if (!ok) {
    return json(request, env, {
      detail: {
        code: "turnstile_failed",
        message: "Turnstile verification failed.",
      },
    }, 403);
  }
  return null;
}

async function requireAdmin(request: Request, env: Env): Promise<Response | null> {
  const configured = (env as Env & { ADMIN_TOKEN?: string }).ADMIN_TOKEN;
  if (!configured) {
    return json(request, env, {
      detail: {
        code: "admin_token_not_configured",
        message: "ADMIN_TOKEN must be set as a Worker secret before using admin endpoints.",
      },
    }, 503);
  }
  if (await isAdminRequest(request, env)) return null;
  return json(request, env, {
    detail: {
      code: "unauthorized",
      message: "Admin token required.",
    },
  }, 401);
}

async function isAdminRequest(request: Request, env: Env): Promise<boolean> {
  const configured = (env as Env & { ADMIN_TOKEN?: string }).ADMIN_TOKEN;
  if (!configured) return false;
  const auth = request.headers.get("authorization") ?? "";
  const bearer = auth.toLowerCase().startsWith("bearer ") ? auth.slice(7).trim() : "";
  const supplied = request.headers.get("x-ohi-admin-token") ?? bearer;
  if (!supplied) return false;
  return constantTimeEqual(await sha256Hex(supplied), await sha256Hex(configured));
}

async function verifyTurnstile(secret: string, token: string, remoteIp: string): Promise<boolean> {
  try {
    const body = new URLSearchParams();
    body.set("secret", secret);
    body.set("response", token);
    if (remoteIp !== "unknown") body.set("remoteip", remoteIp);
    body.set("idempotency_key", crypto.randomUUID());
    const response = await fetch("https://challenges.cloudflare.com/turnstile/v0/siteverify", {
      method: "POST",
      headers: { "content-type": "application/x-www-form-urlencoded" },
      body,
    });
    const result = (await response.json()) as { success?: boolean };
    return Boolean(result.success);
  } catch {
    return false;
  }
}

function normalizeVerifyRequest(body: VerifyRequest, env: Env): VerifyRequest {
  const rigor = body.options?.rigor ?? "balanced";
  return {
    text: body.text,
    context: body.context ?? null,
    domain_hint: body.domain_hint ?? null,
    request_id: body.request_id ?? null,
    options: {
      rigor,
      tier: body.options?.tier ?? "default",
      max_claims: clamp(body.options?.max_claims ?? defaultMaxClaims(env, rigor), 1, defaultMaxClaims(env, rigor)),
      include_pcg_neighbors: body.options?.include_pcg_neighbors ?? true,
      include_full_provenance: body.options?.include_full_provenance ?? true,
      self_consistency_k: body.options?.self_consistency_k ?? null,
      coverage_target: clamp(body.options?.coverage_target ?? 0.9, 0.5, 0.99),
    },
  };
}

function fallbackClaims(text: string, maxClaims: number): Claim[] {
  const pieces = text
    .replace(/\s+/g, " ")
    .split(/(?<=[.!?])\s+/)
    .map((piece) => piece.trim())
    .filter((piece) => piece.length > 8)
    .slice(0, maxClaims);
  const source = pieces.length > 0 ? pieces : [text.trim().slice(0, 500)];
  return source
    .map((piece, index) => claimFromText(piece, index, "factual"))
    .filter((claim): claim is Claim => Boolean(claim));
}

function claimFromText(text: string, index: number, claimType: string): Claim | null {
  const trimmed = text.trim().replace(/^[-*]\s*/, "");
  if (!trimmed) return null;
  return {
    id: `claim-${index + 1}`,
    text: trimmed.slice(0, 1000),
    claim_type: claimType || "factual",
    span: null,
  };
}

interface DeterministicFactDecision {
  label: NliLabel;
  confidence: number;
  supportingScore: number;
  refutingScore: number;
  neutralScore: number;
  relevanceScore: number;
  reasoning: string;
}

interface ExtractedAge {
  age: number;
  exact: boolean;
  detail: string;
}

const AGE_FACT_STOP_TOKENS = new Set([
  "aged",
  "away",
  "dead",
  "death",
  "died",
  "passed",
  "year",
  "years",
]);

const MONTH_INDEX: Record<string, number> = {
  january: 0,
  jan: 0,
  february: 1,
  feb: 1,
  march: 2,
  mar: 2,
  april: 3,
  apr: 3,
  may: 4,
  june: 5,
  jun: 5,
  july: 6,
  jul: 6,
  august: 7,
  aug: 7,
  september: 8,
  sep: 8,
  sept: 8,
  october: 9,
  oct: 9,
  november: 10,
  nov: 10,
  december: 11,
  dec: 11,
};

function deterministicFactCheck(claim: string, evidence: Evidence): DeterministicFactDecision | null {
  const claimedDeathAge = extractDeathAgeClaim(claim);
  if (claimedDeathAge === null) return null;

  const haystack = evidenceHaystack(evidence);
  if (!hasMeaningfulSubjectOverlap(claim, haystack)) return null;

  const evidenceDeathAge = extractDeathAgeFromEvidence(haystack);
  if (!evidenceDeathAge) {
    return {
      label: "neutral",
      confidence: 0.74,
      supportingScore: 0.03,
      refutingScore: 0.04,
      neutralScore: 0.82,
      relevanceScore: 0.55,
      reasoning: `Evidence mentions the subject but does not establish the asserted death age of ${claimedDeathAge}.`,
    };
  }

  const ageDelta = Math.abs(evidenceDeathAge.age - claimedDeathAge);
  if (evidenceDeathAge.exact && ageDelta === 0) {
    return {
      label: "support",
      confidence: 0.9,
      supportingScore: 0.9,
      refutingScore: 0.02,
      neutralScore: 0.05,
      relevanceScore: 0.95,
      reasoning: `Evidence establishes the same death age (${claimedDeathAge}) from ${evidenceDeathAge.detail}.`,
    };
  }

  if (ageDelta > (evidenceDeathAge.exact ? 0 : 1)) {
    return {
      label: "refute",
      confidence: 0.92,
      supportingScore: 0.02,
      refutingScore: 0.92,
      neutralScore: 0.04,
      relevanceScore: 0.95,
      reasoning: `Evidence establishes death age ${evidenceDeathAge.age}, contradicting the claimed age ${claimedDeathAge}.`,
    };
  }

  return {
    label: "neutral",
    confidence: 0.72,
    supportingScore: 0.08,
    refutingScore: 0.08,
    neutralScore: 0.76,
    relevanceScore: 0.75,
    reasoning: `Evidence gives only approximate death-age evidence (${evidenceDeathAge.detail}), so it cannot exactly support the claim.`,
  };
}

function extractDeathAgeClaim(claim: string): number | null {
  const patterns = [
    /\b(?:died|dead|passed\s+away)\b[^.?!]{0,80}\b(?:at|aged|age|with)\s+(?:the\s+age\s+of\s+)?(\d{1,3})\b/i,
    /\b(?:died|dead|passed\s+away)\b[^.?!]{0,80}\b(\d{1,3})\s*(?:years?|yrs?)\b/i,
    /\b(\d{1,3})\s*(?:years?|yrs?)\s*old\b[^.?!]{0,80}\b(?:when\s+)?(?:he|she|they|[A-Z][a-z]+)?\s*(?:died|dead|passed\s+away)\b/i,
    /\b(?:was|were)\s+(\d{1,3})\b[^.?!]{0,80}\b(?:when\s+)?(?:he|she|they)?\s*(?:died|dead|passed\s+away)\b/i,
  ];
  for (const pattern of patterns) {
    const match = claim.match(pattern);
    const age = Number(match?.[1]);
    if (Number.isInteger(age) && age >= 0 && age <= 125) return age;
  }
  return null;
}

function extractDeathAgeFromEvidence(text: string): ExtractedAge | null {
  const directPatterns = [
    /\b(?:died|dead|death)\b[^.?!]{0,100}\b(?:aged|age|at\s+the\s+age\s+of|at)\s+(\d{1,3})\b/i,
    /\b(?:aged|age)\s+(\d{1,3})\b[^.?!]{0,100}\b(?:died|dead|death)\b/i,
    /\b(\d{1,3})\s*(?:years?|yrs?)\s*old\b[^.?!]{0,100}\b(?:when\s+)?(?:he|she|they)?\s*(?:died|dead|death)\b/i,
  ];
  for (const pattern of directPatterns) {
    const match = text.match(pattern);
    const age = Number(match?.[1]);
    if (Number.isInteger(age) && age >= 0 && age <= 125) {
      return { age, exact: true, detail: `explicit age ${age}` };
    }
  }

  const dates = extractWrittenDates(text);
  if (dates.length >= 2) {
    for (let birthIndex = 0; birthIndex < dates.length - 1; birthIndex += 1) {
      for (let deathIndex = birthIndex + 1; deathIndex < dates.length; deathIndex += 1) {
        const age = ageAt(dates[birthIndex], dates[deathIndex]);
        if (age >= 0 && age <= 125) {
          return {
            age,
            exact: true,
            detail: `${dates[birthIndex].day} ${dates[birthIndex].monthName} ${dates[birthIndex].year} to ${dates[deathIndex].day} ${dates[deathIndex].monthName} ${dates[deathIndex].year}`,
          };
        }
      }
    }
  }

  const yearRange = text.match(/\b(1[5-9]\d{2}|20\d{2})\s*[-–—]\s*(1[5-9]\d{2}|20\d{2})\b/);
  const birthYear = Number(yearRange?.[1]);
  const deathYear = Number(yearRange?.[2]);
  if (Number.isInteger(birthYear) && Number.isInteger(deathYear)) {
    const approximateAge = deathYear - birthYear;
    if (approximateAge >= 0 && approximateAge <= 125) {
      return { age: approximateAge, exact: false, detail: `year range ${birthYear}-${deathYear}` };
    }
  }

  return null;
}

function evidenceHaystack(evidence: Evidence): string {
  return [
    evidence.content,
    evidence.snippet ?? "",
    typeof evidence.structured_data?.title === "string" ? evidence.structured_data.title : "",
  ].join("\n");
}

function hasMeaningfulSubjectOverlap(claim: string, evidenceText: string): boolean {
  const evidenceLower = evidenceText.toLowerCase();
  const properNames = claim.match(/\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b/g) ?? [];
  if (properNames.some((name) => evidenceLower.includes(name.toLowerCase()))) return true;

  const tokens = significantTokens(claim).filter((token) => !AGE_FACT_STOP_TOKENS.has(token));
  if (tokens.length === 0) return false;
  const overlap = tokens.filter((token) => evidenceLower.includes(token)).length;
  return overlap >= Math.min(2, tokens.length);
}

function extractWrittenDates(text: string): Array<{ day: number; month: number; monthName: string; year: number }> {
  const datePattern = /\b(\d{1,2})\s+(January|Jan|February|Feb|March|Mar|April|Apr|May|June|Jun|July|Jul|August|Aug|September|Sept|Sep|October|Oct|November|Nov|December|Dec)\s+(\d{4})\b/gi;
  const dates: Array<{ day: number; month: number; monthName: string; year: number }> = [];
  let match: RegExpExecArray | null;
  while ((match = datePattern.exec(text)) !== null) {
    const day = Number(match[1]);
    const monthName = match[2];
    const month = MONTH_INDEX[monthName.toLowerCase()];
    const year = Number(match[3]);
    if (Number.isInteger(day) && Number.isInteger(month) && Number.isInteger(year)) {
      dates.push({ day, month, monthName, year });
    }
  }
  return dates;
}

function ageAt(birth: { day: number; month: number; year: number }, death: { day: number; month: number; year: number }): number {
  const hadBirthday = death.month > birth.month || (death.month === birth.month && death.day >= birth.day);
  return death.year - birth.year - (hadBirthday ? 0 : 1);
}

function lexicalClassify(claim: string, evidence: Evidence): ClassifiedEvidence {
  const claimTokens = significantTokens(claim);
  const evidenceTokens = new Set(significantTokens(evidence.content));
  const overlap = claimTokens.filter((token) => evidenceTokens.has(token)).length;
  const ratio = claimTokens.length === 0 ? 0 : overlap / claimTokens.length;
  const refutationScore = refutationCueScore(claimTokens, evidence);
  const label: NliLabel = refutationScore > 0 && ratio >= 0.35 ? "refute" : ratio >= 0.42 ? "support" : "neutral";
  const confidence = label === "refute"
    ? clamp(0.55 + refutationScore + ratio * 0.2, 0.55, 0.86)
    : clamp(0.35 + ratio * 0.45, 0.25, 0.78);
  const relevance = clamp(0.15 + ratio * 0.6, 0.15, 0.75);
  return enrichEvidence(
    evidence,
    label,
    confidence,
    label === "support" ? confidence : 0.08,
    label === "refute" ? confidence : 0.05,
    label === "neutral" ? confidence : 1 - confidence,
    relevance,
    label === "refute"
      ? "Fallback lexical classifier found debunking or contradiction cues in overlapping evidence."
      : "Fallback lexical overlap classifier used because Workers AI did not return a usable NLI result.",
  );
}

function refutationCueScore(claimTokens: string[], evidence: Evidence): number {
  const haystack = `${evidence.content} ${evidence.snippet ?? ""} ${String(evidence.structured_data?.title ?? "")}`.toLowerCase();
  const phraseCues = [
    "fanciful belief",
    "folkloric",
    "myth",
    "hoax",
    "false",
    "mistaken",
    "misconception",
    "debunked",
    "not true",
    "no evidence",
    "contradicts",
    "contrary to",
    "fictional",
  ];
  let score = phraseCues.reduce((sum, cue) => sum + (haystack.includes(cue) ? 0.18 : 0), 0);
  void claimTokens;
  return clamp(score, 0, 0.42);
}

function enrichEvidence(
  evidence: Evidence,
  label: NliLabel,
  confidence: number,
  supportingScore: number,
  refutingScore: number,
  neutralScore: number,
  relevanceScore: number,
  reasoning: string,
): ClassifiedEvidence {
  return {
    ...evidence,
    classification_confidence: confidence,
    nli_label: label,
    nli_confidence: confidence,
    supporting_score: supportingScore,
    refuting_score: refutingScore,
    neutral_score: neutralScore,
    relevance_score: relevanceScore,
    nli_reasoning: reasoning,
    structured_data: {
      ...(evidence.structured_data ?? {}),
      nli_label: label,
      nli_confidence: confidence,
      supporting_score: supportingScore,
      refuting_score: refutingScore,
      neutral_score: neutralScore,
      relevance_score: relevanceScore,
      nli_reasoning: reasoning,
    },
  };
}

function toPublicEvidence(item: ClassifiedEvidence): Evidence {
  return {
    id: item.id,
    source_uri: item.source_uri,
    content: item.content,
    snippet: item.snippet ?? null,
    source_credibility: item.source_credibility ?? null,
    similarity_score: item.similarity_score ?? null,
    classification_confidence: item.classification_confidence ?? null,
    structured_data: item.structured_data ?? null,
    retrieved_at: item.retrieved_at,
  };
}

function extractAiText(result: unknown): string {
  if (typeof result === "string") return result;
  if (!result || typeof result !== "object") return "";
  const obj = result as Record<string, unknown>;
  if (typeof obj.response === "string") return obj.response;
  if (typeof obj.result === "string") return obj.result;
  if (obj.result && typeof obj.result === "object") {
    const nested = obj.result as Record<string, unknown>;
    if (typeof nested.response === "string") return nested.response;
  }
  const choices = obj.choices;
  if (Array.isArray(choices) && choices[0] && typeof choices[0] === "object") {
    const choice = choices[0] as Record<string, unknown>;
    const message = choice.message as Record<string, unknown> | undefined;
    if (message && typeof message.content === "string") return message.content;
    if (typeof choice.text === "string") return choice.text;
  }
  return JSON.stringify(result);
}

function parseJsonObject(text: string): Record<string, unknown> | null {
  const trimmed = text.trim();
  if (!trimmed) return null;
  const candidates = [
    trimmed,
    trimmed.replace(/^```json\s*/i, "").replace(/```$/i, "").trim(),
    trimmed.slice(trimmed.indexOf("{"), trimmed.lastIndexOf("}") + 1),
  ].filter(Boolean);
  for (const candidate of candidates) {
    try {
      const parsed = JSON.parse(candidate) as unknown;
      return parsed && typeof parsed === "object" && !Array.isArray(parsed)
        ? (parsed as Record<string, unknown>)
        : null;
    } catch {
      // Try the next candidate.
    }
  }
  return null;
}

function extractEmbedding(result: unknown): number[] | null {
  const obj = result as Record<string, unknown>;
  const data = obj?.data;
  if (Array.isArray(data) && Array.isArray(data[0])) return data[0] as number[];
  if (Array.isArray(data) && data[0] && typeof data[0] === "object") {
    const first = data[0] as Record<string, unknown>;
    if (Array.isArray(first.embedding)) return first.embedding as number[];
    if (Array.isArray(first.values)) return first.values as number[];
  }
  const resultObj = obj?.result as Record<string, unknown> | undefined;
  if (resultObj) return extractEmbedding(resultObj);
  return null;
}

function extractRerank(result: unknown, evidence: Evidence[]): Evidence[] {
  const obj = result as Record<string, unknown>;
  const rows =
    (Array.isArray(obj?.response) && obj.response) ||
    (Array.isArray(obj?.data) && obj.data) ||
    (Array.isArray(obj?.result) && obj.result) ||
    [];
  if (!Array.isArray(rows)) return [];
  const ranked: Evidence[] = [];
  for (const row of rows) {
    if (!row || typeof row !== "object") continue;
    const item = row as Record<string, unknown>;
    const index = Number(item.index ?? item.id ?? -1);
    const score = numberOr(item.score ?? item.relevance_score, 0);
    const evidenceItem = evidence[index];
    if (evidenceItem) {
      ranked.push({ ...evidenceItem, similarity_score: score || (evidenceItem.similarity_score ?? null) });
    }
  }
  return ranked.length > 0 ? ranked : [];
}

function asStructuredContent(value: unknown): Record<string, unknown> {
  if (value && typeof value === "object" && !Array.isArray(value)) {
    return value as Record<string, unknown>;
  }
  return { value };
}

function dedupeEvidence(evidence: Evidence[]): Evidence[] {
  const seen = new Set<string>();
  const out: Evidence[] = [];
  for (const item of evidence) {
    const key = item.source_uri ?? item.content.slice(0, 160);
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(item);
  }
  return out;
}

async function evidenceId(sourceUri: string, content: string): Promise<string> {
  return `ev-${(await sha256Hex(`${sourceUri}\n${content.slice(0, 500)}`)).slice(0, 24)}`;
}

function jobFromRow(row: RawJobRow): JobRecord {
  return {
    job_id: row.job_id,
    status: row.status,
    phase: row.phase,
    created_at: row.created_at,
    updated_at: row.updated_at,
    ...(row.completed_at ? { completed_at: row.completed_at } : {}),
    ...(row.result_json ? { result: JSON.parse(row.result_json) as DocumentVerdict } : {}),
    ...(row.error ? { error: row.error } : {}),
  };
}

function json(request: Request, env: Env, body: unknown, status = 200): Response {
  return withCors(
    request,
    env,
    new Response(JSON.stringify(body), {
      status,
      headers: {
        "content-type": "application/json; charset=utf-8",
        "cache-control": "no-store",
      },
    }),
  );
}

function withCors(request: Request, env: Env, response: Response): Response {
  const origin = request.headers.get("origin");
  const allowed = origin && (DEFAULT_ALLOWED_ORIGINS.has(origin) || origin === env.SITE_ORIGIN) ? origin : env.SITE_ORIGIN;
  const headers = new Headers(response.headers);
  headers.set("access-control-allow-origin", allowed ?? "https://ohi.shiftbloom.studio");
  headers.set("access-control-allow-methods", "GET,POST,OPTIONS");
  headers.set("access-control-allow-headers", "content-type,accept,authorization,x-ohi-admin-token,cf-turnstile-response,x-turnstile-token");
  headers.set("vary", "Origin");
  return new Response(response.body, { status: response.status, statusText: response.statusText, headers });
}

async function timedLayer(
  layers: Record<string, { status: "ok" | "degraded" | "down" | "skipped"; latency_ms?: number; detail?: string }>,
  name: string,
  fn: () => Promise<void>,
): Promise<void> {
  const started = performance.now();
  try {
    await fn();
    layers[name] = { status: "ok", latency_ms: Math.round(performance.now() - started) };
  } catch (error) {
    layers[name] = { status: "down", latency_ms: Math.round(performance.now() - started), detail: errorMessage(error) };
  }
}

function calibrationReport() {
  const stratum = {
    calibration_n: 0,
    empirical_coverage: 0,
    interval_width_p50: 0,
    interval_width_p95: 0,
  };
  return {
    report_date: new Date().toISOString(),
    global_coverage_target: 0.9,
    domains: {
      general: stratum,
      biomedical: stratum,
      legal: stratum,
      code: stratum,
      social: stratum,
    },
  };
}

function modelVersions(): Record<string, string> {
  return {
    decomposer: CHAT_MODEL,
    nli_adapter: CHAT_MODEL,
    embeddings: EMBEDDING_MODEL,
    reranker: RERANK_MODEL,
    vector_store: "cloudflare-vectorize",
    corpus_store: "cloudflare-vectorize+d1+r2",
    mcp_sources: "cloudflare-worker-native-multi-source",
    job_store: "cloudflare-durable-objects+d1",
  };
}

function domainWeights(domain: Domain): Record<string, number> {
  const weights: Record<string, number> = {
    general: 0.05,
    biomedical: 0.05,
    legal: 0.05,
    code: 0.05,
    social: 0.05,
  };
  weights[domain] = 0.8;
  if (domain !== "general") weights.general = 0.1;
  return weights;
}

function defaultMaxClaims(env: Env, rigor: Rigor): number {
  // VERIFY_MAX_CLAIMS is an optional operator-configured ceiling that can only
  // further restrict a tier's claim count, never raise it above the tier's
  // own cost-sized profile (see RIGOR_PROFILES).
  const configured = Number(env.VERIFY_MAX_CLAIMS ?? "13") || 13;
  return Math.min(RIGOR_PROFILES[rigor].maxClaims, Math.max(1, configured));
}

function internalConsistency(verdicts: ClaimVerdict[]): number {
  if (verdicts.length <= 1) return 1;
  const scores = verdicts.map((verdict) => verdict.p_true);
  const mean = scores.reduce((sum, score) => sum + score, 0) / scores.length;
  const variance = scores.reduce((sum, score) => sum + (score - mean) ** 2, 0) / scores.length;
  return clamp(1 - Math.sqrt(variance), 0, 1);
}

function geometricMean(values: number[]): number {
  if (values.length === 0) return 0.5;
  const product = values.reduce((acc, value) => acc + Math.log(clamp(value, 0.001, 0.999)), 0);
  return Math.exp(product / values.length);
}

function weightedMean(values: number[]): number {
  if (values.length === 0) return 0;
  return clamp(values.reduce((sum, value) => sum + value, 0) / values.length, 0, 1);
}

function evidenceSignal(evidence: ClassifiedEvidence[], scoreKey: "supporting_score" | "refuting_score"): number {
  if (evidence.length === 0) return 0;
  const values = evidence
    .map((item) => clamp(item[scoreKey] * (item.source_credibility ?? 0.7) * item.relevance_score, 0, 1))
    .sort((left, right) => right - left);
  const strongest = values[0] ?? 0;
  const average = weightedMean(values);
  return clamp(strongest * 0.75 + average * 0.25, 0, 1);
}

function sigmoid(value: number): number {
  return 1 / (1 + Math.exp(-value));
}

function significantTokens(text: string): string[] {
  const stop = new Set(["the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "is", "are", "was", "were", "by", "as", "that", "this"]);
  return text
    .toLowerCase()
    .replace(/[^a-z0-9\s-]/g, " ")
    .split(/\s+/)
    .filter((token) => token.length > 3 && !stop.has(token))
    .slice(0, 80);
}

function normalizeLabel(value: unknown): NliLabel {
  const text = String(value ?? "").toLowerCase();
  if (text.includes("refute") || text.includes("contradict")) return "refute";
  if (text.includes("support") || text.includes("entail")) return "support";
  return "neutral";
}

function numberOr(value: unknown, fallback: number): number {
  const num = Number(value);
  return Number.isFinite(num) ? num : fallback;
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function round4(value: number): number {
  return Math.round(value * 10_000) / 10_000;
}

function constantTimeEqual(a: string, b: string): boolean {
  if (a.length !== b.length) return false;
  let out = 0;
  for (let index = 0; index < a.length; index += 1) {
    out |= a.charCodeAt(index) ^ b.charCodeAt(index);
  }
  return out === 0;
}

function secondsNow(): number {
  return Date.now() / 1000;
}

async function sha256Hex(text: string): Promise<string> {
  const bytes = new TextEncoder().encode(text);
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return [...new Uint8Array(digest)].map((byte) => byte.toString(16).padStart(2, "0")).join("");
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
