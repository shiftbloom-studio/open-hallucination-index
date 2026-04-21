import type {
  ApiErrorBody,
  CalibrationReport,
  DocumentVerdict,
  FeedbackRequest,
  HealthDeep,
  HealthStatus,
  ReadinessStatus,
  VerifyRequest,
} from "./ohi-types";

const DEFAULT_PUBLIC_API_BASE = "https://api.ohi.shiftbloom.studio/api/v2";

function apiBase(): string {
  const base = process.env.NEXT_PUBLIC_API_BASE;
  if (!base) {
    return DEFAULT_PUBLIC_API_BASE;
  }
  return base.endsWith("/") ? base.slice(0, -1) : base;
}

function healthUrl(path: string): string {
  return new URL(path, apiBase()).toString();
}

export class OhiError extends Error {
  readonly status: number;
  readonly body: ApiErrorBody;
  readonly retryAfterSec?: number;

  constructor(status: number, body: ApiErrorBody, retryAfterSec?: number) {
    super(`OHI ${status}: ${JSON.stringify(body)}`);
    this.name = "OhiError";
    this.status = status;
    this.body = body;
    this.retryAfterSec = retryAfterSec;
  }

  get isResting(): boolean {
    return (
      this.status === 503 &&
      "status" in this.body &&
      (this.body as { status?: string }).status === "resting"
    );
  }

  get isLlmDown(): boolean {
    return (
      this.status === 503 &&
      "status" in this.body &&
      (this.body as { status?: string }).status === "llm_unavailable"
    );
  }

  get isBudgetExhausted(): boolean {
    const detail = (this.body as { detail?: string }).detail;
    return this.status === 503 && typeof detail === "string" && detail.includes("budget exhausted");
  }

  get isDegraded(): boolean {
    const layers = (this.body as { degraded_layers?: string[] }).degraded_layers;
    return Array.isArray(layers) && layers.length > 0;
  }

  get isRateLimited(): boolean {
    return this.status === 429;
  }
}

async function toError(res: Response): Promise<OhiError> {
  const retryAfter = Number(res.headers.get("Retry-After") ?? "") || undefined;
  let body: ApiErrorBody;
  try {
    body = (await res.json()) as ApiErrorBody;
  } catch {
    body = { detail: res.statusText };
  }
  return new OhiError(res.status, body, retryAfter);
}

async function asJson<T>(res: Response): Promise<T> {
  if (!res.ok) throw await toError(res);
  return (await res.json()) as T;
}

export interface RequestOptions {
  signal?: AbortSignal;
}

export interface JobAccepted {
  job_id: string;
}

export interface JobStatus {
  job_id: string;
  status: "pending" | "done" | "error";
  phase: string;
  created_at: number;
  updated_at: number;
  completed_at?: number;
  result?: DocumentVerdict;
  error?: string;
}

export const ohi = {
  /**
   * Submit a verification job. Post-D2 this no longer blocks on the
   * pipeline — it returns 202 with a job_id; poll `.verifyStatus(id)`
   * until the status transitions to "done" or "error".
   */
  verify: async (req: VerifyRequest, opts: RequestOptions = {}): Promise<JobAccepted> => {
    const res = await fetch(`${apiBase()}/verify`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "application/json" },
      body: JSON.stringify(req),
      signal: opts.signal,
    });
    return asJson<JobAccepted>(res);
  },

  verifyStatus: async (jobId: string, opts: RequestOptions = {}): Promise<JobStatus> => {
    const res = await fetch(`${apiBase()}/verify/status/${encodeURIComponent(jobId)}`, {
      headers: { Accept: "application/json" },
      signal: opts.signal,
    });
    return asJson<JobStatus>(res);
  },

  verdict: async (id: string, opts: RequestOptions = {}): Promise<DocumentVerdict> => {
    const res = await fetch(`${apiBase()}/verdict/${encodeURIComponent(id)}`, {
      headers: { Accept: "application/json" },
      signal: opts.signal,
    });
    return asJson<DocumentVerdict>(res);
  },

  feedback: async (
    req: FeedbackRequest,
    opts: RequestOptions = {},
  ): Promise<{ feedback_id: string; queued: true }> => {
    const res = await fetch(`${apiBase()}/feedback`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "application/json" },
      body: JSON.stringify(req),
      signal: opts.signal,
    });
    return asJson<{ feedback_id: string; queued: true }>(res);
  },

  calibrationReport: async (opts: RequestOptions = {}): Promise<CalibrationReport> => {
    const res = await fetch(`${apiBase()}/calibration/report`, {
      headers: { Accept: "application/json" },
      signal: opts.signal,
    });
    return asJson<CalibrationReport>(res);
  },

  healthDeep: async (opts: RequestOptions = {}): Promise<HealthDeep> => {
    // /health/* lives at the origin root, not under /api/v2 — Lambda keeps it
    // there so its own container health probes hit /health/live without the
    // /api/v2 prefix. Build a URL that drops whatever path apiBase() carries.
    const url = healthUrl("/health/deep");
    const res = await fetch(url, {
      headers: { Accept: "application/json" },
      signal: opts.signal,
    });
    return asJson<HealthDeep>(res);
  },

  healthLive: async (opts: RequestOptions = {}): Promise<HealthStatus> => {
    const res = await fetch(healthUrl("/health/live"), {
      headers: { Accept: "application/json" },
      signal: opts.signal,
    });
    return asJson<HealthStatus>(res);
  },

  healthReady: async (opts: RequestOptions = {}): Promise<ReadinessStatus> => {
    const res = await fetch(healthUrl("/health/ready"), {
      headers: { Accept: "application/json" },
      signal: opts.signal,
    });
    return asJson<ReadinessStatus>(res);
  },
};

export type OhiClient = typeof ohi;
