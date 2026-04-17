import type {
  ApiErrorBody,
  CalibrationReport,
  DocumentVerdict,
  FeedbackRequest,
  HealthDeep,
  VerifyRequest,
} from "./ohi-types";

function apiBase(): string {
  const base = process.env.NEXT_PUBLIC_API_BASE;
  if (!base) {
    throw new Error(
      "NEXT_PUBLIC_API_BASE is not set. Expected e.g. https://api.ohi.shiftbloom.studio/api/v2",
    );
  }
  return base.endsWith("/") ? base.slice(0, -1) : base;
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

export const ohi = {
  verify: async (req: VerifyRequest, opts: RequestOptions = {}): Promise<DocumentVerdict> => {
    const res = await fetch(`${apiBase()}/verify`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "application/json" },
      body: JSON.stringify(req),
      signal: opts.signal,
    });
    return asJson<DocumentVerdict>(res);
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
    const url = new URL("/health/deep", apiBase()).toString();
    const res = await fetch(url, {
      headers: { Accept: "application/json" },
      signal: opts.signal,
    });
    return asJson<HealthDeep>(res);
  },
};

export type OhiClient = typeof ohi;
