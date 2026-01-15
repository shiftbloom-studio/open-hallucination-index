// eslint-disable-next-line @typescript-eslint/no-unused-vars
import { z } from "zod";

// --- Types tailored to Python Pydantic Models ---

// From ref/requests.py
export interface VerifyTextRequest {
  text: string;
  context?: string | null;
  strategy?: "graph_exact" | "vector_semantic" | "hybrid" | "cascading" | "mcp_enhanced" | "adaptive" | null;
  use_cache?: boolean;
  target_sources?: number;
}

export interface BatchVerifyRequest {
  texts: string[];
  strategy?: "graph_exact" | "vector_semantic" | "hybrid" | "cascading" | "mcp_enhanced" | "adaptive" | null;
  use_cache?: boolean;
}

// From ref/health.py
export interface HealthStatus {
  status: "healthy" | "degraded" | "unhealthy";
  timestamp: string; // datetime iso format
  version: string;
  environment: string;
  checks: Record<string, boolean>;
}

export interface ReadinessStatus {
  ready: boolean;
  timestamp: string;
  services: Record<string, Record<string, boolean | string>>;
}

// From ref/responses.py & ref/verification.py
export type VerificationStatus = "verified" | "refuted" | "unverified" | string; // Implied from usage, fallback to string

export interface ClaimSummary {
  id: string; // UUID
  text: string;
  status: VerificationStatus;
  confidence: number;
  reasoning: string;
}

// TrustScore structure matching Python backend (domain/results.py)
export interface TrustScore {
  overall: number;
  claims_total: number;
  claims_supported: number;
  claims_refuted: number;
  claims_unverifiable: number;
  confidence: number;
  scoring_method: string;
  // Convenience getter for backward compatibility
  score?: number;
}

export interface VerifyTextResponse {
  id: string; // UUID
  trust_score: TrustScore;
  summary: string | null;
  claims: ClaimSummary[];
  processing_time_ms: number;
  cached: boolean;
}

export interface BatchVerifyResponse {
  results: VerifyTextResponse[];
  total_processing_time_ms: number;
}

// --- API Client ---

export class ApiClient {
  constructor(private baseUrl: string = "/api/ohi") {
    // Ensure no trailing slash
    this.baseUrl = baseUrl.replace(/\/$/, "");
  }

  private getUrl(path: string): string {
    // Ensure path starts with slash
    const cleanPath = path.startsWith("/") ? path : `/${path}`;
    return `${this.baseUrl}${cleanPath}`;
  }

  async getHealth(): Promise<HealthStatus> {
    const res = await fetch(this.getUrl("/health/live"), {
      method: "GET",
      headers: { "Content-Type": "application/json" },
    });
    if (!res.ok) throw new Error(`Health check failed: ${res.statusText}`);
    return res.json();
  }

  async verifyText(request: VerifyTextRequest): Promise<VerifyTextResponse> {
    const res = await fetch(this.getUrl("/api/v1/verify"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request),
    });
    if (!res.ok) {
      const errorText = await res.text();
      throw new Error(`Verification failed: ${res.status} ${errorText}`);
    }
    return res.json();
  }

  async verifyBatch(request: BatchVerifyRequest): Promise<BatchVerifyResponse> {
    const res = await fetch(this.getUrl("/api/v1/verify/batch"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request),
    });
    if (!res.ok)
      throw new Error(`Batch verification failed: ${res.statusText}`);
    return res.json();
  }
}

// Helper to create client instance
export const createApiClient = (baseUrl: string = "/api/ohi") =>
  new ApiClient(baseUrl);
