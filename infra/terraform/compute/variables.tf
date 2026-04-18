variable "region" {
  type    = string
  default = "eu-central-1"
}

variable "image_tag" {
  description = "ECR image tag to deploy. Defaults to the moving `prod` tag. CI release workflow passes the semver tag."
  type        = string
  default     = "prod"
}

variable "memory_mb" {
  type    = number
  default = 2048
}

variable "timeout_s" {
  description = "Lambda function timeout. Bumped 60->180 for Phase 2 D1: a 5-claim x 3-evidence document issues up to 15 Gemini 3 Pro NLI calls with thinkingLevel=HIGH (~10-20s each). Semaphore(10) in pipeline caps concurrency at 10 in-flight, so max wall-clock ~= ceil(n_pairs/10) * ~20s. 180s covers the serial tail and matches the SSE /verify/stream path that Stream D2 will land (served via Lambda Function URL with RESPONSE_STREAM, which bypasses the HTTP API's 30s integration cap). Sync /verify still transits api_gateway.tf's 30000ms integration timeout — a slow long-document sync request will 504 at API Gateway even though Lambda keeps running; that's expected until D2 routes the slow path through streaming."
  type        = number
  default     = 180
}

variable "nli_llm_model" {
  description = "Gemini model id used by the Phase 2 LLM-based NLI adapter. Read by NLISettings as NLI_LLM_MODEL. Default preview; flip to gemini-2.5-pro GA if preview is flaky."
  type        = string
  default     = "gemini-3-pro-preview"
}

variable "nli_thinking_level" {
  description = "Gemini 3 thinking budget for NLI. Plumbed to the env for future tuning; GeminiLLMAdapter intrinsically sets thinkingLevel=HIGH for any gemini-3* model today."
  type        = string
  default     = "HIGH"
}

variable "nli_self_consistency_k" {
  description = "Majority-vote samples per (claim, evidence) NLI classification. K=1 disables self-consistency. Flipping to K>1 multiplies Gemini spend by K (plan §6.2 G6 gate)."
  type        = number
  default     = 1
}

variable "log_retention_days" {
  type    = number
  default = 7
}

variable "tunnel_hostname_neo4j" {
  type    = string
  default = "ohi-neo4j.shiftbloom.studio"
}

variable "tunnel_hostname_qdrant" {
  type    = string
  default = "ohi-qdrant.shiftbloom.studio"
}

variable "tunnel_hostname_pg_rest" {
  type    = string
  default = "ohi-pg.shiftbloom.studio"
}

variable "tunnel_hostname_webdis" {
  type    = string
  default = "ohi-redis.shiftbloom.studio"
}

variable "tunnel_hostname_embed" {
  description = "Hostname of the PC-side embedding service via the CF tunnel."
  type        = string
  default     = "ohi-embed.shiftbloom.studio"
}

variable "embedding_backend" {
  description = "'local' (in-process sentence-transformers) or 'remote' (HTTP to pc-embed)."
  type        = string
  default     = "remote"
}

variable "neo4j_uri" {
  description = "Neo4j connection URI. Aura Pro Frankfurt instance — neo4j+s://<id>.databases.neo4j.io"
  type        = string
  default     = "neo4j+s://0193408e.databases.neo4j.io"
}

variable "gemini_model" {
  type    = string
  default = "gemini-3-flash-preview"
}

variable "gemini_daily_ceiling_eur" {
  description = "0 = unlimited (Phase 1 per spec §9.1 R10)"
  type        = number
  default     = 0
}

variable "cors_origins" {
  description = "Comma-separated list of allowed CORS origins. Production value is the Vercel-hosted frontend apex."
  type        = string
  default     = "https://ohi.shiftbloom.studio"
}

variable "async_verify_ttl_seconds" {
  description = "Seconds a DynamoDB verify-job record survives after creation before TTL reaps it. 3600 (1h) comfortably covers the ~3min frontend polling cap plus a post-mortem debugging margin. Stream D2."
  type        = number
  default     = 3600
}
