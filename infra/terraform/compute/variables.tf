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
  type    = number
  default = 60
}

variable "log_retention_days" {
  type    = number
  default = 7
}

variable "tunnel_hostname_neo4j" {
  type    = string
  default = "neo4j.ohi.shiftbloom.studio"
}

variable "tunnel_hostname_qdrant" {
  type    = string
  default = "qdrant.ohi.shiftbloom.studio"
}

variable "tunnel_hostname_pg_rest" {
  type    = string
  default = "pg.ohi.shiftbloom.studio"
}

variable "tunnel_hostname_webdis" {
  type    = string
  default = "redis.ohi.shiftbloom.studio"
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
