variable "region" {
  type    = string
  default = "eu-central-1"
}

variable "zone_name" {
  description = "Cloudflare zone managed by this layer (apex of delegated subdomain)."
  type        = string
  default     = "ohi.shiftbloom.studio"
}

variable "cf_account_id" {
  description = "Cloudflare account ID. Find at dash.cloudflare.com → right sidebar."
  type        = string
  sensitive   = false
}

variable "edge_secret" {
  description = "Shared secret that CF Transform Rule injects as X-OHI-Edge-Secret. Pass via -var or TF_VAR_edge_secret; NOT stored in tfvars."
  type        = string
  sensitive   = true
}

variable "rate_limit_verify_per_min" {
  type    = number
  default = 100
}

variable "rate_limit_global_per_hour" {
  type    = number
  default = 1000
}
