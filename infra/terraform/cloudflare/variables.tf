variable "region" {
  type    = string
  default = "eu-central-1"
}

variable "zone_name" {
  description = "Cloudflare zone managed by this layer. Full-zone delegation ('shiftbloom.studio') with apex_subdomain='ohi' is the current topology."
  type        = string
  default     = "shiftbloom.studio"
}

variable "apex_subdomain" {
  description = "Subdomain under zone_name that serves the OHI frontend. Set to '' to use the zone apex directly."
  type        = string
  default     = "ohi"
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

variable "vercel_verification_token" {
  description = "Optional Vercel _vercel TXT value. Empty = skip record (CNAME-based verification usually sufficient)."
  type        = string
  default     = ""
}

variable "api_subdomain" {
  description = "Subdomain label for the API endpoint. Full hostname = <api_subdomain>.<apex_subdomain>.<zone_name> when apex_subdomain is set, else <api_subdomain>.<zone_name>."
  type        = string
  default     = "api"
}
