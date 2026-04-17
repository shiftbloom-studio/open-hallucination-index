variable "region" {
  type    = string
  default = "eu-central-1"
}

variable "project_name" {
  description = "Vercel project name (slug). Appears in dashboard + default URLs."
  type        = string
  default     = "ohi-frontend"
}

variable "github_org" {
  type    = string
  default = "shiftbloom-studio"
}

variable "github_repo" {
  type    = string
  default = "open-hallucination-index"
}

variable "production_branch" {
  description = "Branch that triggers production deployments."
  type        = string
  default     = "main"
}

variable "root_directory" {
  description = "Vercel's project root inside the monorepo (matches Phase 3 plan)."
  type        = string
  default     = "src/frontend"
}

variable "apex_domain" {
  description = "Public apex served by Vercel."
  type        = string
  default     = "ohi.shiftbloom.studio"
}

variable "api_base_url" {
  description = "Absolute cross-origin URL the browser hits for API calls."
  type        = string
  default     = "https://api.ohi.shiftbloom.studio/api/v2"
}

variable "vercel_team_id" {
  description = "Vercel team ID. Empty string if using a personal account."
  type        = string
  default     = ""
}
