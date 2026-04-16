variable "region" {
  description = "AWS region for workload resources (not CloudFront — we don't use CloudFront)."
  type        = string
  default     = "eu-central-1"
}

variable "github_org" {
  description = "GitHub org (owner) for OIDC subject claims."
  type        = string
}

variable "github_repo" {
  description = "GitHub repo name for OIDC subject claims."
  type        = string
}

variable "github_branch_pattern" {
  description = "Branch/tag pattern the OIDC-assumable role accepts. `ref:refs/heads/*` + `ref:refs/tags/v*`."
  type        = list(string)
  default = [
    "repo:%s/%s:ref:refs/heads/main",
    "repo:%s/%s:ref:refs/heads/develop",
    "repo:%s/%s:ref:refs/tags/v*",
    "repo:%s/%s:pull_request",
  ]
}
