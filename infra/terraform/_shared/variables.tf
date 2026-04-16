variable "layer" {
  description = "Name of the layer consuming this shared module (for the Layer tag)."
  type        = string
  validation {
    condition     = contains(["bootstrap", "storage", "secrets", "compute", "cloudflare", "observability"], var.layer)
    error_message = "layer must be one of bootstrap, storage, secrets, compute, cloudflare, observability."
  }
}

variable "region" {
  description = "AWS region for workload resources."
  type        = string
  default     = "eu-central-1"
}

variable "project" {
  description = "Project short-name prefix for resources."
  type        = string
  default     = "ohi"
}

variable "environment" {
  description = "Environment short-name. Single-env design = prod."
  type        = string
  default     = "prod"
}
