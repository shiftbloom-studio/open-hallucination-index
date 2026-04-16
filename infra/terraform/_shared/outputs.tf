output "tags" {
  description = "Default tag map applied via provider default_tags in each layer."
  value = {
    Project     = var.project
    Environment = var.environment
    Layer       = var.layer
    ManagedBy   = "terraform"
    CostCenter  = var.project
  }
}

output "name_prefix" {
  description = "Resource name prefix. Single-env means no env suffix."
  value       = var.project
}

output "region" {
  description = "Workload region."
  value       = var.region
}

output "project" {
  value = var.project
}

output "environment" {
  value = var.environment
}
