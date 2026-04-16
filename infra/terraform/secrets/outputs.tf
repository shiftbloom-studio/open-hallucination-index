output "secret_arns" {
  description = "Map of secret role -> ARN. Consumed by compute/ and cloudflare/ layers."
  value       = { for k, s in aws_secretsmanager_secret.this : k => s.arn }
}

output "secret_names" {
  description = "Map of secret role -> name."
  value       = { for k, s in aws_secretsmanager_secret.this : k => s.name }
}
