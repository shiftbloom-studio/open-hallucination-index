output "verify_jobs_table_name" {
  description = "DynamoDB table name consumed by compute/ as the Lambda env var JOBS_TABLE_NAME."
  value       = aws_dynamodb_table.verify_jobs.name
}

output "verify_jobs_table_arn" {
  description = "DynamoDB table ARN used by compute/ to scope the Lambda IAM policy to this table only."
  value       = aws_dynamodb_table.verify_jobs.arn
}
