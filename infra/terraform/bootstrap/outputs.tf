output "state_bucket" {
  description = "S3 bucket holding all layer state files."
  value       = aws_s3_bucket.tfstate.bucket
}

output "state_lock_table" {
  description = "DynamoDB table for state locking."
  value       = aws_dynamodb_table.tfstate_lock.name
}

output "kms_key_arn" {
  description = "CMK for state bucket + Secrets Manager."
  value       = aws_kms_key.ohi_secrets.arn
}

output "kms_key_alias" {
  description = "KMS alias that other layers reference."
  value       = aws_kms_alias.ohi_secrets.name
}

output "ecr_repository_url" {
  description = "ECR repo URL for pushing Lambda images."
  value       = aws_ecr_repository.ohi_api.repository_url
}

output "ecr_repository_name" {
  value = aws_ecr_repository.ohi_api.name
}

output "github_oidc_provider_arn" {
  value = aws_iam_openid_connect_provider.github.arn
}

output "terraform_apply_role_arn" {
  description = "Store this in GitHub repo vars as AWS_ROLE_ARN."
  value       = aws_iam_role.terraform_apply.arn
}

output "terraform_drift_role_arn" {
  description = "Store this in GitHub repo vars as AWS_DRIFT_ROLE_ARN."
  value       = aws_iam_role.terraform_drift.arn
}

output "aws_region" {
  description = "Store this in GitHub repo vars as AWS_REGION."
  value       = var.region
}

output "aws_account_id" {
  value = data.aws_caller_identity.current.account_id
}
