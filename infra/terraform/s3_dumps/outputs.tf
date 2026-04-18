output "dumps_bucket" {
  value       = aws_s3_bucket.dumps.bucket
  description = "S3 bucket name for cached Wikimedia dump files."
}

output "dumps_bucket_arn" {
  value       = aws_s3_bucket.dumps.arn
  description = "ARN of the dump-cache bucket; used by Stream I's IAM policy."
}
