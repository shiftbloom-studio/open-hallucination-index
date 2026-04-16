output "artifacts_bucket" {
  value = aws_s3_bucket.artifacts.bucket
}

output "artifacts_bucket_arn" {
  value = aws_s3_bucket.artifacts.arn
}

output "artifacts_public_bucket" {
  value = aws_s3_bucket.artifacts_public.bucket
}

output "artifacts_public_url" {
  description = "Base URL for publicly-accessible calibration artifacts."
  value       = "https://${aws_s3_bucket.artifacts_public.bucket}.s3.${var.region}.amazonaws.com"
}
