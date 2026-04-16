# ---------------------------------------------------------------------------
# Private artifacts bucket (NLI heads, calibration, retraining reports)
# ---------------------------------------------------------------------------
resource "aws_s3_bucket" "artifacts" {
  bucket = "${local.prefix}-artifacts-${local.account_id}"
}

resource "aws_s3_bucket_versioning" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id
  versioning_configuration { status = "Enabled" }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id
  rule {
    apply_server_side_encryption_by_default { sse_algorithm = "AES256" }
  }
}

resource "aws_s3_bucket_public_access_block" "artifacts" {
  bucket                  = aws_s3_bucket.artifacts.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_lifecycle_configuration" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id

  rule {
    id     = "expire-old-retraining-reports"
    status = "Enabled"
    filter { prefix = "retraining-reports/" }
    expiration { days = 365 }
  }

  rule {
    id     = "expire-old-eval-snapshots"
    status = "Enabled"
    filter { prefix = "eval-snapshots/" }
    expiration { days = 90 }
  }
}

# ---------------------------------------------------------------------------
# Public calibration bucket — open-source transparency
# ---------------------------------------------------------------------------
resource "aws_s3_bucket" "artifacts_public" {
  bucket = "${local.prefix}-artifacts-public-${local.account_id}"
}

resource "aws_s3_bucket_public_access_block" "artifacts_public" {
  bucket = aws_s3_bucket.artifacts_public.id

  block_public_acls       = false
  block_public_policy     = false
  ignore_public_acls      = false
  restrict_public_buckets = false
}

resource "aws_s3_bucket_policy" "artifacts_public" {
  bucket     = aws_s3_bucket.artifacts_public.id
  depends_on = [aws_s3_bucket_public_access_block.artifacts_public]

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid       = "PublicRead"
        Effect    = "Allow"
        Principal = "*"
        Action    = ["s3:GetObject"]
        Resource  = ["${aws_s3_bucket.artifacts_public.arn}/*"]
      },
    ]
  })
}

resource "aws_s3_bucket_cors_configuration" "artifacts_public" {
  bucket = aws_s3_bucket.artifacts_public.id

  cors_rule {
    allowed_methods = ["GET", "HEAD"]
    allowed_origins = ["*"]
    allowed_headers = ["*"]
    max_age_seconds = 86400
  }
}
