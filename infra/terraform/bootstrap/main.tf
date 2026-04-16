data "aws_caller_identity" "current" {}

locals {
  account_id = data.aws_caller_identity.current.account_id
  prefix     = module.shared.name_prefix
}

# ---------------------------------------------------------------------------
# KMS key used by the state bucket AND Secrets Manager
# ---------------------------------------------------------------------------
resource "aws_kms_key" "ohi_secrets" {
  description             = "OHI state + secrets KMS CMK"
  deletion_window_in_days = 30
  enable_key_rotation     = true
}

resource "aws_kms_alias" "ohi_secrets" {
  name          = "alias/${local.prefix}-secrets"
  target_key_id = aws_kms_key.ohi_secrets.key_id
}

# ---------------------------------------------------------------------------
# Terraform state S3 bucket (SSE-KMS, versioned, block public)
# ---------------------------------------------------------------------------
resource "aws_s3_bucket" "tfstate" {
  bucket = "${local.prefix}-tfstate-${local.account_id}"
}

resource "aws_s3_bucket_versioning" "tfstate" {
  bucket = aws_s3_bucket.tfstate.id
  versioning_configuration { status = "Enabled" }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "tfstate" {
  bucket = aws_s3_bucket.tfstate.id
  rule {
    apply_server_side_encryption_by_default {
      kms_master_key_id = aws_kms_key.ohi_secrets.arn
      sse_algorithm     = "aws:kms"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "tfstate" {
  bucket                  = aws_s3_bucket.tfstate.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# ---------------------------------------------------------------------------
# DynamoDB state lock table
# ---------------------------------------------------------------------------
resource "aws_dynamodb_table" "tfstate_lock" {
  name         = "${local.prefix}-tfstate-lock"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "LockID"

  attribute {
    name = "LockID"
    type = "S"
  }
}

# ---------------------------------------------------------------------------
# ECR repository for the Lambda container image
# ---------------------------------------------------------------------------
resource "aws_ecr_repository" "ohi_api" {
  name                 = "${local.prefix}-api"
  image_tag_mutability = "MUTABLE" # `prod` tag moves across releases

  image_scanning_configuration {
    scan_on_push = true
  }

  encryption_configuration {
    encryption_type = "KMS"
    kms_key         = aws_kms_key.ohi_secrets.arn
  }
}

resource "aws_ecr_lifecycle_policy" "ohi_api" {
  repository = aws_ecr_repository.ohi_api.name
  policy = jsonencode({
    rules = [
      {
        rulePriority = 1
        description  = "Keep last 5 tagged images"
        selection = {
          tagStatus      = "tagged"
          tagPatternList = ["v*", "prod", "stub"]
          countType      = "imageCountMoreThan"
          countNumber    = 5
        }
        action = { type = "expire" }
      },
      {
        rulePriority = 2
        description  = "Expire untagged images after 1 day"
        selection = {
          tagStatus   = "untagged"
          countType   = "sinceImagePushed"
          countUnit   = "days"
          countNumber = 1
        }
        action = { type = "expire" }
      },
    ]
  })
}
