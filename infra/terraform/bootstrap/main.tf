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

# ---------------------------------------------------------------------------
# GitHub OIDC provider (one per account; idempotent)
# ---------------------------------------------------------------------------
# Modern AWS IAM auto-validates the cert chain for token.actions.githubusercontent.com,
# so thumbprint_list is not strictly required, but providing one satisfies older
# providers. AWS ignores it on modern accounts.
resource "aws_iam_openid_connect_provider" "github" {
  url             = "https://token.actions.githubusercontent.com"
  client_id_list  = ["sts.amazonaws.com"]
  thumbprint_list = ["6938fd4d98bab03faadb97b34396831e3780aea1"]
}

# ---------------------------------------------------------------------------
# Main apply role — assumed by CI for plan/apply. NOT admin.
# ---------------------------------------------------------------------------
locals {
  oidc_sub_patterns = [for pat in var.github_branch_pattern : format(pat, var.github_org, var.github_repo)]
}

data "aws_iam_policy_document" "apply_trust" {
  statement {
    actions = ["sts:AssumeRoleWithWebIdentity"]
    principals {
      type        = "Federated"
      identifiers = [aws_iam_openid_connect_provider.github.arn]
    }
    condition {
      test     = "StringEquals"
      variable = "token.actions.githubusercontent.com:aud"
      values   = ["sts.amazonaws.com"]
    }
    condition {
      test     = "StringLike"
      variable = "token.actions.githubusercontent.com:sub"
      values   = local.oidc_sub_patterns
    }
  }
}

resource "aws_iam_role" "terraform_apply" {
  name                 = "${local.prefix}-terraform-apply"
  assume_role_policy   = data.aws_iam_policy_document.apply_trust.json
  max_session_duration = 3600
}

# Scoped inline policy — wide enough to manage all layers, narrow enough to not be admin.
data "aws_iam_policy_document" "apply_policy" {
  statement {
    sid = "TerraformStateBucket"
    actions = [
      "s3:GetObject", "s3:PutObject", "s3:DeleteObject", "s3:ListBucket",
    ]
    resources = [
      aws_s3_bucket.tfstate.arn,
      "${aws_s3_bucket.tfstate.arn}/*",
    ]
  }

  statement {
    sid = "TerraformStateLock"
    actions = [
      "dynamodb:GetItem", "dynamodb:PutItem", "dynamodb:DeleteItem", "dynamodb:DescribeTable",
    ]
    resources = [aws_dynamodb_table.tfstate_lock.arn]
  }

  statement {
    sid = "KMSUse"
    actions = [
      "kms:Encrypt", "kms:Decrypt", "kms:ReEncrypt*", "kms:GenerateDataKey*", "kms:DescribeKey",
    ]
    resources = [aws_kms_key.ohi_secrets.arn]
  }

  statement {
    sid    = "LayerManagement"
    effect = "Allow"
    actions = [
      # Lambda
      "lambda:*",
      # IAM (scoped below)
      "iam:GetRole", "iam:PassRole", "iam:CreateRole", "iam:DeleteRole",
      "iam:AttachRolePolicy", "iam:DetachRolePolicy", "iam:PutRolePolicy",
      "iam:DeleteRolePolicy", "iam:GetRolePolicy", "iam:ListRolePolicies",
      "iam:ListAttachedRolePolicies", "iam:CreatePolicy", "iam:DeletePolicy",
      "iam:GetPolicy", "iam:GetPolicyVersion", "iam:ListPolicyVersions",
      "iam:CreatePolicyVersion", "iam:DeletePolicyVersion", "iam:TagRole",
      "iam:UntagRole", "iam:TagPolicy", "iam:UntagPolicy",
      # Secrets Manager
      "secretsmanager:*",
      # S3 artifact buckets
      "s3:*",
      # CloudWatch Logs + Metrics + Dashboard
      "logs:*", "cloudwatch:*",
      # SNS
      "sns:*",
      # Budgets
      "budgets:*",
      # ECR
      "ecr:*",
      # IAM OIDC (read-only; bootstrap owns the provider)
      "iam:GetOpenIDConnectProvider", "iam:ListOpenIDConnectProviders",
    ]
    resources = ["*"]
  }

  # NOTE: `iam:AttachRolePolicy` appears both in the broad allow above AND in
  # this deny. Because Deny always wins, the net effect is: CI can attach any
  # managed policy to roles EXCEPT AdministratorAccess. That's the desired
  # "no admin escalation" guardrail.
  statement {
    sid    = "DenyIAMUserAndAdminEscalation"
    effect = "Deny"
    actions = [
      "iam:CreateUser", "iam:DeleteUser", "iam:CreateAccessKey", "iam:DeleteAccessKey",
      "iam:AttachUserPolicy", "iam:PutUserPolicy",
      "iam:AttachRolePolicy",
    ]
    resources = ["*"]
    condition {
      test     = "ArnLike"
      variable = "iam:PolicyARN"
      values   = ["arn:aws:iam::aws:policy/AdministratorAccess"]
    }
  }
}

resource "aws_iam_role_policy" "terraform_apply" {
  name   = "${local.prefix}-terraform-apply-inline"
  role   = aws_iam_role.terraform_apply.id
  policy = data.aws_iam_policy_document.apply_policy.json
}

# ---------------------------------------------------------------------------
# Drift-check read-only role — used by bootstrap-drift.yml nightly workflow.
# ---------------------------------------------------------------------------
data "aws_iam_policy_document" "drift_trust" {
  statement {
    actions = ["sts:AssumeRoleWithWebIdentity"]
    principals {
      type        = "Federated"
      identifiers = [aws_iam_openid_connect_provider.github.arn]
    }
    condition {
      test     = "StringEquals"
      variable = "token.actions.githubusercontent.com:aud"
      values   = ["sts.amazonaws.com"]
    }
    condition {
      test     = "StringLike"
      variable = "token.actions.githubusercontent.com:sub"
      values   = [format("repo:%s/%s:ref:refs/heads/main", var.github_org, var.github_repo)]
    }
  }
}

resource "aws_iam_role" "terraform_drift" {
  name               = "${local.prefix}-terraform-drift"
  assume_role_policy = data.aws_iam_policy_document.drift_trust.json
}

resource "aws_iam_role_policy_attachment" "terraform_drift_readonly" {
  role       = aws_iam_role.terraform_drift.name
  policy_arn = "arn:aws:iam::aws:policy/ReadOnlyAccess"
}
