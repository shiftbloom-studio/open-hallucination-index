data "aws_iam_policy_document" "lambda_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["lambda.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "lambda_exec" {
  name               = "${local.prefix}-api-exec"
  assume_role_policy = data.aws_iam_policy_document.lambda_assume.json
}

# Basic CloudWatch Logs permissions
resource "aws_iam_role_policy_attachment" "lambda_basic" {
  role       = aws_iam_role.lambda_exec.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

# Secrets read policy (all 8 secrets, KMS Decrypt scoped)
data "aws_iam_policy_document" "secrets_read" {
  statement {
    sid       = "ReadAllOhiSecrets"
    actions   = ["secretsmanager:GetSecretValue", "secretsmanager:DescribeSecret"]
    resources = values(local.secret_arns)
  }
  statement {
    sid       = "DecryptViaSecretsManager"
    actions   = ["kms:Decrypt"]
    resources = [data.aws_kms_alias.ohi_secrets.target_key_arn]
    condition {
      test     = "StringEquals"
      variable = "kms:ViaService"
      values   = ["secretsmanager.${var.region}.amazonaws.com"]
    }
  }
}

resource "aws_iam_role_policy" "lambda_secrets" {
  name   = "${local.prefix}-api-secrets"
  role   = aws_iam_role.lambda_exec.id
  policy = data.aws_iam_policy_document.secrets_read.json
}

# S3 artifacts bucket read/write (Lambda writes calibration, reads NLI heads)
data "aws_iam_policy_document" "artifacts_rw" {
  statement {
    actions = ["s3:GetObject", "s3:PutObject", "s3:ListBucket", "s3:DeleteObject"]
    resources = [
      "arn:aws:s3:::${local.artifacts_bucket}",
      "arn:aws:s3:::${local.artifacts_bucket}/*",
    ]
  }
}

resource "aws_iam_role_policy" "lambda_artifacts" {
  name   = "${local.prefix}-api-artifacts"
  role   = aws_iam_role.lambda_exec.id
  policy = data.aws_iam_policy_document.artifacts_rw.json
}
