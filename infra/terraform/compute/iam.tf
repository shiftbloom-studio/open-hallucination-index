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

# ---------------------------------------------------------------------------
# Stream D2: DynamoDB jobs table R/W + Lambda self-async-invoke permission.
#
# POST /api/v2/verify creates a DynamoDB record, self-async-invokes, and
# returns 202. The async handler reads the record, runs pipeline.verify(),
# and updates the record at each of the five phase boundaries. IAM is
# scoped narrowly to the single table and to the Lambda's own ARN so this
# permission set cannot be reused to invoke other Lambdas or read other
# tables if the role were compromised.
# ---------------------------------------------------------------------------
data "aws_iam_policy_document" "jobs_rw" {
  statement {
    sid = "VerifyJobsReadWrite"
    actions = [
      "dynamodb:PutItem",
      "dynamodb:GetItem",
      "dynamodb:UpdateItem",
    ]
    resources = [local.jobs_table_arn]
  }
}

resource "aws_iam_role_policy" "lambda_jobs" {
  name   = "${local.prefix}-api-jobs"
  role   = aws_iam_role.lambda_exec.id
  policy = data.aws_iam_policy_document.jobs_rw.json
}

# Self-async-invoke. `aws_lambda_function.api.arn` is a self-reference, so
# Terraform orders this policy AFTER the Lambda is created. We intentionally
# do NOT add this resource to `aws_lambda_function.api.depends_on` — doing
# so would create a cycle. The async-invoke path is exercised only when the
# first /verify request arrives, well after apply completes, so the brief
# window between "Lambda created" and "self-invoke policy attached" is
# harmless. A defensive `depth <= 1` guard in the handler caps worst-case
# runaway recursion to 2 invocations per request if the policy ever
# regresses.
data "aws_iam_policy_document" "lambda_self_invoke" {
  statement {
    sid       = "SelfAsyncInvoke"
    actions   = ["lambda:InvokeFunction"]
    resources = [aws_lambda_function.api.arn]
  }
}

resource "aws_iam_role_policy" "lambda_self_invoke" {
  name   = "${local.prefix}-api-self-invoke"
  role   = aws_iam_role.lambda_exec.id
  policy = data.aws_iam_policy_document.lambda_self_invoke.json
}
