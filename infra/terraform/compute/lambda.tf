data "aws_ecr_image" "api" {
  repository_name = data.aws_ecr_repository.api.name
  image_tag       = var.image_tag
}

# Neo4j credentials read from AWS Secrets Manager and surfaced to Lambda as
# NEO4J_PASSWORD env var. The in-code SecretsLoader is not wired up for the
# Neo4j adapter today (infra_env.neo4j_credentials_secret_arn() is defined
# but never called), so the adapter relies on pydantic-settings reading
# NEO4J_USERNAME / NEO4J_PASSWORD from the environment.
data "aws_secretsmanager_secret_version" "neo4j" {
  secret_id = local.secret_arns["neo4j_credentials"]
}

# CF Access service-token creds read from AWS Secrets Manager (written by
# cloudflare/ layer when the service token resource is created). Surfaced
# to Lambda as OHI_CF_ACCESS_CLIENT_ID / OHI_CF_ACCESS_CLIENT_SECRET so
# tunnel-proxied adapters (embed, qdrant, ...) can authenticate against
# the CF Access apps gating the tunnel.
data "aws_secretsmanager_secret_version" "cf_access" {
  secret_id = local.secret_arns["cf_access_service_token"]
}

# Gemini API key (raw value) for the LLM adapter, which uses Gemini's
# OpenAI-compatible endpoint. Surfaced as LLM_API_KEY so pydantic-settings
# (env_prefix="LLM_") picks it up without any code change.
data "aws_secretsmanager_secret_version" "gemini" {
  secret_id = local.secret_arns["gemini_api_key"]
}

resource "aws_lambda_function" "api" {
  function_name = "${local.prefix}-api"
  role          = aws_iam_role.lambda_exec.arn

  package_type = "Image"
  image_uri    = "${data.aws_ecr_repository.api.repository_url}@${data.aws_ecr_image.api.image_digest}"

  memory_size = var.memory_mb
  timeout     = var.timeout_s

  environment {
    variables = {
      OHI_ENV                        = "prod"
      OHI_REGION                     = var.region
      OHI_LOG_LEVEL                  = "INFO"
      OHI_GEMINI_MODEL               = var.gemini_model
      OHI_GEMINI_DAILY_CEILING_EUR   = tostring(var.gemini_daily_ceiling_eur)
      OHI_CORS_ORIGINS               = var.cors_origins
      OHI_CF_TUNNEL_HOSTNAME_NEO4J   = var.tunnel_hostname_neo4j
      OHI_CF_TUNNEL_HOSTNAME_QDRANT  = var.tunnel_hostname_qdrant
      OHI_CF_TUNNEL_HOSTNAME_PG_REST = var.tunnel_hostname_pg_rest
      OHI_CF_TUNNEL_HOSTNAME_WEBDIS  = var.tunnel_hostname_webdis
      OHI_CF_TUNNEL_HOSTNAME_EMBED   = var.tunnel_hostname_embed
      OHI_EMBEDDING_BACKEND          = var.embedding_backend
      OHI_EMBEDDING_REMOTE_URL       = "https://${var.tunnel_hostname_embed}"
      OHI_S3_ARTIFACTS_BUCKET        = local.artifacts_bucket

      # Neo4j connection — Aura-hosted (neo4j+s://...)
      NEO4J_URI      = var.neo4j_uri
      NEO4J_USERNAME = jsondecode(data.aws_secretsmanager_secret_version.neo4j.secret_string)["username"]
      NEO4J_PASSWORD = jsondecode(data.aws_secretsmanager_secret_version.neo4j.secret_string)["password"]

      # Qdrant — reached via CF tunnel on HTTPS (port 443 at CF edge, routed
      # to http://qdrant:6333 on the PC). CF Access gates the hostname with
      # a service-token app; Lambda sends the headers below.
      QDRANT_HOST  = var.tunnel_hostname_qdrant
      QDRANT_PORT  = "443"
      QDRANT_HTTPS = "true"

      # CF Access service token — forwarded by tunnel-proxied adapters
      # (adapters/qdrant.py, adapters/embeddings.py remote mode, ...).
      OHI_CF_ACCESS_CLIENT_ID     = jsondecode(data.aws_secretsmanager_secret_version.cf_access.secret_string)["client_id"]
      OHI_CF_ACCESS_CLIENT_SECRET = jsondecode(data.aws_secretsmanager_secret_version.cf_access.secret_string)["client_secret"]

      # Redis disabled for MVP — webdis over CF tunnel doesn't speak native
      # Redis protocol; proper solution is managed Redis (ElastiCache/Upstash).
      REDIS_ENABLED = "false"

      # LLM adapter points at Gemini's OpenAI-compatible endpoint. The adapter
      # (src/api/adapters/openai.py via pydantic-settings LLM_*) reads these.
      LLM_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
      LLM_API_KEY  = data.aws_secretsmanager_secret_version.gemini.secret_string
      LLM_MODEL    = var.gemini_model

      # Phase 2 LLM-based NLI (D1 wire). A dedicated NliGeminiAdapter runs on
      # a second GeminiLLMAdapter instance with this model, reusing LLM_API_KEY
      # via pydantic-settings in dependencies.py (model_copy with model override).
      # Self-consistency is off by default (K=1) — G6 gate before flipping to 3.
      NLI_LLM_MODEL          = var.nli_llm_model
      NLI_THINKING_LEVEL     = var.nli_thinking_level
      NLI_SELF_CONSISTENCY_K = tostring(var.nli_self_consistency_k)

      # Secret ARNs (values fetched at runtime via SecretsLoader)
      OHI_GEMINI_KEY_SECRET_ARN               = local.secret_arns["gemini_api_key"]
      OHI_INTERNAL_BEARER_SECRET_ARN          = local.secret_arns["internal_bearer_token"]
      OHI_CF_EDGE_SECRET_ARN                  = local.secret_arns["cf_edge_secret"]
      OHI_CF_ACCESS_SERVICE_TOKEN_SECRET_ARN  = local.secret_arns["cf_access_service_token"]
      OHI_CLOUDFLARED_TUNNEL_TOKEN_SECRET_ARN = local.secret_arns["cloudflared_tunnel_token"]
      OHI_LABELER_TOKENS_SECRET_ARN           = local.secret_arns["labeler_tokens"]
      OHI_PC_ORIGIN_CREDENTIALS_SECRET_ARN    = local.secret_arns["pc_origin_credentials"]
      OHI_NEO4J_CREDENTIALS_SECRET_ARN        = local.secret_arns["neo4j_credentials"]
    }
  }

  logging_config {
    log_format = "JSON"
    log_group  = aws_cloudwatch_log_group.api.name
  }

  depends_on = [
    aws_iam_role_policy_attachment.lambda_basic,
    aws_iam_role_policy.lambda_secrets,
    aws_iam_role_policy.lambda_artifacts,
    aws_cloudwatch_log_group.api,
  ]
}

resource "aws_lambda_function_url" "api" {
  function_name      = aws_lambda_function.api.function_name
  authorization_type = "NONE"
  invoke_mode        = "RESPONSE_STREAM"

  # Lambda Function URL CORS is permissive ("*"); the precise origin lockdown
  # lives in FastAPI's CORSMiddleware (via OHI_CORS_ORIGINS). Function URL's
  # allow_methods validator rejects "OPTIONS" (>6 chars) so use "*".
  cors {
    allow_origins = ["*"]
    allow_methods = ["*"]
    allow_headers = ["*"]
    max_age       = 86400
  }
}

# auth_type=NONE doesn't automatically grant public invocation; an explicit
# Lambda resource-based policy is required. EdgeSecretMiddleware is what
# actually gates requests — this just allows the anonymous HTTP→Lambda hop.
resource "aws_lambda_permission" "public_url" {
  statement_id           = "FunctionURLAllowPublicAccess"
  action                 = "lambda:InvokeFunctionUrl"
  function_name          = aws_lambda_function.api.function_name
  principal              = "*"
  function_url_auth_type = "NONE"
}
