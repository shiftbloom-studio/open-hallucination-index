locals {
  secret_names = {
    gemini_api_key           = "ohi/gemini-api-key"
    internal_bearer_token    = "ohi/internal-bearer-token"
    cloudflared_tunnel_token = "ohi/cloudflared-tunnel-token"
    cf_access_service_token  = "ohi/cf-access-service-token"
    cf_edge_secret           = "ohi/cf-edge-secret"
    labeler_tokens           = "ohi/labeler-tokens"
    pc_origin_credentials    = "ohi/pc-origin-credentials"
    neo4j_credentials        = "ohi/neo4j-credentials"
  }
}

resource "aws_secretsmanager_secret" "this" {
  for_each = local.secret_names

  name                    = each.value
  kms_key_id              = data.aws_kms_alias.ohi_secrets.target_key_arn
  recovery_window_in_days = 0 # immediate delete; single-env accepted risk

  tags = {
    SecretRole = each.key
  }
}
