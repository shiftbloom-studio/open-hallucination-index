locals {
  secret_names = {
    gemini_api_key           = "ohi/gemini-api-key"
    openai_api_key           = "ohi/openai-api-key"
    internal_bearer_token    = "ohi/internal-bearer-token"
    cloudflared_tunnel_token = "ohi/cloudflared-tunnel-token"
    cf_access_service_token  = "ohi/cf-access-service-token"
    cf_edge_secret           = "ohi/cf-edge-secret"
    labeler_tokens           = "ohi/labeler-tokens"
    pc_origin_credentials    = "ohi/pc-origin-credentials"
    neo4j_credentials        = "ohi/neo4j-credentials"
    # Phase 2 of Neo4j Aura -> PC-Tailscale migration. Holds a reusable
    # + ephemeral Tailscale auth key generated in the Tailscale admin
    # UI; consumed by docker/lambda/tsproxy at Lambda cold start. The
    # secret is created empty by TF; the value is populated out-of-band
    # via `aws secretsmanager put-secret-value`.
    tailscale_authkey        = "ohi/tailscale-authkey"
  }
}
# Wave 3 Stream P (Decision H): ``ohi/openai-api-key`` was created
# directly via AWS CLI during the Wave 3 autonomous deploy
# (account 349744179866, region eu-central-1, name suffix ``-Cwjs76``)
# so the adapter could deploy without blocking on a separate TF apply.
# To bring it under TF state, run:
#   cd infra/terraform/secrets
#   terraform import \
#     'aws_secretsmanager_secret.this["openai_api_key"]' \
#     'arn:aws:secretsmanager:eu-central-1:349744179866:secret:ohi/openai-api-key-Cwjs76'
# after which subsequent ``terraform apply`` runs will manage the secret
# metadata (tags, KMS) without touching the secret value. The value
# itself stays out of TF state — rotate via ``aws secretsmanager
# put-secret-value`` or the rotate-secret runbook.

resource "aws_secretsmanager_secret" "this" {
  for_each = local.secret_names

  name                    = each.value
  kms_key_id              = data.aws_kms_alias.ohi_secrets.target_key_arn
  recovery_window_in_days = 0 # immediate delete; single-env accepted risk

  tags = {
    SecretRole = each.key
  }
}
