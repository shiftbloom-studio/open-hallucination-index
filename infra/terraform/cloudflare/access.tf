# Single service token issued to Lambda; reused across all 4 tunnel apps.
resource "cloudflare_zero_trust_access_service_token" "lambda" {
  account_id = local.account_id
  name       = "ohi-lambda-tunnel-service-token"
  duration   = "8760h" # 1 year; we'll rotate manually in a runbook
}

# Access application + policy per tunnel hostname
locals {
  tunnel_access_apps = {
    neo4j  = "neo4j.${var.zone_name}"
    qdrant = "qdrant.${var.zone_name}"
    pg     = "pg.${var.zone_name}"
    redis  = "redis.${var.zone_name}"
  }
}

resource "cloudflare_zero_trust_access_application" "tunnel" {
  for_each = local.tunnel_access_apps

  account_id                = local.account_id
  name                      = "OHI tunnel — ${each.key}"
  domain                    = each.value
  type                      = "self_hosted"
  session_duration          = "24h"
  auto_redirect_to_identity = false
}

resource "cloudflare_zero_trust_access_policy" "tunnel_service_token" {
  for_each = local.tunnel_access_apps

  account_id     = local.account_id
  application_id = cloudflare_zero_trust_access_application.tunnel[each.key].id
  name           = "Lambda service token — ${each.key}"
  precedence     = 1
  decision       = "non_identity"
  include {
    service_token = [cloudflare_zero_trust_access_service_token.lambda.id]
  }
}

# Write the service token client_id + client_secret to Secrets Manager as JSON.
resource "aws_secretsmanager_secret_version" "cf_access_service_token" {
  secret_id = local.secret_arns["cf_access_service_token"]
  secret_string = jsonencode({
    client_id     = cloudflare_zero_trust_access_service_token.lambda.client_id
    client_secret = cloudflare_zero_trust_access_service_token.lambda.client_secret
  })
}
