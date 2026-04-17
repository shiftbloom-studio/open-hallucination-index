# Cloudflare Transform Rule that injects X-OHI-Edge-Secret on every request to origin.
# Lambda's EdgeSecretMiddleware validates this header.
# The edge-secret VALUE comes from var.edge_secret (supplied by runbook / CI), and
# is ALSO written to AWS Secrets Manager so Lambda can fetch it at runtime.
resource "cloudflare_ruleset" "transform_edge_secret" {
  zone_id     = local.zone_id
  name        = "ohi-prod-transform-edge-secret"
  description = "Add X-OHI-Edge-Secret header to all requests going to origin"
  kind        = "zone"
  phase       = "http_request_transform"

  rules {
    action      = "rewrite"
    description = "Inject edge secret header on API traffic only"
    # Frontend traffic goes to the apex (DNS-only to Vercel) and never hits CF's
    # proxy, so scoping by host is equivalent to "API-only" and also defends
    # against future zones being added.
    expression = "(http.host eq \"${var.api_subdomain}.${var.zone_name}\")"
    action_parameters {
      headers {
        name      = "X-OHI-Edge-Secret"
        operation = "set"
        value     = var.edge_secret
      }
    }
    enabled = true
  }
}

# Mirror the edge-secret value into AWS Secrets Manager so Lambda can read it.
resource "aws_secretsmanager_secret_version" "cf_edge_secret" {
  secret_id     = local.secret_arns["cf_edge_secret"]
  secret_string = var.edge_secret
}
