# Cache rules: bypass everything under /api/, aggressively cache /health/live,
# let defaults handle the rest.
resource "cloudflare_ruleset" "cache" {
  zone_id     = local.zone_id
  name        = "ohi-prod-cache"
  description = "Cache-policy overrides"
  kind        = "zone"
  phase       = "http_request_cache_settings"

  rules {
    action      = "set_cache_settings"
    description = "Bypass cache for all API calls"
    expression  = "(starts_with(http.request.uri.path, \"/api/\"))"
    action_parameters {
      cache = false
    }
    enabled = true
  }

  rules {
    action      = "set_cache_settings"
    description = "Cache /health/live for 60s to absorb liveness-probe floods"
    expression  = "(http.request.uri.path eq \"/health/live\")"
    action_parameters {
      cache = true
      edge_ttl {
        mode    = "override_origin"
        default = 60
      }
      browser_ttl {
        mode    = "override_origin"
        default = 60
      }
    }
    enabled = true
  }
}
