# Cache rules — scoped to the api.* host (apex is DNS-only to Vercel, which
# owns its own edge cache).
resource "cloudflare_ruleset" "cache" {
  zone_id     = local.zone_id
  name        = "ohi-prod-cache"
  description = "Cache-policy overrides for the API subdomain"
  kind        = "zone"
  phase       = "http_request_cache_settings"

  rules {
    action      = "set_cache_settings"
    description = "Bypass cache for all API calls on the api subdomain"
    expression  = "(http.host eq \"${local.api_hostname}\") and (starts_with(http.request.uri.path, \"/api/\"))"
    action_parameters {
      cache = false
    }
    enabled = true
  }

  rules {
    action      = "set_cache_settings"
    description = "Cache /health/live for 60s on api subdomain to absorb liveness-probe floods"
    expression  = "(http.host eq \"${local.api_hostname}\") and (http.request.uri.path eq \"/health/live\")"
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
