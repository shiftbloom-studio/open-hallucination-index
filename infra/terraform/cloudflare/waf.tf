# --- Free-tier WAF managed ruleset + custom rules -----------------------------
# The full CF Managed Ruleset / OWASP Core Ruleset are paid. Free tier gives:
#   - Free Managed Ruleset (auto-enabled via zone setting below)
#   - Bot Fight Mode (per-zone setting)
#   - Our own cloudflare_ruleset custom rules

# Enable free-tier managed protection via zone-level settings
resource "cloudflare_zone_settings_override" "this" {
  zone_id = local.zone_id

  settings {
    # Security
    security_level      = "medium"
    challenge_ttl       = 1800
    browser_check       = "on"
    email_obfuscation   = "on"
    server_side_exclude = "on"
    hotlink_protection  = "off"
    # Caching
    cache_level       = "aggressive"
    browser_cache_ttl = 14400
    always_online     = "off"
    # SSL/TLS
    ssl                      = "strict"
    automatic_https_rewrites = "on"
    min_tls_version          = "1.2"
    tls_1_3                  = "on"
    opportunistic_encryption = "on"
    always_use_https         = "on"
  }
}

# Custom WAF rules
resource "cloudflare_ruleset" "custom_waf" {
  zone_id     = local.zone_id
  name        = "ohi-prod-waf-custom"
  description = "OHI production custom WAF rules"
  kind        = "zone"
  phase       = "http_request_firewall_custom"

  rules {
    action      = "block"
    description = "Block obvious SSRF attempts (metadata endpoints) on api host"
    expression  = <<-EOT
      (http.host eq "${local.api_hostname}") and (
        (http.request.uri.path contains "/169.254.169.254") or
        (http.request.uri.path contains "/latest/meta-data")
      )
    EOT
    enabled     = true
  }

  rules {
    action      = "managed_challenge"
    description = "Challenge non-browser User-Agent patterns on /feedback"
    expression  = <<-EOT
      (http.host eq "${local.api_hostname}") and
      (http.request.uri.path eq "/api/v2/feedback") and
      (http.user_agent eq "" or http.user_agent contains "curl" or http.user_agent contains "wget")
    EOT
    enabled     = true
  }
}

# Rate-limit rules (modern 4.x pattern: cloudflare_ruleset phase=http_ratelimit)
resource "cloudflare_ruleset" "rate_limits" {
  zone_id     = local.zone_id
  name        = "ohi-prod-rate-limits"
  description = "Per-route rate limits"
  kind        = "zone"
  phase       = "http_ratelimit"

  # CF free tier allows only 1 rule in the http_ratelimit phase per zone.
  # We keep the per-endpoint /verify rule because that's the Gemini-cost path.
  # Global per-IP ceiling dropped; Lambda is inherently bounded by account
  # concurrency (1000 default) so abuse is capped by AWS cost, not CF.
  rules {
    action      = "block"
    description = "POST /api/v2/verify on api host — ~${var.rate_limit_verify_per_min} req/min/IP (CF free tier forces 10s windows)"
    expression  = "(http.host eq \"${local.api_hostname}\") and (http.request.uri.path eq \"/api/v2/verify\") and (http.request.method eq \"POST\")"
    ratelimit {
      characteristics     = ["ip.src", "cf.colo.id"]
      period              = 10
      requests_per_period = ceil(var.rate_limit_verify_per_min / 6)
      mitigation_timeout  = 10
    }
    enabled = true
  }
}
