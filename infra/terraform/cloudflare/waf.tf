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
    description = "Block obvious SSRF attempts (metadata endpoints)"
    expression  = <<-EOT
      (http.request.uri.path contains "/169.254.169.254") or
      (http.request.uri.path contains "/latest/meta-data") or
      (http.request.body.raw contains "169.254.169.254")
    EOT
    enabled     = true
  }

  rules {
    action      = "managed_challenge"
    description = "Challenge non-browser User-Agent patterns on /feedback"
    expression  = <<-EOT
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

  rules {
    action      = "block"
    description = "POST /api/v2/verify — ${var.rate_limit_verify_per_min} req/min/IP"
    expression  = "(http.request.uri.path eq \"/api/v2/verify\") and (http.request.method eq \"POST\")"
    ratelimit {
      characteristics     = ["ip.src"]
      period              = 60
      requests_per_period = var.rate_limit_verify_per_min
      mitigation_timeout  = 60
    }
    enabled = true
  }

  rules {
    action      = "block"
    description = "Global per-IP ceiling — ${var.rate_limit_global_per_hour} req/hour/IP"
    expression  = "(http.request.uri.path wildcard \"*\")"
    ratelimit {
      characteristics     = ["ip.src"]
      period              = 3600
      requests_per_period = var.rate_limit_global_per_hour
      mitigation_timeout  = 300
    }
    enabled = true
  }
}
