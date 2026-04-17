# ---------------------------------------------------------------------------
# Apex record — Vercel-hosted static frontend (DNS-only, Vercel owns the edge).
# When apex_subdomain="" we use "@" (zone apex). When set, we prefix the label.
# CF CNAME flattening lets us put a CNAME at any name including the zone apex;
# Vercel issues + serves the TLS cert at its edge.
# ---------------------------------------------------------------------------
resource "cloudflare_record" "apex" {
  zone_id = local.zone_id
  name    = var.apex_subdomain == "" ? "@" : var.apex_subdomain
  type    = "CNAME"
  content = "cname.vercel-dns.com"
  proxied = false # DNS-only (gray cloud) — Vercel's edge, not CF's
  ttl     = 300
  comment = "OHI v2 public frontend served by Vercel (${local.apex_hostname})."
}

# Optional Vercel domain-ownership TXT. Usually NOT needed when CNAME-based
# verification works. If Vercel dashboard shows a TXT challenge, supply the
# token via -var=vercel_verification_token and this record appears.
resource "cloudflare_record" "vercel_verify" {
  count = var.vercel_verification_token != "" ? 1 : 0

  zone_id = local.zone_id
  name    = var.apex_subdomain == "" ? "_vercel" : "_vercel.${var.apex_subdomain}"
  type    = "TXT"
  content = var.vercel_verification_token
  proxied = false
  ttl     = 300
  comment = "Vercel domain-ownership challenge"
}

# ---------------------------------------------------------------------------
# API subdomain — Lambda Function URL, CF-proxied (WAF + edge-secret + rate limits).
# ---------------------------------------------------------------------------
resource "cloudflare_record" "api" {
  zone_id = local.zone_id
  name    = "${local.record_prefix}${var.api_subdomain}"
  type    = "CNAME"
  content = local.lambda_fn_hostname
  proxied = true # orange cloud — all CF edge features apply to API traffic
  ttl     = 1
  comment = "OHI v2 API — AWS Lambda via CF (WAF + rate limits)"
}

# Tunnel hostnames (neo4j/qdrant/pg/redis/embed) are declared in tunnel.tf with matching label logic.
