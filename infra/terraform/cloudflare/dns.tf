# ---------------------------------------------------------------------------
# Apex record — Vercel-hosted static frontend (DNS-only, Vercel owns the edge).
# CF CNAME flattening lets us put a CNAME at zone apex; Vercel issues + serves
# the TLS cert at its edge.
# ---------------------------------------------------------------------------
resource "cloudflare_record" "apex" {
  zone_id = local.zone_id
  name    = "@"
  type    = "CNAME"
  content = "cname.vercel-dns.com"
  proxied = false # DNS-only (gray cloud) — Vercel's edge, not CF's
  ttl     = 300
  comment = "OHI v2 public frontend served by Vercel."
}

# Optional Vercel domain-ownership TXT. Usually NOT needed when CF CNAME
# flattening at apex resolves cleanly — Vercel accepts CNAME-based verification.
# If Vercel's dashboard shows a TXT challenge after the apex CNAME is live,
# supply the token via -var=vercel_verification_token and this record is created.
resource "cloudflare_record" "vercel_verify" {
  count = var.vercel_verification_token != "" ? 1 : 0

  zone_id = local.zone_id
  name    = "_vercel"
  type    = "TXT"
  content = var.vercel_verification_token
  proxied = false
  ttl     = 300
  comment = "Vercel domain-ownership challenge (only present if Vercel requested one)"
}

# ---------------------------------------------------------------------------
# API subdomain — Lambda Function URL, CF-proxied (WAF + edge-secret + rate limits).
# ---------------------------------------------------------------------------
resource "cloudflare_record" "api" {
  zone_id = local.zone_id
  name    = "api"
  type    = "CNAME"
  content = local.lambda_fn_hostname
  proxied = true # orange cloud — all CF edge features apply to API traffic
  ttl     = 1
  comment = "OHI v2 API served by AWS Lambda, fronted by Cloudflare (WAF + rate limits + X-OHI-Edge-Secret injection)."
}

# Tunnel hostnames (neo4j/qdrant/pg/redis) are declared in tunnel.tf; unchanged.
