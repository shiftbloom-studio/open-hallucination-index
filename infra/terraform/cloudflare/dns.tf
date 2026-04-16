# Apex record — proxied, points to Lambda Function URL.
resource "cloudflare_record" "apex" {
  zone_id = local.zone_id
  name    = "@"
  type    = "CNAME"
  content = local.lambda_fn_hostname
  proxied = true
  ttl     = 1 # "Automatic" for proxied records
  comment = "OHI v2 public endpoint; proxied through CF (WAF, TLS, cache)."
}

# Tunnel hostnames are declared in tunnel.tf next to the tunnel resource.
