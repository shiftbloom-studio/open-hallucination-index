# Tunnel resource — generates the tunnel secret we'll store in Secrets Manager.
resource "random_password" "tunnel_secret" {
  length  = 64
  special = false
}

resource "cloudflare_zero_trust_tunnel_cloudflared" "pc" {
  account_id    = local.account_id
  name          = "ohi-pc"
  tunnel_secret = base64encode(random_password.tunnel_secret.result)
  config_src    = "cloudflare" # we manage config via cloudflare_zero_trust_tunnel_cloudflared_config
}

# Ingress rules routing tunnel requests to PC-side docker services.
resource "cloudflare_zero_trust_tunnel_cloudflared_config" "pc" {
  account_id = local.account_id
  tunnel_id  = cloudflare_zero_trust_tunnel_cloudflared.pc.id

  config {
    ingress_rule {
      hostname = "neo4j.${var.zone_name}"
      service  = "http://neo4j:7474"
    }
    ingress_rule {
      hostname = "qdrant.${var.zone_name}"
      service  = "http://qdrant:6333"
    }
    ingress_rule {
      hostname = "pg.${var.zone_name}"
      service  = "http://postgrest:3000"
    }
    ingress_rule {
      hostname = "redis.${var.zone_name}"
      service  = "http://webdis:7379"
    }
    # Catch-all (required as the last rule)
    ingress_rule {
      service = "http_status:404"
    }
  }
}

# DNS records for each tunneled hostname, proxied.
locals {
  tunnel_hostnames = {
    neo4j  = "neo4j"
    qdrant = "qdrant"
    pg     = "pg"
    redis  = "redis"
  }
}

resource "cloudflare_record" "tunnel" {
  for_each = local.tunnel_hostnames

  zone_id = local.zone_id
  name    = each.value
  type    = "CNAME"
  content = "${cloudflare_zero_trust_tunnel_cloudflared.pc.id}.cfargotunnel.com"
  proxied = true
  ttl     = 1
  comment = "OHI tunnel public hostname — protected by CF Access (see access.tf)"
}

# Seed the cloudflared-tunnel-token secret from the CF-generated value.
resource "aws_secretsmanager_secret_version" "cloudflared_tunnel_token" {
  secret_id     = local.secret_arns["cloudflared_tunnel_token"]
  secret_string = cloudflare_zero_trust_tunnel_cloudflared.pc.tunnel_token
}
