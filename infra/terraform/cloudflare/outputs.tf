output "zone_id" {
  value = local.zone_id
}

output "apex_hostname" {
  value = var.zone_name
}

output "tunnel_id" {
  value = cloudflare_zero_trust_tunnel_cloudflared.pc.id
}

output "tunnel_cname_target" {
  value = "${cloudflare_zero_trust_tunnel_cloudflared.pc.id}.cfargotunnel.com"
}

output "service_token_client_id" {
  description = "Non-secret; useful for operator dashboards."
  value       = cloudflare_zero_trust_access_service_token.lambda.client_id
}

output "tunnel_hostnames" {
  value = { for k, v in local.tunnel_hostnames : k => "${v}.${var.zone_name}" }
}
