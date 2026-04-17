output "zone_id" {
  value = local.zone_id
}

output "apex_hostname" {
  description = "Zone apex — served by Vercel (frontend)."
  value       = var.zone_name
}

output "api_hostname" {
  description = "API subdomain — served by AWS Lambda through CF proxy (WAF + rate limits + edge-secret)."
  value       = local.api_hostname
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
