output "project_id" {
  value = vercel_project.frontend.id
}

output "project_name" {
  value = vercel_project.frontend.name
}

output "apex_domain" {
  value = var.apex_domain
}

output "api_base_url" {
  value = var.api_base_url
}

output "verification_required" {
  description = <<-EOT
    If the apex CNAME resolves successfully to Vercel, domain is verified
    automatically. If Vercel requires an explicit _vercel TXT challenge
    (rare with Cloudflare's CNAME flattening), check the Vercel dashboard
    after first apply and supply the value to var.vercel_verification_token
    on the cloudflare/ layer.
  EOT
  value       = "Check Vercel dashboard → Settings → Domains → ${var.apex_domain}"
}
