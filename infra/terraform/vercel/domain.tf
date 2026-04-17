resource "vercel_project_domain" "apex" {
  project_id = vercel_project.frontend.id
  domain     = var.apex_domain
}
