resource "vercel_project" "frontend" {
  name      = var.project_name
  framework = "nextjs"

  git_repository = {
    type              = "github"
    repo              = "${var.github_org}/${var.github_repo}"
    production_branch = var.production_branch
  }

  root_directory  = var.root_directory
  install_command = "npm ci"
  build_command   = "npm run build"
  # Leave output_directory unset — Vercel's Next.js adapter auto-detects
  # the layout produced by `output: 'export'`. Explicitly setting it to
  # "out" makes Vercel skip the .next metadata path and fail with
  # "routes-manifest.json not found".

  # Phase 3 plan: static export, no SSR. Disable Vercel's Node server features
  # that are irrelevant for a static site.
  serverless_function_region = "fra1" # Frankfurt — matches AWS eu-central-1 for latency consistency IF we ever move off static

  # Security headers — good defaults for a public open-source frontend.
  # Omitted here; Phase 3 frontend plan will add these in next.config.js.
}

# NEXT_PUBLIC_API_BASE — exposed to the browser, tells it where to reach the API.
resource "vercel_project_environment_variable" "api_base" {
  project_id = vercel_project.frontend.id
  key        = "NEXT_PUBLIC_API_BASE"
  value      = var.api_base_url
  target     = ["production", "preview", "development"]
  comment    = "Cross-origin URL the browser uses for fetch() / EventSource against the OHI API"
}

resource "vercel_project_environment_variable" "site_url" {
  project_id = vercel_project.frontend.id
  key        = "NEXT_PUBLIC_SITE_URL"
  value      = "https://${var.apex_domain}"
  target     = ["production", "preview", "development"]
  comment    = "Public site URL for canonical links, sitemap, metadata"
}
