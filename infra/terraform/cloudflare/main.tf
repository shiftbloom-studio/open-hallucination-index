module "shared" {
  source = "../_shared"
  layer  = "cloudflare"
  region = var.region
}

data "aws_caller_identity" "current" {}

data "terraform_remote_state" "secrets" {
  backend = "s3"
  config = {
    bucket = "ohi-tfstate-${data.aws_caller_identity.current.account_id}"
    key    = "prod/secrets/terraform.tfstate"
    region = var.region
  }
}

data "terraform_remote_state" "compute" {
  backend = "s3"
  config = {
    bucket = "ohi-tfstate-${data.aws_caller_identity.current.account_id}"
    key    = "prod/compute/terraform.tfstate"
    region = var.region
  }
}

# Vercel layer provides the apex domain + the verification token (if any).
# Optional: if the Vercel layer hasn't been applied yet (first-run order:
# vercel → cloudflare), the `vercel` remote state won't exist; allow that by
# defaulting zone_name + apex to tfvars instead.
data "terraform_remote_state" "vercel" {
  backend = "s3"
  config = {
    bucket = "ohi-tfstate-${data.aws_caller_identity.current.account_id}"
    key    = "prod/vercel/terraform.tfstate"
    region = var.region
  }
  # If vercel state doesn't exist yet, TF errors — acceptable since apply order
  # requires vercel first. Documented in infra/terraform/README.md.
}

# Zone is created manually in the CF dashboard (free-tier subdomain zone setup).
# We read it as a data source and manage records/rulesets under it.
data "cloudflare_zone" "this" {
  name = var.zone_name
}

locals {
  zone_id            = data.cloudflare_zone.this.id
  account_id         = var.cf_account_id
  lambda_fn_hostname = data.terraform_remote_state.compute.outputs.function_url_hostname
  secret_arns        = data.terraform_remote_state.secrets.outputs.secret_arns
  api_hostname       = "${var.api_subdomain}.${var.zone_name}"
}
