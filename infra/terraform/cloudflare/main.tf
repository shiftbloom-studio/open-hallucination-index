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
  zone_id    = data.cloudflare_zone.this.id
  account_id = var.cf_account_id
  # API origin used by cloudflare_record.api (CNAME target). We front Lambda
  # with API Gateway HTTP API because Lambda Function URLs validate the Host
  # header against their own hostname and CF free-tier can't override Host
  # upstream. The custom domain's regional target is created by
  # aws_apigatewayv2_domain_name.api (this layer), which presents the ACM
  # cert for ohi-api.shiftbloom.studio and accepts that Host.
  api_origin_hostname = aws_apigatewayv2_domain_name.api.domain_name_configuration[0].target_domain_name
  # Kept for historical / back-compat refs. Not used as the CF origin.
  lambda_fn_hostname = data.terraform_remote_state.compute.outputs.function_url_hostname
  secret_arns        = data.terraform_remote_state.secrets.outputs.secret_arns

  # Record-naming helper: service records live at the zone root with a
  # `<apex_subdomain>-` prefix so free-tier Universal SSL (which covers
  # only *.zone, not *.apex.zone) covers them. CF doesn't allow subzones
  # on free tier, so zone-root flattening is the only cert path.
  # Examples with apex_subdomain="ohi":
  #   api       → ohi-api.shiftbloom.studio   (service record)
  #   neo4j     → ohi-neo4j.shiftbloom.studio (service record)
  #   <apex>    → ohi.shiftbloom.studio       (frontend, kept as nested CNAME
  #                                            at `name=ohi` in the zone)
  record_prefix = var.apex_subdomain == "" ? "" : "${var.apex_subdomain}-"

  # Full public hostnames (for use in outputs, env vars, cross-references)
  apex_hostname = var.apex_subdomain == "" ? var.zone_name : "${var.apex_subdomain}.${var.zone_name}"
  api_hostname  = "${local.record_prefix}${var.api_subdomain}.${var.zone_name}"
}
