# ACM cert for the API's public hostname. Validation CNAME lives in CF, so
# this cert is created here in PENDING_VALIDATION state; cloudflare/ adds
# the CNAME and the matching `aws_acm_certificate_validation` resource.
# API Gateway custom domain + mapping also live in cloudflare/ (they depend
# on the validated cert, kept in the same apply as the validation record).

resource "aws_acm_certificate" "api" {
  domain_name       = local.api_public_host
  validation_method = "DNS"

  lifecycle {
    create_before_destroy = true
  }
}

locals {
  # Source of truth for the public API hostname. Kept separate from
  # tunnel_hostname_* so the ACM cert stays in sync with the CF record.
  api_public_host = "ohi-api.shiftbloom.studio"
}
