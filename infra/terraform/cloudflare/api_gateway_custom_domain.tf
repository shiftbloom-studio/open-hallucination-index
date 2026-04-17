# API Gateway custom domain wiring.
#
# The ACM cert is created in compute/ (it needs the AWS provider from that
# layer's state). This layer:
#   1) creates the DNS-01 validation CNAME that makes ACM issue the cert,
#   2) waits for ACM to observe it via aws_acm_certificate_validation,
#   3) creates the apigatewayv2 custom domain using the validated cert,
#   4) maps the API (compute layer's aws_apigatewayv2_api) onto that domain.
#
# cloudflare_record.api (in dns.tf) CNAMEs ohi-api.shiftbloom.studio at the
# custom domain's regional target — that's what makes API Gateway accept
# the visitor's Host header.

locals {
  acm_cert_arn              = data.terraform_remote_state.compute.outputs.acm_api_certificate_arn
  acm_validation_options    = data.terraform_remote_state.compute.outputs.acm_api_domain_validation_options
  api_gateway_id            = data.terraform_remote_state.compute.outputs.api_gateway_id
  api_gateway_stage_id      = data.terraform_remote_state.compute.outputs.api_gateway_stage_id
  api_public_host           = data.terraform_remote_state.compute.outputs.api_public_host
}

# DNS validation CNAME (ACM demands a record of the form
#   _<hash>.ohi-api.shiftbloom.studio  CNAME  _<hash>.acm-validations.aws.
# where both halves are echoed by AWS in domain_validation_options).
resource "cloudflare_record" "api_acm_validation" {
  for_each = {
    for dvo in local.acm_validation_options : dvo.domain_name => {
      name  = dvo.resource_record_name
      type  = dvo.resource_record_type
      value = dvo.resource_record_value
    }
  }

  zone_id = local.zone_id
  # CF stores names relative to the zone; strip the trailing dot AWS emits.
  name    = trimsuffix(each.value.name, ".")
  type    = each.value.type
  content = trimsuffix(each.value.value, ".")
  ttl     = 60
  proxied = false
  comment = "ACM DNS-01 validation for ${each.key}"
}

# Wait for AWS to observe the CNAME and flip the cert to ISSUED.
resource "aws_acm_certificate_validation" "api" {
  certificate_arn = local.acm_cert_arn
  validation_record_fqdns = [
    for r in cloudflare_record.api_acm_validation : r.hostname
  ]
}

# Custom domain using the validated cert.
resource "aws_apigatewayv2_domain_name" "api" {
  domain_name = local.api_public_host

  domain_name_configuration {
    certificate_arn = local.acm_cert_arn
    endpoint_type   = "REGIONAL"
    security_policy = "TLS_1_2"
  }

  depends_on = [aws_acm_certificate_validation.api]
}

# Route requests that arrive with Host = ohi-api.shiftbloom.studio to the
# HTTP API's $default stage.
resource "aws_apigatewayv2_api_mapping" "api" {
  api_id      = local.api_gateway_id
  domain_name = aws_apigatewayv2_domain_name.api.id
  stage       = local.api_gateway_stage_id
}

# Expose the custom-domain's regional target so cloudflare_record.api (in
# dns.tf) can CNAME to it.
output "api_gateway_custom_domain_target" {
  description = "Regional target hostname for the API Gateway custom domain; CF CNAME content."
  value       = aws_apigatewayv2_domain_name.api.domain_name_configuration[0].target_domain_name
}
