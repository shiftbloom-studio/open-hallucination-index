output "function_arn" {
  value = aws_lambda_function.api.arn
}

output "function_name" {
  value = aws_lambda_function.api.function_name
}

output "function_url" {
  description = "Direct Lambda Function URL (admin back-channel; not user-facing)."
  value       = aws_lambda_function_url.api.function_url
}

output "function_url_hostname" {
  description = "Hostname-only form for the Lambda Function URL (kept for back-compat / historical refs)."
  value       = replace(replace(aws_lambda_function_url.api.function_url, "https://", ""), "/", "")
}

# API Gateway endpoint — raw execute-api URL. Cloudflare proxies
# ohi-api.shiftbloom.studio to the API Gateway CUSTOM DOMAIN's regional
# target (configured in cloudflare/api_gateway_custom_domain.tf), not this
# one directly, because a custom domain is what makes API Gateway accept
# Host = ohi-api.shiftbloom.studio.
output "api_gateway_id" {
  description = "HTTP API id; the custom-domain layer references this for api mapping."
  value       = aws_apigatewayv2_api.api.id
}

output "api_gateway_stage_id" {
  description = "Stage id used by api mapping."
  value       = aws_apigatewayv2_stage.default.id
}

output "api_gateway_url" {
  description = "Raw execute-api URL of the HTTP API. Not user-facing."
  value       = aws_apigatewayv2_api.api.api_endpoint
}

output "acm_api_certificate_arn" {
  description = "ACM cert ARN for ohi-api.shiftbloom.studio; cloudflare layer uses it on the API Gateway custom domain."
  value       = aws_acm_certificate.api.arn
}

output "acm_api_domain_validation_options" {
  description = "Validation CNAMEs that cloudflare/ must create for the ACM cert to become ISSUED."
  value       = aws_acm_certificate.api.domain_validation_options
}

output "api_public_host" {
  description = "Public hostname served at ohi-api.shiftbloom.studio."
  value       = local.api_public_host
}

output "log_group_name" {
  value = aws_cloudwatch_log_group.api.name
}
