output "function_arn" {
  value = aws_lambda_function.api.arn
}

output "function_name" {
  value = aws_lambda_function.api.function_name
}

output "function_url" {
  description = "Direct Lambda Function URL (not user-facing; Cloudflare proxies it)."
  value       = aws_lambda_function_url.api.function_url
}

output "function_url_hostname" {
  description = "Hostname-only form for the CF CNAME target."
  value       = replace(replace(aws_lambda_function_url.api.function_url, "https://", ""), "/", "")
}

output "log_group_name" {
  value = aws_cloudwatch_log_group.api.name
}
