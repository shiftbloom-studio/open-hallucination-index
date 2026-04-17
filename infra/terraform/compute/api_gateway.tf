# API Gateway HTTP API sits in front of the Lambda function. This layer
# creates ONLY the API / integration / route / stage / invoke permission.
# The public-facing pieces (ACM cert validation, custom domain, api mapping,
# CF DNS pointing at the custom domain target) live in cloudflare/ so they
# can share a single apply with the DNS records they depend on.
#
# We pivoted off the Lambda Function URL because Function URLs validate
# Host header against <id>.lambda-url.<region>.on.aws and CF free/pro
# tiers cannot override Host upstream. The stock API Gateway endpoint has
# the SAME restriction; custom domains are what let it accept an arbitrary
# Host (in our case ohi-api.shiftbloom.studio). See cloudflare/
# api_gateway_custom_domain.tf for the custom-domain side.

resource "aws_apigatewayv2_api" "api" {
  name          = "${local.prefix}-api"
  protocol_type = "HTTP"
  description   = "OHI v2 API - HTTP API fronting the ohi-api Lambda; served through CF at ohi-api.shiftbloom.studio via an API Gateway custom domain."

  # CORS is enforced at Lambda by CORSMiddleware (driven by OHI_CORS_ORIGINS).
  # Keep API Gateway's CORS permissive to avoid double-scoping.
  cors_configuration {
    allow_origins = ["*"]
    allow_methods = ["*"]
    allow_headers = ["*"]
    max_age       = 86400
  }
}

resource "aws_apigatewayv2_integration" "lambda" {
  api_id                 = aws_apigatewayv2_api.api.id
  integration_type       = "AWS_PROXY"
  integration_uri        = aws_lambda_function.api.invoke_arn
  integration_method     = "POST"
  payload_format_version = "2.0"
  timeout_milliseconds   = 30000
}

resource "aws_apigatewayv2_route" "default" {
  api_id    = aws_apigatewayv2_api.api.id
  route_key = "$default"
  target    = "integrations/${aws_apigatewayv2_integration.lambda.id}"
}

resource "aws_apigatewayv2_stage" "default" {
  api_id      = aws_apigatewayv2_api.api.id
  name        = "$default"
  auto_deploy = true

  default_route_settings {
    throttling_burst_limit = 200
    throttling_rate_limit  = 100
  }
}

resource "aws_lambda_permission" "api_gateway" {
  statement_id  = "AllowAPIGatewayInvoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.api.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_apigatewayv2_api.api.execution_arn}/*/*"
}
