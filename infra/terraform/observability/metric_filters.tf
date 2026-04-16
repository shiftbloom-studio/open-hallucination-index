locals {
  metric_namespace = "OHI/App"

  metric_filters = {
    pipeline_error = {
      pattern = "{ $.level = \"ERROR\" && $.pipeline_stage = * }"
      metric  = "PipelineError"
    }
    rate_limit_triggered = {
      pattern = "{ $.msg = \"rate_limit_triggered\" }"
      metric  = "RateLimitApp"
    }
    pc_origin_timeout = {
      pattern = "{ $.msg = \"pc_origin_timeout\" }"
      metric  = "PCOriginTimeout"
    }
    cold_start = {
      pattern = "{ $.msg = \"lambda_cold_start\" }"
      metric  = "LambdaColdStart"
    }
  }
}

resource "aws_cloudwatch_log_metric_filter" "this" {
  for_each = local.metric_filters

  name           = "${local.prefix}-${each.key}"
  log_group_name = local.log_group_name
  pattern        = each.value.pattern

  metric_transformation {
    name          = each.value.metric
    namespace     = local.metric_namespace
    value         = "1"
    default_value = "0"
  }
}
