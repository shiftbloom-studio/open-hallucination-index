resource "aws_cloudwatch_dashboard" "ohi_prod" {
  dashboard_name = "${local.prefix}-prod"
  dashboard_body = jsonencode({
    widgets = [
      {
        type   = "metric"
        width  = 12
        height = 6
        properties = {
          region = var.region
          title  = "Lambda: invocations / errors / duration p99"
          metrics = [
            ["AWS/Lambda", "Invocations", "FunctionName", local.function_name, { stat = "Sum" }],
            [".", "Errors", ".", ".", { stat = "Sum" }],
            [".", "Duration", ".", ".", { stat = "p99" }],
          ]
          view   = "timeSeries"
          period = 300
        }
      },
      {
        type   = "metric"
        width  = 12
        height = 6
        properties = {
          region = var.region
          title  = "App: PipelineError / RateLimitApp / PCOriginTimeout / ColdStart"
          metrics = [
            [local.metric_namespace, "PipelineError", { stat = "Sum" }],
            [".", "RateLimitApp", { stat = "Sum" }],
            [".", "PCOriginTimeout", { stat = "Sum" }],
            [".", "LambdaColdStart", { stat = "Sum" }],
          ]
          view   = "timeSeries"
          period = 300
        }
      },
    ]
  })
}
