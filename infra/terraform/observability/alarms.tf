# Lambda 5xx rate
resource "aws_cloudwatch_metric_alarm" "lambda_5xx" {
  alarm_name          = "${local.prefix}-lambda-5xx-rate"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 3
  threshold           = 0.1
  alarm_actions       = [aws_sns_topic.alerts.arn]

  metric_query {
    id          = "error_rate"
    expression  = "errors / invocations"
    label       = "5xx error rate"
    return_data = true
  }

  metric_query {
    id = "errors"
    metric {
      namespace   = "AWS/Lambda"
      metric_name = "Errors"
      period      = 300
      stat        = "Sum"
      dimensions  = { FunctionName = local.function_name }
    }
  }

  metric_query {
    id = "invocations"
    metric {
      namespace   = "AWS/Lambda"
      metric_name = "Invocations"
      period      = 300
      stat        = "Sum"
      dimensions  = { FunctionName = local.function_name }
    }
  }
}

# PC origin timeout spike
resource "aws_cloudwatch_metric_alarm" "pc_origin_timeout" {
  alarm_name          = "${local.prefix}-pc-origin-timeout-rate"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 3
  threshold           = 5
  namespace           = local.metric_namespace
  metric_name         = "PCOriginTimeout"
  period              = 300
  statistic           = "Sum"
  alarm_actions       = [aws_sns_topic.alerts.arn]
}
