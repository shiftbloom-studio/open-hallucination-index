# Wave 3 Stream G.1 auto-rollback alarms + SNS topic
# ---------------------------------------------------
#
# Per Wave 3 spec §2.6 the merge-to-main flow ends with a 30-minute
# post-deploy synthetic probe + CloudWatch alarms; on either signal
# the auto-rollback workflow fires. These resources were created
# directly via AWS CLI during the Wave 3 autonomous deploy so the
# post-deploy observability was live immediately. To bring them
# under TF state:
#
#   cd infra/terraform/observability
#   terraform import aws_sns_topic.rollback_alerts \
#     arn:aws:sns:eu-central-1:349744179866:ohi-rollback-alerts
#   terraform import aws_sns_topic_subscription.rollback_alerts_email \
#     arn:aws:sns:eu-central-1:349744179866:ohi-rollback-alerts:<sub-id>
#   terraform import aws_cloudwatch_metric_alarm.wave3_error_rate     ohi-api-error-rate
#   terraform import aws_cloudwatch_metric_alarm.wave3_p95_latency    ohi-api-p95-latency
#   terraform import aws_cloudwatch_metric_alarm.wave3_5xx_rate       ohi-api-5xx-rate
#   terraform import aws_cloudwatch_metric_alarm.wave3_throttles      ohi-api-throttles

resource "aws_sns_topic" "rollback_alerts" {
  name = "${local.prefix}-rollback-alerts"

  tags = {
    Purpose = "wave3-auto-rollback"
  }
}

resource "aws_sns_topic_subscription" "rollback_alerts_email" {
  topic_arn = aws_sns_topic.rollback_alerts.arn
  protocol  = "email"
  endpoint  = var.alert_email
}

# Lambda error rate: > 10 Errors over 5 1-minute periods (spec default).
resource "aws_cloudwatch_metric_alarm" "wave3_error_rate" {
  alarm_name          = "${local.prefix}-api-error-rate"
  alarm_description   = "OHI /verify Lambda error count > 10 over 5 min"
  metric_name         = "Errors"
  namespace           = "AWS/Lambda"
  statistic           = "Sum"
  period              = 60
  evaluation_periods  = 5
  threshold           = 10
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "notBreaching"
  dimensions          = { FunctionName = local.function_name }
  alarm_actions       = [aws_sns_topic.rollback_alerts.arn]
}

# Lambda p95 duration: > 150s (spec default; matches the rigor=maximum ceiling).
resource "aws_cloudwatch_metric_alarm" "wave3_p95_latency" {
  alarm_name          = "${local.prefix}-api-p95-latency"
  alarm_description   = "OHI /verify Lambda p95 duration > 150s"
  metric_name         = "Duration"
  namespace           = "AWS/Lambda"
  extended_statistic  = "p95"
  period              = 300
  evaluation_periods  = 3
  threshold           = 150000
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "notBreaching"
  dimensions          = { FunctionName = local.function_name }
  alarm_actions       = [aws_sns_topic.rollback_alerts.arn]
}

# API Gateway 5xx: > 5 over 5 1-minute periods.
resource "aws_cloudwatch_metric_alarm" "wave3_5xx_rate" {
  alarm_name          = "${local.prefix}-api-5xx-rate"
  alarm_description   = "OHI API Gateway 5xx count > 5 over 5 min"
  metric_name         = "5xx"
  namespace           = "AWS/ApiGateway"
  statistic           = "Sum"
  period              = 60
  evaluation_periods  = 5
  threshold           = 5
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "notBreaching"
  dimensions          = { ApiName = "${local.prefix}-api" }
  alarm_actions       = [aws_sns_topic.rollback_alerts.arn]
}

# Lambda throttles: > 3 over 5 1-minute periods.
resource "aws_cloudwatch_metric_alarm" "wave3_throttles" {
  alarm_name          = "${local.prefix}-api-throttles"
  alarm_description   = "OHI /verify Lambda throttles > 3 over 5 min"
  metric_name         = "Throttles"
  namespace           = "AWS/Lambda"
  statistic           = "Sum"
  period              = 60
  evaluation_periods  = 5
  threshold           = 3
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "notBreaching"
  dimensions          = { FunctionName = local.function_name }
  alarm_actions       = [aws_sns_topic.rollback_alerts.arn]
}
