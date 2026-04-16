resource "aws_budgets_budget" "forecast" {
  name         = "${local.prefix}-budget-forecast"
  budget_type  = "COST"
  limit_amount = tostring(var.budget_forecast_eur)
  limit_unit   = "EUR"
  time_unit    = "MONTHLY"

  cost_filter {
    name   = "TagKeyValue"
    values = ["aws:CostCenter$ohi"]
  }

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 100
    threshold_type             = "PERCENTAGE"
    notification_type          = "FORECASTED"
    subscriber_email_addresses = [var.alert_email]
    subscriber_sns_topic_arns  = [aws_sns_topic.alerts.arn]
  }
}

resource "aws_budgets_budget" "actual" {
  name         = "${local.prefix}-budget-actual"
  budget_type  = "COST"
  limit_amount = tostring(var.budget_actual_eur)
  limit_unit   = "EUR"
  time_unit    = "MONTHLY"

  cost_filter {
    name   = "TagKeyValue"
    values = ["aws:CostCenter$ohi"]
  }

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 100
    threshold_type             = "PERCENTAGE"
    notification_type          = "ACTUAL"
    subscriber_email_addresses = [var.alert_email]
    subscriber_sns_topic_arns  = [aws_sns_topic.alerts.arn]
  }
}
