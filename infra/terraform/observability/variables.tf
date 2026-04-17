variable "region" {
  type    = string
  default = "eu-central-1"
}

variable "alert_email" {
  type    = string
  default = "fabian@shiftbloom.studio"
}

variable "budget_forecast_usd" {
  type    = number
  default = 110
}

variable "budget_actual_usd" {
  type    = number
  default = 150
}
