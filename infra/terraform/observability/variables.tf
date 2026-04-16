variable "region" {
  type    = string
  default = "eu-central-1"
}

variable "alert_email" {
  type    = string
  default = "fabian@shiftbloom.studio"
}

variable "budget_forecast_eur" {
  type    = number
  default = 100
}

variable "budget_actual_eur" {
  type    = number
  default = 140
}
