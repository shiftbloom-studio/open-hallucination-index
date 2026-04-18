variable "region" {
  type    = string
  default = "eu-central-1"
}

variable "ttl_attribute" {
  description = "Name of the numeric (unix-epoch seconds) attribute DynamoDB uses for automatic expiry. Items older than this value get reaped within ~48h."
  type        = string
  default     = "ttl"
}
