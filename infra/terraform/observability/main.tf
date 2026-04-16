module "shared" {
  source = "../_shared"
  layer  = "observability"
  region = var.region
}

data "aws_caller_identity" "current" {}

data "terraform_remote_state" "compute" {
  backend = "s3"
  config = {
    bucket = "ohi-tfstate-${data.aws_caller_identity.current.account_id}"
    key    = "prod/compute/terraform.tfstate"
    region = var.region
  }
}

locals {
  prefix         = module.shared.name_prefix
  log_group_name = data.terraform_remote_state.compute.outputs.log_group_name
  function_name  = data.terraform_remote_state.compute.outputs.function_name
}
