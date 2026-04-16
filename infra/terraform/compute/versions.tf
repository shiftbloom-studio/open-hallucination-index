terraform {
  required_version = ">= 1.10.0, < 2.0.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.80"
    }
  }

  backend "s3" {
    # bucket passed via -backend-config at init time.
    key            = "prod/compute/terraform.tfstate"
    region         = "eu-central-1"
    dynamodb_table = "ohi-tfstate-lock"
    encrypt        = true
    kms_key_id     = "alias/ohi-secrets"
  }
}

provider "aws" {
  region = var.region
  default_tags { tags = module.shared.tags }
}

module "shared" {
  source = "../_shared"
  layer  = "compute"
  region = var.region
}

data "aws_caller_identity" "current" {}
data "aws_kms_alias" "ohi_secrets" { name = "alias/ohi-secrets" }

data "terraform_remote_state" "secrets" {
  backend = "s3"
  config = {
    bucket = "ohi-tfstate-${data.aws_caller_identity.current.account_id}"
    key    = "prod/secrets/terraform.tfstate"
    region = var.region
  }
}

data "terraform_remote_state" "storage" {
  backend = "s3"
  config = {
    bucket = "ohi-tfstate-${data.aws_caller_identity.current.account_id}"
    key    = "prod/storage/terraform.tfstate"
    region = var.region
  }
}

data "aws_ecr_repository" "api" {
  name = "${module.shared.name_prefix}-api"
}

locals {
  account_id       = data.aws_caller_identity.current.account_id
  prefix           = module.shared.name_prefix
  secret_arns      = data.terraform_remote_state.secrets.outputs.secret_arns
  artifacts_bucket = data.terraform_remote_state.storage.outputs.artifacts_bucket
}
