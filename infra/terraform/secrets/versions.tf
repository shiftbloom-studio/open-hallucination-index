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
    key            = "prod/secrets/terraform.tfstate"
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
  layer  = "secrets"
  region = var.region
}

data "aws_kms_alias" "ohi_secrets" {
  name = "alias/ohi-secrets"
}
