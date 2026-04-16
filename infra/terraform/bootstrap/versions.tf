terraform {
  required_version = ">= 1.10.0, < 2.0.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.80"
    }
  }
}

provider "aws" {
  region = var.region

  default_tags {
    tags = module.shared.tags
  }
}

module "shared" {
  source = "../_shared"
  layer  = "bootstrap"
  region = var.region
}
