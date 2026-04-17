terraform {
  required_version = ">= 1.10.0, < 2.0.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.80"
    }
    vercel = {
      source  = "vercel/vercel"
      version = "~> 2.0"
    }
  }

  backend "s3" {
    # bucket passed via -backend-config at init time.
    key            = "prod/vercel/terraform.tfstate"
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

# API token read from env var VERCEL_API_TOKEN (GH secret).
# Team ID optional; set via TF_VAR_vercel_team_id if applicable.
provider "vercel" {
  team = var.vercel_team_id
}
