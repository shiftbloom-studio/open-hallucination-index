module "shared" {
  source = "../_shared"
  layer  = "vercel"
  region = var.region
}

data "aws_caller_identity" "current" {}
