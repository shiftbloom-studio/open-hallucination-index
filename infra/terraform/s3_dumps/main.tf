# Wave 3 Stream C dump-cache bucket
# ---------------------------------
#
# Mirror of the Wikimedia dump archive so Stream I re-runs (corpus
# ingestion on Fabian's PC) don't re-hammer the public dump mirror.
# Created out-of-band via AWS CLI during the Wave 3 autonomous deploy
# (name = `ohi-corpus-dumps-349744179866`, eu-central-1) so Stream I
# isn't blocked on a separate TF apply. To bring it under TF state,
# run:
#
#   cd infra/terraform/s3_dumps
#   terraform init -backend-config=...
#   terraform import aws_s3_bucket.dumps ohi-corpus-dumps-349744179866
#   terraform import aws_s3_bucket_versioning.dumps ohi-corpus-dumps-349744179866
#   terraform import aws_s3_bucket_public_access_block.dumps ohi-corpus-dumps-349744179866
#
# after which subsequent ``terraform apply`` runs will manage the bucket
# metadata. Contents are ~130 GB (enwiki + Wikidata snapshots) and
# **not** versioned beyond the default AWS CLI-enabled versioning (one
# snapshot per pin date is enough; older snapshots can be purged
# manually when space becomes an issue).

resource "aws_s3_bucket" "dumps" {
  bucket = "${local.prefix}-corpus-dumps-${local.account_id}"

  tags = {
    Purpose = "wiki-dump-cache"
  }
}

resource "aws_s3_bucket_versioning" "dumps" {
  bucket = aws_s3_bucket.dumps.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "dumps" {
  bucket = aws_s3_bucket.dumps.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "dumps" {
  bucket                  = aws_s3_bucket.dumps.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Lifecycle: older snapshot versions expire after 180 days so we
# don't silently accumulate multi-hundred-GB of historical corpus
# dumps when Stream I runs repeatedly.
resource "aws_s3_bucket_lifecycle_configuration" "dumps" {
  bucket = aws_s3_bucket.dumps.id

  rule {
    id     = "expire-old-dump-versions"
    status = "Enabled"

    filter {}

    noncurrent_version_expiration {
      noncurrent_days = 180
    }
  }
}
