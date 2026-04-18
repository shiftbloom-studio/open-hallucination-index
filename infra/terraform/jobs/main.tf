# ---------------------------------------------------------------------------
# ohi-verify-jobs — per-request async verification job table (Stream D2).
#
# POST /api/v2/verify writes a pending record here, self-invokes the Lambda
# asynchronously with the job_id, and returns 202 + job_id to the caller.
# The async handler drives pipeline.verify() and updates the record at each
# of the five natural phase boundaries (decomposing / retrieving_evidence /
# classifying / calibrating / assembling) so the polling client can surface
# progress. The client then polls GET /api/v2/verify/status/{job_id} until
# status transitions to done or error.
#
# PAY_PER_REQUEST billing is chosen over provisioned for two reasons:
#   1. The workload is spiky — a handful of writes per verify, then idle.
#      Provisioned throughput would have to be sized for peak and would
#      bill idle time.
#   2. Monthly cost at OHI's scale (~1k verifies/day * ~7 writes/job *
#      ~2KB per item) rounds to cents — well under the $1 floor of any
#      provisioned plan.
#
# Encryption uses the AWS-managed key (KMS alias `aws/dynamodb`) to avoid
# spinning a dedicated CMK — there's nothing secret in the records beyond
# user-submitted text, which retention middleware already gates.
# ---------------------------------------------------------------------------
resource "aws_dynamodb_table" "verify_jobs" {
  name         = "${local.prefix}-verify-jobs"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "job_id"

  attribute {
    name = "job_id"
    type = "S"
  }

  # TTL reaps completed / abandoned jobs within ~48h of their `ttl` epoch.
  # Producers set ttl = created_at + 1h so a successful verify's record is
  # available long enough for the UI to poll (3 min abs cap) plus a safety
  # margin for debugging, then garbage-collected without a cron.
  ttl {
    attribute_name = var.ttl_attribute
    enabled        = true
  }

  server_side_encryption {
    enabled     = true
    kms_key_arn = null # AWS-managed key; no CMK cost.
  }

  point_in_time_recovery {
    enabled = false
  }
}
