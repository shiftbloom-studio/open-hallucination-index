region                   = "eu-central-1"
memory_mb                = 2048
timeout_s                = 180 # Phase 2 D1: Gemini 3 Pro NLI fan-out needs headroom (see variables.tf rationale).
log_retention_days       = 7
async_verify_ttl_seconds = 3600 # Stream D2: DynamoDB verify-job TTL, see variables.tf rationale.
