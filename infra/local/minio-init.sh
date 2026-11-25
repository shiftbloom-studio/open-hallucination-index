#!/bin/sh
set -eu
mc alias set local http://minio:9000 ohi-local ohi-local-dev-key
mc mb --ignore-existing local/ohi-artifacts
mc mb --ignore-existing local/ohi-artifacts/nli-heads
mc mb --ignore-existing local/ohi-artifacts/calibration
mc mb --ignore-existing local/ohi-artifacts/source-cred
mc mb --ignore-existing local/ohi-artifacts/retraining-reports
mc mb --ignore-existing local/ohi-artifacts/eval-snapshots
echo "MinIO buckets initialised."
