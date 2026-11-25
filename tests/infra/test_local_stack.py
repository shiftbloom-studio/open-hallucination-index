"""Smoke tests for the local Phase 0 dev stack (Postgres + MinIO).

These tests are opt-in (`-m infra`) and require the stack to be up:

    docker compose -f infra/local/docker-compose.dev.yml up -d
    pytest tests/infra/ -v -m infra

A fast connect timeout (3s) is set so the tests fail quickly when the stack
is not running, rather than hanging on the default TCP connect timeout.
"""

from __future__ import annotations

import os

import psycopg
import pytest
from minio import Minio

pytestmark = pytest.mark.infra  # opt-in marker; CI gates this on docker availability


def test_postgres_reachable_with_required_schemas():
    """Postgres is reachable and the spec §12 tables exist after init."""
    conn = psycopg.connect(
        host=os.environ.get("POSTGRES_HOST", "localhost"),
        port=int(os.environ.get("POSTGRES_PORT", "5432")),
        dbname=os.environ.get("POSTGRES_DB", "ohi"),
        user=os.environ.get("POSTGRES_USER", "ohi"),
        password=os.environ.get("POSTGRES_PASSWORD", "ohi-local-dev"),
        connect_timeout=3,
    )
    with conn, conn.cursor() as cur:
        cur.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'public' ORDER BY table_name;"
        )
        tables = {row[0] for row in cur.fetchall()}
    expected = {
        "verifications",
        "claim_verdicts",
        "feedback_pending",
        "calibration_set",
        "disputed_claims_queue",
        "retraining_runs",
    }
    missing = expected - tables
    assert not missing, f"Missing tables: {missing}"


def test_minio_reachable_with_required_buckets():
    """MinIO is reachable and the artifact bucket exists."""
    import urllib3

    http_client = urllib3.PoolManager(timeout=urllib3.Timeout(connect=3, read=3))
    client = Minio(
        os.environ.get("S3_ENDPOINT", "localhost:9000"),
        access_key=os.environ.get("S3_ACCESS_KEY", "ohi-local"),
        secret_key=os.environ.get("S3_SECRET_KEY", "ohi-local-dev-key"),
        secure=False,
        http_client=http_client,
    )
    assert client.bucket_exists("ohi-artifacts"), "ohi-artifacts bucket missing"
