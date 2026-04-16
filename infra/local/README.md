# Local OHI v2 dev stack (Phase 0)

Postgres 16 + MinIO for local development of the OHI v2 algorithm. Mirrors
the production-target storage surface without requiring AWS access.

## Why

Phase 0 of the v2 plan needs the spec §12 Postgres schema (verifications,
claim_verdicts, feedback_pending, calibration_set, disputed_claims_queue,
retraining_runs) and an S3-compatible artifact store (for NLI heads,
conformal quantile snapshots, calibration reports, etc.). This compose file
brings both up with the schema pre-loaded and the `ohi-artifacts` bucket
pre-created.

## Bring up

```
docker compose -f infra/local/docker-compose.dev.yml up -d
```

Services:

| Service  | Host port | Purpose                                       |
|----------|-----------|-----------------------------------------------|
| postgres | 5432      | v2 feedback store + calibration set + verdicts |
| minio    | 9000      | S3-compatible artifact store (API)            |
| minio    | 9001      | MinIO console (browser UI)                    |

The `minio-init` one-shot container creates the `ohi-artifacts` bucket +
subfolders the moment MinIO is healthy.

### Credentials (local dev only — do not reuse elsewhere)

- Postgres: `ohi` / `ohi-local-dev`, database `ohi`
- MinIO root: `ohi-local` / `ohi-local-dev-key`

These match the defaults in `.env.example`. Export them into your shell
(`source .env`) or let the smoke test pick them up from defaults.

## Verify

```
pytest tests/infra/ -v -m infra
```

Both tests should PASS:

- `test_postgres_reachable_with_required_schemas` — Postgres is up and every
  spec §12 table exists in the `public` schema.
- `test_minio_reachable_with_required_buckets` — MinIO is up and the
  `ohi-artifacts` bucket exists.

## Tear down

Keep data:
```
docker compose -f infra/local/docker-compose.dev.yml down
```

Wipe data (starts fresh next time):
```
docker compose -f infra/local/docker-compose.dev.yml down -v
rm -rf infra/local/data
```

## What's next

- Task 0.2: benchmark harness skeleton.
- Task 0.3: capture v1 baseline numbers into `benchmark_results/`.
- Task 0.4: CI gate that fails the build without a committed baseline.

See [../../docs/superpowers/plans/2026-04-16-ohi-v2-implementation.md](../../docs/superpowers/plans/2026-04-16-ohi-v2-implementation.md).

## Changing the schema

The schema in `postgres-init.sql` is the source of truth for local dev and
must mirror spec §12. If you need to change it, update the spec first, then
this file, then wipe the Postgres volume (`-v` on teardown) and re-create
the stack so init runs again.

Postgres only runs `docker-entrypoint-initdb.d/*.sql` on a FRESH volume.
Modifying `postgres-init.sql` on an existing volume has no effect until the
volume is wiped.
