# OHI Docker Assets

Docker build contexts and compose stacks for the Open Hallucination
Index. This directory is NOT the authoritative source for
architecture — see
[docs/CURRENT_ARCHITECTURE.md](../docs/CURRENT_ARCHITECTURE.md) for
the SSoT on what runs where in production vs. local dev.

---

## What's in here

```
docker/
├── compose/
│   ├── pc-data.yml              # PC-hosted data services (Qdrant is in prod path;
│   │                            #   Neo4j/pc-embed/Postgres/Redis are legacy or dev-only)
│   ├── pc-data.local-dev.yml    # Overlay that binds 127.0.0.1 host ports for dev,
│   │                            #   no Cloudflare Tunnel
│   └── docker-compose.yml       # Legacy full-stack compose (API + MCP + vLLM +
│                                #   Nginx + TLS) — retained for e2e smoke, not the
│                                #   canonical dev workflow
├── lambda/                      # Lambda container image build context (deployed image)
├── pc-embed/                    # sentence-transformers HTTP embedder; used for
│                                #   local dev + as legacy remote backend. Prod uses
│                                #   AWS Bedrock Titan Text v2.
├── cloudflared/                 # Cloudflare Tunnel image
├── api/                         # (legacy) API Dockerfile — prod now ships as Lambda
├── mcp-server/                  # (legacy) standalone MCP server
├── nginx/                       # (legacy) reverse-proxy configs
└── data/                        # Persistent-volume mounts for legacy compose
```

---

## Canonical stacks

### Production (AWS Lambda + PC Qdrant)

See
[docs/CURRENT_ARCHITECTURE.md §1](../docs/CURRENT_ARCHITECTURE.md#1-production-architecture).
Lambda container image is built from `docker/lambda/` by the
`.github/workflows/v2-main-deploy.yml` workflow and pushed to ECR on
every `main` push. The only PC-side container reached from Lambda in
prod is **Qdrant**, via Cloudflare Tunnel at
`ohi-qdrant.shiftbloom.studio`.

### PC-side prod services

```bash
# Starts Qdrant (+ the other containers — only Qdrant is in prod path)
# plus cloudflared for tunnel exposure.
docker compose -f docker/compose/pc-data.yml --profile pc-prod up -d
```

Service-by-service status is in
[docs/CURRENT_ARCHITECTURE.md §3](../docs/CURRENT_ARCHITECTURE.md#3-pc-side-data-stack).

Runbook: [docs/runbooks/pc-compose-start.md](../docs/runbooks/pc-compose-start.md).

### Local dev

```bash
# Host-ports-exposed overlay for local dev — skips cloudflared, binds
# 127.0.0.1:7474 etc., does NOT use AWS.
docker compose -f docker/compose/pc-data.yml \
  -f docker/compose/pc-data.local-dev.yml \
  --profile local-dev up -d
```

Then run the FastAPI app against those containers:

```bash
cd src/api
pip install -e ".[dev]"
export OHI_EMBEDDING_BACKEND=local   # in-process sentence-transformers
ohi-server                           # http://localhost:8080
```

See [docs/CURRENT_ARCHITECTURE.md §2](../docs/CURRENT_ARCHITECTURE.md#2-local-development-architecture).

---

## Image registry

- **Prod**: AWS ECR (`349744179866.dkr.ecr.eu-central-1.amazonaws.com/ohi-api`),
  tags `:prod` (mutable) + `:sha-<commit>` (immutable). Lambda pins an
  immutable digest.
- **Legacy**: GitHub Container Registry paths referenced in the
  legacy `docker-compose.yml` are not maintained; the active Lambda
  image is built from `docker/lambda/` only.

---

## Related docs

- [../docs/CURRENT_ARCHITECTURE.md](../docs/CURRENT_ARCHITECTURE.md) — **SSoT** for topology
- [../CLAUDE.md](../CLAUDE.md) — agent operational knowledge + traps
- [../docs/runbooks/pc-compose-start.md](../docs/runbooks/pc-compose-start.md) — PC stack bring-up
- [../docs/runbooks/rollback-deploy.md](../docs/runbooks/rollback-deploy.md) — emergency Lambda rollback
