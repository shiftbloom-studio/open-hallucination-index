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
│   ├── pc-data.yml              # Data services for the Python stack (Neo4j, Qdrant,
│   │                            #   Postgres, Redis, pc-embed)
│   ├── pc-data.local-dev.yml    # Overlay that binds 127.0.0.1 host ports for dev
│   └── docker-compose.yml       # Full stack (API + MCP + vLLM + Nginx + TLS)
├── api/                         # FastAPI image (src/api)
├── pc-embed/                    # sentence-transformers HTTP embedder
├── cloudflared/                 # Cloudflare Tunnel image, if you expose a local stack
├── mcp-server/                  # Standalone MCP server image
└── nginx/                       # Reverse-proxy config for the full stack
```

---

## Canonical stacks

### Production

Production runs on Cloudflare, see
[docs/CURRENT_ARCHITECTURE.md](../docs/CURRENT_ARCHITECTURE.md). The
containers here run the Python research stack (`src/api`) with Neo4j,
Qdrant and friends on your own machine.

### Local dev

```bash
# Host-ports-exposed overlay for local dev — skips cloudflared, binds
# 127.0.0.1:7474 etc..
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

See [docs/runbooks/pc-compose-start.md](../docs/runbooks/pc-compose-start.md).

---

## Related docs

- [../docs/CURRENT_ARCHITECTURE.md](../docs/CURRENT_ARCHITECTURE.md) — **SSoT** for topology
- [../docs/runbooks/pc-compose-start.md](../docs/runbooks/pc-compose-start.md) — PC stack bring-up
- [../docs/runbooks/rollback-deploy.md](../docs/runbooks/rollback-deploy.md) — Worker rollback
