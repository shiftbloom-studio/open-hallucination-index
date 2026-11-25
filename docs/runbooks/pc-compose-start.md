# PC Compose Start / Stop Runbook

## Prereqs

- Docker Desktop (Windows) or Docker Engine (Linux) installed.
- Optional: a Cloudflare Tunnel token, only if you want to expose the stack.
  For local work use the `local-dev` overlay instead.

## First-time setup

1. Copy env template:
   ```bash
   cp docker/compose/.env.pc-data.example docker/compose/.env.pc-data
   ```
2. Fill all values in `.env.pc-data`. Sources:
   - `TUNNEL_TOKEN` → from the Cloudflare dashboard (Zero Trust, Tunnels), optional
   - `NEO4J_AUTH`, Postgres and PostgREST JWT, `WEBDIS_HTTP_AUTH` → generate strong
     values and keep them in your password manager
3. Start the stack:
   ```bash
   docker compose -f docker/compose/pc-data.yml --profile pc-prod up -d
   ```
4. Verify:
   ```bash
   docker compose -f docker/compose/pc-data.yml ps
   docker compose -f docker/compose/pc-data.yml logs cloudflared | grep "Connection established"
   ```
   You should see the tunnel come up and 4 ingress rules registered.

## Day-to-day

Start: `docker compose -f docker/compose/pc-data.yml --profile pc-prod up -d`
Stop:  `docker compose -f docker/compose/pc-data.yml --profile pc-prod down`
Logs:  `docker compose -f docker/compose/pc-data.yml logs -f cloudflared`

## Local-dev mode (for feature work, NOT prod)

```bash
docker compose -f docker/compose/pc-data.yml \
               -f docker/compose/pc-data.local-dev.yml \
               --profile local-dev up -d
```
- Host ports bound on 127.0.0.1 only.
- `cloudflared` NOT started.
- Separate volumes (`*-dev`) - dev data doesn't touch prod data.

Stop prod before starting local-dev (volumes are different but the container
names could collide if you forgot `--profile`).

## When to restart

- PC reboot: `cloudflared` auto-restarts via `restart: unless-stopped`.
- After rotating `TUNNEL_TOKEN`: edit `.env.pc-data`, then
  `docker compose -f docker/compose/pc-data.yml --profile pc-prod up -d --force-recreate cloudflared`.
- After Neo4j password rotation: `up -d --force-recreate neo4j` and bolt clients
  will need to re-auth.
