# Current Architecture

This document captures the current architecture that is actively used in this
repository across:

- production: hybrid deployment across Vercel, Cloudflare, AWS, and PC-hosted services
- local development: browser + local app processes with Docker-backed infra

The diagrams below are based on the current repo state in:

- `README.md`
- `docs/FRONTEND.md`
- `infra/terraform/compute/`
- `infra/terraform/cloudflare/`
- `infra/terraform/storage/`
- `infra/terraform/jobs/`
- `infra/terraform/vercel/`
- `docker/compose/pc-data.yml`
- `docker/compose/docker-compose.yml`

## 1. Production Architecture

```mermaid
flowchart LR
    user["User Browser"]
    site["Vercel Static Frontend\nohi.shiftbloom.studio"]
    edge["Cloudflare Edge\nDNS + WAF + rate limiting"]

    subgraph aws["AWS"]
        apigw["API Gateway HTTP API\ncustom domain"]
        lambda["Lambda Container\nOHI API"]
        ddb["DynamoDB\nverify jobs"]
        sm["Secrets Manager"]
        s3["S3 Artifacts Buckets"]
        logs["CloudWatch Logs"]
    end

    subgraph pc["PC-side Services via Cloudflare Tunnel"]
        cfaccess["Cloudflare Tunnel + Access"]
        qdrant["Qdrant\npassage vectors"]
        embed["pc-embed\nMiniLM embeddings"]
        pg["PostgREST / Postgres"]
        redis["Webdis / Redis\navailable, disabled in prod path"]
    end

    aura["Neo4j Aura Pro"]
    wiki["MediaWiki / Wikipedia"]
    gemini["Gemini API"]
    openai["OpenAI API\nWave 3 claim-claim NLI"]

    user -->|"GET /"| site
    user -->|"POST/GET /api/v2/*"| edge
    edge -->|"ohi-api.shiftbloom.studio"| apigw
    apigw --> lambda

    lambda --> ddb
    lambda --> sm
    lambda --> s3
    lambda --> logs

    lambda --> aura
    lambda --> wiki
    lambda --> gemini
    lambda -.-> openai

    lambda -->|"HTTPS + service token"| cfaccess
    cfaccess --> qdrant
    cfaccess --> embed
    cfaccess -.-> pg
    cfaccess -.-> redis
```

### Production flow summary

1. The browser loads the statically exported frontend from Vercel.
2. The browser talks directly to the public API domain for verify polling.
3. Cloudflare protects the public API entrypoint and forwards traffic to the
   API Gateway custom domain.
4. API Gateway invokes the Lambda-based OHI API.
5. Lambda stores async job state in DynamoDB and reads secrets from Secrets
   Manager.
6. Verification uses Gemini, MediaWiki, Neo4j Aura, and the PC-hosted vector
   and embedding services exposed through Cloudflare Tunnel + Access.

## 2. Local Development Architecture

```mermaid
flowchart LR
    browser["Developer Browser"]

    subgraph app["Local App Layer"]
        next["Next.js Dev Server\nlocalhost:3000"]
        api["FastAPI / ohi-server\nlocalhost:8080"]
    end

    subgraph infra["Local Infra"]
        neo4j["Neo4j\n7474 / 7687"]
        qdrant["Qdrant\n6333 / 6334"]
        redis["Redis\n6379"]
        mcp["OHI MCP Server\n8083"]
        vllm["vLLM\n8000"]
        postgres["Postgres 16\n5432"]
        minio["MinIO / artifact store\n9000 / 9001"]
    end

    subgraph optional["Optional Full Docker Edge"]
        nginx["Nginx\n80 / 443"]
        fe["Frontend Container"]
        api_docker["API Container"]
    end

    browser -->|"http://localhost:3000"| next
    next -->|"verify + status requests"| api

    api --> neo4j
    api --> qdrant
    api --> redis
    api --> mcp
    api --> vllm
    api -.-> postgres
    api -.-> minio

    browser -.->|"optional full-docker access"| nginx
    nginx --> fe
    nginx --> api_docker
```

### Local flow summary

- Preferred feature workflow is local-first: run Next.js and FastAPI locally,
  use Docker for supporting infra.
- The older full Docker stack still exists for end-to-end validation and routes
  browser traffic through Nginx.
- The local v2 support stack also includes Postgres and MinIO for algorithm and
  artifact workflows.

## 3. PC-side Data Stack Used by Production

```mermaid
flowchart LR
    tunnel["cloudflared"]
    neo4j["Neo4j"]
    qdrant["Qdrant"]
    postgres["Postgres"]
    postgrest["PostgREST"]
    redis["Redis"]
    webdis["Webdis"]
    embed["pc-embed"]

    tunnel --> neo4j
    tunnel --> qdrant
    tunnel --> postgrest
    tunnel --> webdis
    tunnel --> embed

    postgrest --> postgres
    webdis --> redis
```

This is the local machine-side service group that is exposed selectively to AWS
through Cloudflare Tunnel in the current hybrid production setup.
