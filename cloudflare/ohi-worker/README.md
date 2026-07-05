# OHI Cloudflare Worker

Cloudflare-native deployment for Open Hallucination Index at
`https://ohi.shiftbloom.studio`.

The Worker serves the static Next.js export and handles `/api/v2/*`,
`/health/*`, and `/mcp` without AWS, Vercel, local tunnels, or local
compute.

## Products

- Workers Static Assets: frontend hosting
- Workers AI: claim decomposition, NLI, embeddings, reranking
- AI Gateway: enabled through Workers AI `gateway: { id: "default" }`
- D1: job history, feedback, evidence cache
- Durable Objects: per-job strongly consistent status and MCP agent state
- Queues: async verification handoff
- Vectorize: hosted semantic evidence cache
- Workers custom domain: `ohi.shiftbloom.studio`

## Commands

```sh
pnpm install
pnpm run types
pnpm run check
pnpm run build
pnpm run deploy
```

Before the first deploy, create and bind:

```sh
wrangler d1 create ohi-prod
wrangler vectorize create ohi-evidence-bge-m3 --dimensions=1024 --metric=cosine
wrangler queues create ohi-verify
wrangler queues create ohi-verify-dlq
wrangler d1 migrations apply ohi-prod --remote
```
