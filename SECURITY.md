# Security policy

## Supported versions

Only the current `main` branch and the live deployment at
<https://ohi.shiftbloom.studio> receive security fixes.

## Reporting a vulnerability

Please do not open a public issue for security problems.

Email **safety@shiftbloom.studio** with:

1. what you found and where,
2. steps or a request that reproduces it,
3. what an attacker could do with it.

You can also use GitHub's private vulnerability reporting on this repository.

We reply within 3 working days and aim to ship a fix within 30 days of
confirming the issue. We are happy to credit you once the fix is live.

## Scope

In scope: the Worker in `cloudflare/ohi-worker`, the frontend in
`src/frontend`, the MCP server in `src/ohi-mcp-server`, and the public
endpoints `/api/v2/*`, `/mcp` and `/health/*`.

Out of scope: rate limit or Turnstile challenges working as intended,
reports from automated scanners without a working proof, and attacks on
Cloudflare itself.
