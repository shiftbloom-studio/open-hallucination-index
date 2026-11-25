# Roll back a deploy

Production is a single Cloudflare Worker (`ohi-cloudflare`), so a rollback is
a Worker version rollback. Static assets ship with the Worker version.

```sh
cd cloudflare/ohi-worker
pnpm wrangler deployments list          # find the last good version id
pnpm wrangler rollback <version-id>     # make it live again
```

Then check the health probes:

```sh
curl -fsS https://ohi.shiftbloom.studio/health/live
curl -fsS https://ohi.shiftbloom.studio/health/ready
curl -fsS https://ohi.shiftbloom.studio/health/deep
```

D1 migrations are not rolled back by this. If a migration caused the incident,
write a forward migration that fixes it and deploy again.
