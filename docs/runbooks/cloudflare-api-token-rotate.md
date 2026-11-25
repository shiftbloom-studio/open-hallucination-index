# Cloudflare API token: scopes and rotation

The deploy workflow (`.github/workflows/cloudflare-production.yml`) needs one
API token stored as the `CLOUDFLARE_API_TOKEN` secret of the `production`
environment, plus `CLOUDFLARE_ACCOUNT_ID` and `OHI_D1_DATABASE_ID`.

## Scopes

Account scoped:

- Workers Scripts: Edit
- Workers AI: Read
- D1: Edit
- Vectorize: Edit
- Queues: Edit
- Workers R2 Storage: Edit
- Account Settings: Read

Zone scoped (the zone that serves the Worker):

- Workers Routes: Edit
- Zone: Read

## Create or rotate

1. Create the token in the Cloudflare dashboard under My Profile, API Tokens.
2. Store it as the `CLOUDFLARE_API_TOKEN` environment secret in GitHub.
3. Check it:
   ```sh
   curl -H "Authorization: Bearer $TOKEN" \
        https://api.cloudflare.com/client/v4/user/tokens/verify
   ```
   The response should say `"status":"active"`.
4. Revoke the old token in the dashboard.

Never commit a token to the repository.
