# Cloudflare API Token — Scopes + Rotation

## Required scopes (on the CF dashboard → My Profile → API Tokens → Create Token)

### Account-scoped:
- `Account:Cloudflare Tunnel:Edit`
- `Account:Access: Apps and Policies:Edit`
- `Account:Account Settings:Read`

### Zone-scoped (to `ohi.shiftbloom.studio`):
- `Zone:Read`
- `Zone:Zone Settings:Edit`
- `Zone:DNS:Edit`
- `Zone:Zone WAF:Edit`
- `Zone:Page Rules:Edit` (for Transform Rules)

## Create the token

1. Create token in CF dashboard.
2. Save the token value (shown ONCE).
3. Update GitHub repo secret `CLOUDFLARE_API_TOKEN`:
   ```bash
   gh secret set CLOUDFLARE_API_TOKEN < /tmp/token.txt
   ```
4. Verify:
   ```bash
   curl -H "Authorization: Bearer $(cat /tmp/token.txt)" \
        https://api.cloudflare.com/client/v4/user/tokens/verify
   ```
   Expect `"status":"active"`.

## Rotation

Same as creation, then revoke the old token from the CF dashboard.
Never commit the token to any file. Store exclusively in GH secrets.
