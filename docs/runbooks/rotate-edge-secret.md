# Rotate the CF Edge Secret

The edge secret is enforced on BOTH sides: AWS Secrets Manager (`ohi/cf-edge-secret`)
for Lambda middleware, and Cloudflare Transform Rule for header injection.

## Procedure

1. Generate a new value:
   ```bash
   openssl rand -hex 32 > /tmp/new-edge-secret.txt
   ```

2. Update Secrets Manager:
   ```bash
   aws secretsmanager put-secret-value --secret-id ohi/cf-edge-secret \
     --secret-string "$(cat /tmp/new-edge-secret.txt)"
   ```

3. Update Cloudflare — via `workflow_dispatch` on `infra-apply.yml` with
   layer=cloudflare, after setting the `CF_EDGE_SECRET` GH secret to the new value.

4. Wait 10 min (Lambda TTL). During the overlap, SOME requests may 403 if
   Cloudflare's new header arrives at a Lambda that still has the old cached
   value. Mitigate by doing steps 2 and 3 within ~30 seconds.

5. Verify: `curl -i https://ohi.shiftbloom.studio/health/live` should return 200.
   Direct-to-Function-URL should return 403.

## Force Lambda to re-read immediately

To skip the 10-min TTL window:
```bash
aws lambda update-function-configuration --function-name ohi-api \
  --description "edge-secret-rotate-$(date -u +%s)"
```
This forces a new Lambda version, so every subsequent invocation cold-starts
and fetches the new secret value.
