# Rotate a Secret (generic procedure)

Applies to: gemini-api-key, internal-bearer-token, labeler-tokens,
pc-origin-credentials, neo4j-credentials.

## Generic steps

1. Generate a new value:
   ```bash
   openssl rand -base64 64   # for opaque tokens
   ```
   For Gemini: generate in Google Cloud Console → APIs → Credentials.

2. Update AWS Secrets Manager:
   ```bash
   aws secretsmanager put-secret-value \
     --secret-id ohi/<secret-name> \
     --secret-string "<new-value>"
   ```

3. Lambda SecretsLoader TTL is 10 min; within 10 min every Lambda cold start
   picks up the new value. Force immediate pickup:
   ```bash
   aws lambda update-function-configuration --function-name ohi-api --description "rotate-$(date -u +%s)"
   ```
   (An arbitrary config change forces a new version → cold start → fresh read.)

4. Update any external consumers (MCP server, CI), then revoke the old value.

## BACKUP BEFORE ROTATING

`recovery_window_in_days = 0` on all secrets (§4.2 of spec). `put-secret-value`
REPLACES the value — there is NO recover flow. Always save the old value to
your password manager before `put`.
