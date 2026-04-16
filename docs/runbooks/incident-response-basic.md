# Incident Response — Basic

## Triage flowchart

1. **User report / alarm email arrives.** Check what triggered:
   - Budget alarm → §Budget
   - Lambda 5xx rate → §Lambda errors
   - PC origin timeout → §PC unreachable
   - WAF block spike → §WAF (via CF dashboard, not SNS)

2. **Verify scope.** Curl each tier:
   ```bash
   curl -i https://ohi.shiftbloom.studio/health/live       # CF + Lambda public path
   curl -i -H "X-OHI-Edge-Secret: $(aws secretsmanager get-secret-value --secret-id ohi/cf-edge-secret --query SecretString --output text)" \
         https://<lambda-fn-url>/health/live               # Direct Lambda (should 200)
   ```

3. **Check CloudWatch dashboard:** `ohi-prod` in `eu-central-1`.

## § PC unreachable

- SSH or RDP to the PC.
- `docker compose -f docker/compose/pc-data.yml --profile pc-prod ps`
- If `cloudflared` is `Exited`: `docker compose ... up -d cloudflared` and
  check logs for auth errors (token may have been rotated without updating `.env.pc-data`).
- If ISP is down: nothing to do. Cloudflare will return 502 until it's back.

## § Lambda errors

- CloudWatch Logs: `/aws/lambda/ohi-api`, filter on `level = ERROR`.
- Top cause: missing/malformed secret (check SecretsLoader log).
- Second: Gemini API key invalid (quota or revocation).
- Rollback: see `rollback-deploy.md`.

## § Budget alarm

- Don't panic. Check Cost Explorer filter `CostCenter=ohi`.
- Most likely culprit: Lambda duration spike (cold-start run) or S3 egress
  from public calibration bucket (popular artifact).
- If runaway, STOP the Lambda:
  ```bash
  aws lambda put-function-concurrency --function-name ohi-api --reserved-concurrent-executions 0
  ```
  Reset when root cause known.

## § WAF / abuse

- Cloudflare dashboard → Security → Events. Filter to last 1h.
- Add a custom block rule if a specific IP range or pattern is hammering.
- Raise a temporary `managed_challenge` rule on `/api/v2/verify` if signal is mixed.
