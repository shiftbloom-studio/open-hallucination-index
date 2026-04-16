# Google Cloud — Gemini API Quota Cap

Backstops our Phase 1 "unlimited Gemini" posture (spec §9.1 R10).

## One-time setup

1. Google Cloud Console → IAM & Admin → Quotas & System Limits.
2. Filter: `Generative Language API` in the project that issued
   `ohi/gemini-api-key`.
3. For each of the quotas below, click "Edit quotas" and set a hard cap.
   Pick values that match the monthly budget you're willing to spend:

| Quota | Suggested cap |
|---|---|
| `Generate content API requests per day` | 5000 |
| `Input tokens per minute` | 500000 |
| `Output tokens per minute` | 50000 |

4. Click "Submit request" (Google may auto-approve; sometimes takes minutes).

## Verification

```bash
gcloud alpha services quota list \
  --service=generativelanguage.googleapis.com \
  --consumer=projects/<project-number>
```

## Operator contract

Review Google Cloud Billing weekly for the first month post-launch to spot
unexpected spend before it becomes material.
