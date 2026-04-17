# Vercel Setup Runbook

Covers one-time Vercel account setup and the GH secrets the infra pipeline
needs. Assumes bootstrap has been applied and the GitHub repo secret
`CLOUDFLARE_API_TOKEN` is already set.

## Prereqs

- Vercel account at [vercel.com](https://vercel.com). Free "Hobby" tier is
  sufficient for OHI's expected traffic (<100 users/month, static export).
- GitHub repo access to `shiftbloom-studio/open-hallucination-index` (Vercel
  will request repo permissions via its GitHub App).

## One-time setup

1. **Create Vercel account / log in.**

2. **Install the Vercel GitHub App** on `shiftbloom-studio/open-hallucination-index`.
   - Vercel dashboard → Settings → Git → Install GitHub App
   - Grant access to the repo only (no org-wide access needed).

3. **Create a Vercel API token:**
   - Vercel dashboard → Settings → Tokens → Create Token
   - Scope: full account (or team if using a team)
   - Expiration: 1 year (set a calendar reminder to rotate)
   - Copy the token string — shown only once.

4. **Add GitHub repo secrets + variables:**
   ```bash
   gh secret set VERCEL_API_TOKEN        # paste token from step 3
   gh secret set VERCEL_TEAM_ID          # empty string if personal account
   # VERCEL_VERIFICATION_TOKEN is only needed if Vercel's dashboard shows a
   # TXT challenge after first apply — see step 7.
   ```

5. **Dispatch the vercel layer:**
   ```bash
   gh workflow run infra-apply.yml -f layer=vercel -f confirm=apply
   ```
   First apply creates the Vercel project + connects to GitHub + configures
   the apex domain.

6. **Verify the Vercel project exists:**
   - Vercel dashboard → Projects → `ohi-frontend` should show up.
   - Git connection should list the OHI repo, production branch `main`.

7. **Check domain verification status:**
   - Vercel dashboard → Projects → `ohi-frontend` → Settings → Domains →
     `ohi.shiftbloom.studio`
   - If status is "Valid Configuration": you're done. CNAME flattening at
     the CF apex (set up by the `cloudflare/` layer) satisfied Vercel.
   - If status asks for a **TXT challenge** (`_vercel` with a specific value):
     ```bash
     gh secret set VERCEL_VERIFICATION_TOKEN   # paste the TXT value
     gh workflow run infra-apply.yml -f layer=cloudflare -f confirm=apply
     ```
     This creates the `_vercel.ohi.shiftbloom.studio` TXT record at CF.
     Wait a few minutes, then re-check domain status in Vercel.

8. **First deploy:**
   - Any push to `main` triggers a Vercel deployment automatically.
   - Preview deployments are created for every PR.
   - Check status: `gh workflow view` (for CI) and Vercel dashboard (for
     Vercel builds).

## Day-to-day

- Vercel redeploys on every `main` push. No human action needed.
- Env var changes: edit `infra/terraform/vercel/project.tf`,
  `gh workflow run infra-apply.yml -f layer=vercel -f confirm=apply`.
- New env var in code: add to `project.tf` alongside the existing
  `vercel_project_environment_variable` blocks.

## API Token rotation

1. Create a new token in Vercel dashboard.
2. `gh secret set VERCEL_API_TOKEN` with the new value.
3. Revoke the old token in Vercel dashboard.
4. Next `infra-apply.yml` run uses the new token automatically.

## Disaster recovery

If the Vercel project is deleted:
- `gh workflow run infra-apply.yml -f layer=vercel -f confirm=apply` recreates it.
- The project history / analytics are lost; deployments will rebuild from Git.
- DNS (apex CNAME) does not change — Vercel auto-attaches to the domain.
