# GitHub Actions Secrets

Um die CI/CD Pipeline vollständig zu nutzen, konfiguriere diese Secrets in deinem Repository unter **Settings → Secrets and variables → Actions**:

## Erforderlich für Build

| Secret | Beschreibung |
|--------|--------------|
| `NEXT_PUBLIC_SUPABASE_URL` | Supabase Projekt-URL |
| `NEXT_PUBLIC_SUPABASE_ANON_KEY` | Supabase Anonymous Key |

## Optional für Vercel Deployment

| Secret | Beschreibung | Wie erhalten |
|--------|--------------|--------------|
| `VERCEL_TOKEN` | Vercel API Token | [Vercel Dashboard → Settings → Tokens](https://vercel.com/account/tokens) |
| `VERCEL_ORG_ID` | Organisation/Team ID | `vercel link` → `.vercel/project.json` |
| `VERCEL_PROJECT_ID` | Projekt ID | `vercel link` → `.vercel/project.json` |

## Optional für Coverage Reports

| Secret | Beschreibung |
|--------|--------------|
| `CODECOV_TOKEN` | Codecov Upload Token (optional für private Repos) |

## Vercel Setup

1. Installiere Vercel CLI: `npm i -g vercel`
2. Verknüpfe dein Projekt: `vercel link`
3. Kopiere die IDs aus `.vercel/project.json`
4. Erstelle einen Token unter vercel.com/account/tokens
