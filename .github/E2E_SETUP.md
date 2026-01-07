# E2E Test Setup

## GitHub Secrets Configuration

Die folgenden Secrets müssen in den GitHub Repository Settings konfiguriert werden, damit die E2E Tests in der CI/CD Pipeline erfolgreich laufen:

### Erforderliche Secrets

| Secret Name | Beschreibung | Beispiel |
|------------|--------------|----------|
| `NEXT_PUBLIC_SUPABASE_URL` | Supabase Project URL | `https://xxxxx.supabase.co` |
| `NEXT_PUBLIC_SUPABASE_ANON_KEY` | Supabase Anonymous/Public Key (sicher für Browser) | `eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...` |
| `SUPABASE_SERVICE_ROLE_KEY` | Supabase Service Role Key (nur Server-side!) | `eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...` |
| `DATABASE_URL` | PostgreSQL Verbindungsstring | `postgresql://user:password@host:5432/db` |
| `DEFAULT_API_URL` | Open Hallucination Index API URL | `https://openhallucination.xyz/api/` |
| `DEFAULT_API_KEY` | API Key für OHI | `943b768424ebb2f7057e4177c46db297...` |

### Secrets hinzufügen

1. Gehe zu **Settings** > **Secrets and variables** > **Actions**
2. Klicke auf **New repository secret**
3. Füge die Secrets mit den oben genannten Namen hinzu

## Lokale Entwicklung

Für lokale E2E Tests, kopiere `.env.example` nach `.env` und fülle die Werte aus:

```bash
cp .env.example .env
```

Dann füge deine lokalen Werte ein.

## E2E Tests ausführen

### Lokal

```bash
# Alle E2E Tests
npm run test:e2e

# Nur Chromium
npx playwright test --project=chromium

# Mit UI
npx playwright test --ui

# Specific Test
npx playwright test e2e/auth.spec.ts
```

### In CI/CD

Die Tests werden automatisch bei Push/PR auf `main` und `develop` Branches ausgeführt.

## Test-Fixes

Die folgenden Fixes wurden implementiert:

1. **Strict Mode Violations**: Verwendung von `.first()` und spezifischeren Selektoren
2. **Accessibility**: `<main>` Tags zu allen Seiten hinzugefügt, `aria-label` zu Buttons
3. **Visual Regression**: `maxDiffPixels` tolerance für Animationen
4. **Responsive Design**: Erhöhte Toleranz für Mobile Viewports
5. **404 Handling**: Dev-Mode kompatible Tests

## Troubleshooting

### Visual Regression Tests schlagen fehl

Wenn Visual Regression Tests fehlschlagen, kannst du die Snapshots aktualisieren:

```bash
npx playwright test --update-snapshots
```

### Tests laufen lokal aber nicht in CI

Stelle sicher, dass alle GitHub Secrets korrekt konfiguriert sind.

### Flaky Tests

Wenn Tests intermittierend fehlschlagen:
- Erhöhe Timeouts in `playwright.config.ts`
- Füge `await page.waitForLoadState('networkidle')` hinzu
- Nutze `test.retry(2)` für spezifische flaky Tests
