# Open Hallucination Index – Frontend‑Dokumentation

> **Zielsetzung:** Das Frontend bietet eine wissenschaftlich orientierte Oberfläche zur Interpretation von Verifikationsergebnissen, Evidenzketten und Trust‑Scores. Der Schwerpunkt liegt auf Transparenz, Nachvollziehbarkeit und kognitiver Ergonomie.

---

## 🧭 Informationsarchitektur

Die UI folgt einer klaren Hierarchie:

1. **Landing & Produktstory** (Problem → Architektur → Features → CTA)
2. **Analyse‑Fluss** (Text → Claims → Evidenz → Trust‑Score)
3. **Ergebnis‑Validierung** (verifizierte vs. widerlegte Claims)
4. **Reproduzierbarkeit** (Export, Quellen, Knowledge‑Track‑Einblicke)

**Primäre Ziele**

- **Transparenz**: Jede Entscheidung ist auf Evidenz rückführbar.
- **Interpretierbarkeit**: Scores werden kontextualisiert.
- **Wissenschaftliche Strenge**: Keine Black‑Box‑Darstellung.

---

## 🎨 Designprinzipien

- **Semantische Typografie**: Statuslabels (supported, refuted, unknown) mit konsistenter Farbsemantik.
- **Progressive Disclosure**: Tiefe Evidenz nur bei Bedarf.
- **Daten‑Dense UI**: Hohe Informationsdichte ohne visuelle Überladung.

---

## 🧩 Hauptkomponenten (konzeptionell)

| Komponente | Aufgabe |
|-----------|---------|
| **Landing Sections** | Hero, Problem, Architekturfluss, Feature‑Grid, CTA |
| **Claim List** | Aggregierte Anzeige aller Claims mit Status |
| **Evidence Panel** | Quellen‑Snippets, Scores, Links |
| **Trust Score Card** | Gesamt‑Score + Confidence |
| **Knowledge Track View** | Provenienz‑Mesh & Quellenliste (API‑gestützt) |
| **Export/Report** | CSV/JSON/Markdown Export |

---

## 🧪 Datenflüsse & State

**Frontend‑State**

- `analysisInput`: Nutzertext
- `analysisResult`: API‑Response
- `activeClaim`: aktuell selektierter Claim
- `showTrace`: Pipeline‑Metadaten
- `knowledgeTrack`: Provenienz‑Response zu Claim‑ID

**Empfohlenes Muster**: Server‑driven Rendering mit asynchroner Hydration

---

## 📐 UX‑Metriken (empfohlen)

- **Time‑to‑Insight**: Zeit bis erste Ergebnisse sichtbar sind
- **Evidence Depth Rate**: Anteil explorierter Evidenzen
- **Trust Score Comprehension**: Nutzerverständnis via Befragung

---

## 🔬 Wissenschaftliche Darstellung

**Claim‑Statuslegende**

- **Supported**: Evidenz bestätigt Claim
- **Refuted**: Evidenz widerspricht Claim
- **Unknown**: keine ausreichende Evidenz

**Score‑Interpretation**

- $0.00$ – $0.39$: niedriges Vertrauen
- $0.40$ – $0.69$: moderates Vertrauen
- $0.70$ – $1.00$: hohes Vertrauen

---

## 🧪 Teststrategie

Empfohlene Testpyramide:

1. **Unit Tests** (Komponentenlogik)
2. **Integration Tests** (API‑Flows)
3. **E2E Tests** (Kritische Journeys)

Beispiele und Konfigurationen befinden sich im Frontend‑Ordner.

## 🔌 API‑Proxy (Frontend)

Das Frontend nutzt eine serverseitige Proxy‑Route:

- `GET/POST /api/ohi/*` → leitet an `DEFAULT_API_URL` weiter
- Header `X-API-KEY` wird automatisch mit `DEFAULT_API_KEY` gesetzt
- Optional wird `X-User-Id` aus Supabase ergänzt

Damit können UI‑Requests ohne direkte API‑Key‑Weitergabe an den Client erfolgen.

## ⚙️ Relevante Umgebungsvariablen

- `DEFAULT_API_URL` (Backend‑Base‑URL)
- `DEFAULT_API_KEY` (Server‑seitiger API‑Key)
- `NEXT_PUBLIC_SUPABASE_URL`
- `NEXT_PUBLIC_SUPABASE_ANON_KEY`
- `NEXT_PUBLIC_APP_URL`

---

## 🔗 Verknüpfte Dokumente

- [docs/API.md](API.md)
- [docs/CONTRIBUTING.md](CONTRIBUTING.md)
- [docs/PUBLIC_ACCESS.md](PUBLIC_ACCESS.md)
