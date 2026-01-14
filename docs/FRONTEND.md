# Open Hallucination Index – Frontend‑Dokumentation

> **Zielsetzung:** Das Frontend bietet eine wissenschaftlich orientierte Oberfläche zur Interpretation von Verifikationsergebnissen, Evidenzketten und Trust‑Scores. Der Schwerpunkt liegt auf Transparenz, Nachvollziehbarkeit und kognitiver Ergonomie.

---

## 🧭 Informationsarchitektur

Die UI folgt einer klaren Hierarchie:

1. **Eingabe‑/Analyse‑Fluss** (Text → Claims → Evidenz → Trust‑Score)
2. **Ergebnis‑Validierung** (verifizierte vs. widerlegte Claims)
3. **Reproduzierbarkeit** (Export, Trace‑Konfiguration, Quellen)

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
| **Claim List** | Aggregierte Anzeige aller Claims mit Verdicts |
| **Evidence Panel** | Quellen‑Snippets, Scores, Links |
| **Trust Score Card** | Gesamt‑Score + Confidence |
| **Trace View** | Pipeline‑Details und Strategien |
| **Export/Report** | CSV/JSON/Markdown Export |

---

## 🧪 Datenflüsse & State

**Frontend‑State**

- `analysisInput`: Nutzertext
- `analysisResult`: API‑Response
- `activeClaim`: aktuell selektierter Claim
- `showTrace`: Pipeline‑Metadaten

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

---

## 🔗 Verknüpfte Dokumente

- [docs/API.md](API.md)
- [docs/CONTRIBUTING.md](CONTRIBUTING.md)
- [docs/PUBLIC_ACCESS.md](PUBLIC_ACCESS.md)
