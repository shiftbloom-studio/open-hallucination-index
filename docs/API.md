# Open Hallucination Index – API‑Dokumentation

> **Zielsetzung:** Diese Spezifikation beschreibt die HTTP‑Schnittstellen der OHI‑API für verifizierbare Faktenprüfung, Evidenzaggregation und Trust‑Scoring. Alle Endpunkte sind deterministisch dokumentiert und für reproduzierbare Forschungsexperimente ausgelegt.

---

## 🧪 Wissenschaftlicher Rahmen

Die API modelliert den Verifikationsprozess als Pipeline:

1. **Claim Decomposition**: Zerlegung von Text in atomare Claims.
2. **Evidence Retrieval**: Paralleles Suchen in Graph‑, Vektor‑ und MCP‑Quellen.
3. **Evidence Alignment**: Mapping der Evidenz auf Claims.
4. **Trust Scoring**: Bewertung durch evidenzbasierte Metriken.

Die Hauptmetriken sind:

- **Support Ratio** $\frac{n_{supported}}{n_{total}}$
- **Refutation Ratio** $\frac{n_{refuted}}{n_{total}}$
- **Confidence** (0–1) als Konfidenzintervall‑Schätzer
- **Overall Trust** als gewichtete Aggregation

---

## 🔐 Authentifizierung

Die API erwartet standardmäßig einen API‑Key‑Header:

```
X-API-Key: <YOUR_API_KEY>
```

Die Konfiguration erfolgt via `API_API_KEY` in der API‑Umgebung.

---

## 🌐 Basis‑URL

Standardmäßig:

```
http://localhost:8080
```

---

## ✅ Kernendpunkte

### 1) Verify (Single)

**Route**
```
POST /api/v1/verify
```

**Beschreibung**: Verifiziert einen Text und liefert Trust‑Scores, Claim‑Evidenz und Zusammenfassung.

**Request‑Schema (JSON)**

| Feld | Typ | Pflicht | Beschreibung |
|------|-----|---------|--------------|
| `text` | string | ✅ | Text zur Verifikation (max. 10.000 Zeichen) |
| `strategy` | string | ❌ | `mcp_enhanced`  `hybrid`  `cascading`  `graph_exact`  `vector_semantic` |
| `use_cache` | boolean | ❌ | Cache‑Nutzung (default: `true`) |
| `language` | string | ❌ | ISO‑Code, z. B. `de`, `en` |
| `trace` | boolean | ❌ | Zusätzliche Pipeline‑Metadaten |

**Beispiel**
```
curl -X POST http://localhost:8080/api/v1/verify \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_API_KEY" \
  -d '{"text": "Die Eiffel Tower steht in Paris und wurde 1889 gebaut."}'
```

**Beispielantwort (gekürzt)**
```
{
  "id": "abc123...",
  "trust_score": {
    "overall": 0.988,
    "claims_total": 2,
    "claims_supported": 2,
    "claims_refuted": 0,
    "confidence": 0.92
  },
  "claims": [
    {
      "claim": "Der Eiffelturm steht in Paris.",
      "verdict": "supported",
      "evidence": ["..."]
    }
  ],
  "summary": "2 Claims analysiert, 2 gestützt. Vertrauensniveau: hoch (0.99)."
}
```

---

### 2) Verify (Batch)

**Route**
```
POST /api/v1/verify/batch
```

**Beschreibung**: Parallelisierte Verifikation mehrerer Texte.

**Request‑Schema**

| Feld | Typ | Pflicht | Beschreibung |
|------|-----|---------|--------------|
| `items` | array | ✅ | Liste von Textobjekten (`text`, optional `strategy`) |
| `use_cache` | boolean | ❌ | Cache‑Nutzung |

**Hinweis**: Max. 10 Items pro Anfrage.

---

### 3) Health

| Endpoint | Zweck |
|----------|------|
| `GET /health` | Gesamte Systemgesundheit |
| `GET /health/live` | Liveness‑Probe |
| `GET /health/ready` | Readiness‑Probe |

---

## 🧠 Verifikationsstrategien

| Strategie | Charakteristik | Empfohlen für |
|-----------|----------------|--------------|
| `mcp_enhanced` | Lokale Quellen + MCP‑Quellen (z. B. Wikipedia/Context7) | Höchste Evidenzabdeckung |
| `hybrid` | Graph + Vektor parallel | Schnelle lokale Verifikation |
| `cascading` | Graph zuerst, Vektor fallback | Präzision vor Recall |
| `graph_exact` | Neo4j‑exact matching | Entity‑Konsistenz |
| `vector_semantic` | Qdrant‑Semantik | Inhaltliche Ähnlichkeit |

---

## 🧾 Fehlerformate

Fehler werden als strukturierte JSON‑Antwort geliefert:

```
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "text must not be empty",
    "details": { "field": "text" }
  }
}
```

**Typische Fehlercodes**

- `AUTH_REQUIRED`
- `AUTH_INVALID`
- `VALIDATION_ERROR`
- `RATE_LIMIT`
- `INTERNAL_ERROR`

---

## 🧰 Datenmodelle (konzeptionell)

**Claim**

- `claim`: string
- `verdict`: `supported` | `refuted` | `unknown`
- `evidence`: Evidence[]

**Evidence**

- `source`: string
- `snippet`: string
- `score`: float
- `url`: string

**TrustScore**

- `overall`: float
- `claims_total`: int
- `claims_supported`: int
- `claims_refuted`: int
- `confidence`: float

---

## 🔬 Reproduzierbarkeit

Für wissenschaftliche Reproduzierbarkeit sollten Sie:

1. Strategien und Quellen konfigurativ fixieren.
2. Versionsstände der Wissensquellen dokumentieren.
3. Den `trace`‑Modus aktivieren und archivieren.

---

## 🔗 Weitere Dokumente

- [docs/FRONTEND.md](FRONTEND.md)
- [docs/CONTRIBUTING.md](CONTRIBUTING.md)
- [docs/PUBLIC_ACCESS.md](PUBLIC_ACCESS.md)
