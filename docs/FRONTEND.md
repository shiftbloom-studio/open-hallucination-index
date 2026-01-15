# Open Hallucination Index – Frontend Documentation

> **Objective:** The frontend provides a scientifically oriented interface for interpreting verification results, evidence chains, and trust scores. The focus is on transparency, traceability, and cognitive ergonomics.

---

## 🧭 Information Architecture

The UI follows a clear hierarchy:

1. **Landing & Product Story** (Problem → Architecture → Features → CTA)
2. **Analysis Flow** (Text → Claims → Evidence → Trust Score)
3. **Result Validation** (Verified vs. Refuted Claims)
4. **Reproducibility** (Export, Sources, Knowledge Track Insights)

**Primary Goals**

- **Transparency**: Every decision is traceable to evidence.
- **Interpretability**: Scores are contextualized.
- **Scientific Rigor**: No black-box representation.

---

## 🎨 Design Principles

- **Semantic Typography**: Status labels (supported, refuted, unknown) with consistent color semantics.
- **Progressive Disclosure**: Detailed evidence only when needed.
- **Data-Dense UI**: High information density without visual overload.

---

## 🧩 Main Components (Conceptual)

| Component | Task |
|-----------|---------|
| **Landing Sections** | Hero, Problem, Architecture Flow, Feature Grid, CTA |
| **Claim List** | Aggregated display of all claims with status |
| **Evidence Panel** | Source snippets, scores, links |
| **Trust Score Card** | Overall score + confidence |
| **Knowledge Track View** | Provenance mesh & source list (API-supported) |
| **Export/Report** | CSV/JSON/Markdown Export |

---

## 🧪 Data Flows & State

**Frontend State**

- `analysisInput`: User text
- `analysisResult`: API response
- `activeClaim`: Currently selected claim
- `showTrace`: Pipeline metadata
- `knowledgeTrack`: Provenance response for claim ID

**Recommended Pattern**: Server-driven rendering with asynchronous hydration

---

## 📐 UX Metrics (Recommended)

- **Time‑to‑Insight**: Time until first results are visible
- **Evidence Depth Rate**: Percentage of explored evidence
- **Trust Score Comprehension**: User understanding via survey

---

## 🔬 Scientific Representation

**Claim Status Legend**

- **Supported**: Evidence confirms claim
- **Refuted**: Evidence contradicts claim
- **Unknown**: Insufficient evidence

**Score Interpretation**

- $0.00$ – $0.39$: Low trust
- $0.40$ – $0.69$: Moderate trust
- $0.70$ – $1.00$: High trust

---

## 🧪 Test Strategy

Recommended test pyramid:

1. **Unit Tests** (Component logic)
2. **Integration Tests** (API flows)
3. **E2E Tests** (Critical journeys)

Examples and configurations are located in the frontend folder.

## 🔌 API Proxy (Frontend)

The frontend uses a server-side proxy route:

- `GET/POST /api/ohi/*` → forwards to `DEFAULT_API_URL`
- Header `X-API-KEY` is automatically set with `DEFAULT_API_KEY`
- Optionally `X-User-Id` is added from Supabase

This allows UI requests to occur without direct API key disclosure to the client.

## ⚙️ Relevant Environment Variables

- `DEFAULT_API_URL` (Backend base URL)
- `DEFAULT_API_KEY` (Server-side API key)
- `NEXT_PUBLIC_SUPABASE_URL`
- `NEXT_PUBLIC_SUPABASE_ANON_KEY`
- `NEXT_PUBLIC_APP_URL`

---

## 🔗 Linked Documents

- [docs/API.md](API.md)
- [docs/CONTRIBUTING.md](CONTRIBUTING.md)
- [docs/PUBLIC_ACCESS.md](PUBLIC_ACCESS.md)
