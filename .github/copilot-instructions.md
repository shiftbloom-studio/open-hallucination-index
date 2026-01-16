# Open Hallucination Index - Copilot Instructions

You are an expert developer working on the **Open Hallucination Index (OHI)**. OHI is a high-performance middleware platform for detecting LLM hallucinations by decomposing claims and verifying them against trusted knowledge sources (Neo4j, Qdrant, MCP).

---

## 🧭 Monorepo map (must respect)

- `src/api/` – Python FastAPI backend (Hexagonal Architecture)
- `src/frontend/` – Next.js 16 App Router UI
- `src/ingestion/` – High‑performance Wikipedia ingestion pipeline
- `src/benchmark/` – Research‑grade evaluation suite
- `src/ohi-mcp-server/` – MCP server (Node.js)
- `docker/compose/docker-compose.yml` – Full stack orchestration
- `docker/README.md` – Docker stack documentation

---

## 🧠 Core verification pipeline (API)

Orchestrated by `VerifyTextUseCase`:

1. Cache lookup (Redis; input hash + claim hash)
2. Claim decomposition (LLM; fallback to single-claim)
3. Claim routing (ClaimRouter → domain + source priorities)
4. Evidence collection (AdaptiveEvidenceCollector; local tier → MCP tier)
5. Verification decision (HybridVerificationOracle)
6. Trust scoring (WeightedScorer)
7. Cache + trace persistence (Redis; 12h TTL for knowledge-track)

Available strategies (`VerificationStrategy`):

- `graph_exact`, `vector_semantic`, `hybrid`, `cascading`, `mcp_enhanced`, `adaptive` (default)

---

## ✅ Hexagonal architecture rules (API)

STRICT separation under `src/api/src/open_hallucination_index/`:

- `domain/`: pure business logic only
- `ports/`: abstract interfaces (ABCs)
- `adapters/`: concrete integrations (Neo4j, Qdrant, Redis, MCP, LLM)
- `application/`: use‑case orchestration
- `infrastructure/`: configuration + DI wiring

Do not import external infrastructure inside `domain/`.

---

## 🔌 Knowledge sources

Local:
- Neo4j graph store
- Qdrant vector store (384‑dim, all‑MiniLM‑L12‑v2)

External (via `ohi-mcp-server`):
- Wikipedia/Wikidata/DBpedia
- OpenAlex, Crossref, EuropePMC
- PubMed, ClinicalTrials.gov
- GDELT, World Bank, OSV
- Context7 (documentation)

---

## ⚛️ Frontend guidelines

- Next.js 16 App Router
- Prefer Server Components; use `"use client"` only for interactivity
- React Query for server state; Server Actions where appropriate
- UI pattern: shadcn/ui + Tailwind v4

---

## 🐳 Docker stack essentials

Services (ports as exposed in compose):

- Neo4j (7474/7687)
- Qdrant (6333/6334)
- Redis (6379)
- vLLM (8000)
- MCP server (8083 → 8080 in-container)
- API (8080 internal)
- Frontend (3000 internal)
- Nginx (80/443)
- Cloudflared tunnel (optional)

LLM model in compose: **cyankiwi/Hermes-4-14B-AWQ-4bit** (GPU).

---

## 🔧 Development workflows

API:
- venv: `src/api/.venv`
- install: `pip install -e "src/api[dev]"`
- run: `ohi-server`

Frontend:
- `npm install`, `npm run dev`

Ingestion:
- `python -m ingestion --limit 1000`

Benchmark:
- `python -m benchmark` or `docker exec ohi-benchmark python -m benchmark`

---

## ✅ Coding standards

- Python 3.14+ features allowed
- Strict typing; use `typing.Annotated` for FastAPI DI
- Async for I/O‑bound operations
- Respect existing linting (ruff, mypy)

---

## 🧾 Documentation priorities

Whenever editing or adding functionality, keep documentation aligned:

- Main overview: `README.md`
- API detail: `src/api/README.md`
- MCP detail: `src/ohi-mcp-server/README.md`
- Benchmark: `src/benchmark/README.md`
- Ingestion: `src/ingestion/README.md`
- Frontend: `src/frontend/README.md`
- Docker stack: `docker/README.md`

---

## ⚠️ Critical project context

- LLM is local (vLLM); do not assume OpenAI API availability
- MCP evidence can be persisted to Neo4j + Qdrant
- Redis enables caching and knowledge‑track endpoints
- API key auth is optional (disabled when API_API_KEY is empty)

---

## ✅ Expected behavior for Copilot

- Preserve hexagonal boundaries
- Keep changes minimal and focused
- Update docs when behavior changes
- Avoid breaking public API endpoints