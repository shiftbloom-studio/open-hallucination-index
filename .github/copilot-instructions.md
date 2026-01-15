# Open Hallucination Index - AI Agent Instructions

You are an expert developer working on the **Open Hallucination Index (OHI)**.
OHI is a high-performance middleware for detecting LLM hallucinations by decomposing claims and verifying them against trusted knowledge sources (Neo4j, Qdrant, MCP).

## 🏗️ Architecture & Structure

- **Monorepo Structure**:
  - `api/`: Python FastAPI backend (Hexagonal Architecture).
  - `frontend/`: Next.js 16+ App Router web application.
  - `ingestion/`: High-performance Python pipeline for Wikipedia ingestion.
  - `benchmark/`: Research-grade evaluation suite.
  - `docker-compose.yml`: Context for all running services (Neo4j, Qdrant, Redis, vLLM, MCP).

### Service Map
| Service | Technology | Role | Port |
|---------|------------|------|------|
| **API** | Python/FastAPI | Core orchestration & verification | 8080 |
| **Frontend** | Next.js/React | User Interface | 3000 |
| **Knowledge Graph** | Neo4j | Exact relationship matching | 7474/7687 |
| **Vector DB** | Qdrant | Semantic search & similarity | 6333 |
| **Cache** | Redis | Session state & caching | 6379 |
| **LLM** | vLLM (Qwen2.5) | Local inference (Decomposition/Scoring) | 8000 |
| **MCP Server** | Node.js | Unified external knowledge sources | 8083 |

---

## 🔄 Verification Pipeline Orchestration

The core verification flow is orchestrated by `VerifyTextUseCase` in `application/verify_text.py`:

```
Input Text
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  1. CACHE CHECK (Redis)                                         │
│     - Hash input text → check for cached VerificationResult     │
│     - Also supports claim-level caching (by claim hash)         │
└─────────────────────────────────────────────────────────────────┘
    │ (cache miss)
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  2. CLAIM DECOMPOSITION (LLM via ClaimDecomposer port)          │
│     - Text → List[Claim] with structured triplets               │
│     - Each Claim has: subject, predicate, object, claim_type    │
│     - Claim types: TEMPORAL, QUANTITATIVE, CAUSAL, etc.         │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  3. CLAIM ROUTING (ClaimRouter in domain/services/)             │
│     - Classifies claim domain: MEDICAL, ACADEMIC, NEWS, etc.    │
│     - Returns prioritized SourceRecommendation list             │
│     - Each source has: tier (LOCAL_FAST, MCP_MEDIUM, MCP_SLOW)  │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  4. EVIDENCE COLLECTION (AdaptiveEvidenceCollector)             │
│     TIERED EXECUTION with early-exit:                           │
│     ┌─────────────────────────────────────────────────────────┐ │
│     │ Tier 1: LOCAL (Neo4j + Qdrant) - <20ms                  │ │
│     │   - GraphKnowledgeStore: exact triplet matching         │ │
│     │   - VectorKnowledgeStore: semantic similarity search    │ │
│     └─────────────────────────────────────────────────────────┘ │
│                    │ (insufficient evidence?)                    │
│                    ▼                                             │
│     ┌─────────────────────────────────────────────────────────┐ │
│     │ Tier 2: MCP SOURCES (via SmartMCPSelector) - 50-500ms   │ │
│     │   - Wikipedia, Wikidata, DBpedia                        │ │
│     │   - OpenAlex, Crossref, PubMed (academic)               │ │
│     │   - GDELT (news), World Bank (economic), OSV (security) │ │
│     │   - Context7 (technical documentation)                  │ │
│     └─────────────────────────────────────────────────────────┘ │
│                                                                  │
│     EARLY EXIT: Stops when sufficient weighted evidence found   │
│     Quality weighting: source_reliability × similarity_score    │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  5. VERIFICATION DECISION (HybridVerificationOracle)            │
│     - Aggregates evidence from all sources                      │
│     - Determines: SUPPORTED, REFUTED, UNVERIFIABLE, PARTIAL     │
│     - Generates CitationTrace with source provenance            │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  6. SCORING (WeightedScorer)                                    │
│     - Computes TrustScore (0.0-1.0) from claim verifications    │
│     - Evidence-ratio based with confidence intervals            │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
VerificationResult (cached + returned)
```

### Verification Strategies (`VerificationStrategy` enum)
| Strategy | Behavior |
|----------|----------|
| `GRAPH_EXACT` | Only Neo4j exact matching |
| `VECTOR_SEMANTIC` | Only Qdrant semantic search |
| `HYBRID` | Graph + Vector in parallel |
| `CASCADING` | Graph first, vector fallback |
| `MCP_ENHANCED` | MCP sources first, local fallback |
| `ADAPTIVE` | **Default**: Tiered collection with early-exit |

---

## 🧠 Backend / Python (`api/`, `ingestion/`, `benchmark/`)

### Architecture Pattern: Hexagonal (Ports & Adapters)
STRICTLY follow this separation in `api/src/open_hallucination_index/`:

```
api/src/open_hallucination_index/
├── domain/           # Pure business logic (NO external imports)
│   ├── entities.py   # Claim, Evidence, EvidenceSource
│   ├── results.py    # VerificationResult, TrustScore, CitationTrace
│   └── services/     # Domain services (ClaimRouter, EvidenceCollector, etc.)
├── ports/            # Abstract interfaces (ABCs)
│   ├── claim_decomposer.py   # Text → Claims
│   ├── knowledge_store.py    # GraphKnowledgeStore, VectorKnowledgeStore
│   ├── mcp_source.py         # MCPKnowledgeSource interface
│   ├── verification_oracle.py # Claim verification orchestration
│   ├── scorer.py             # Trust score computation
│   └── cache.py              # Result caching
├── adapters/outbound/  # Concrete implementations
│   ├── graph_neo4j.py        # Neo4jGraphAdapter
│   ├── vector_qdrant.py      # QdrantVectorAdapter
│   ├── llm_openai.py         # OpenAI-compatible LLM (vLLM)
│   ├── mcp_ohi.py            # OHI MCP Server adapter
│   ├── cache_redis.py        # Redis cache
│   └── embeddings_local.py   # Local sentence-transformers
├── application/        # Use-case orchestration
│   └── verify_text.py        # VerifyTextUseCase (main pipeline)
└── infrastructure/     # Wiring & configuration
    ├── dependencies.py       # DI container (FastAPI lifespan)
    └── config.py             # Pydantic Settings
```

### Key Domain Services
| Service | Location | Responsibility |
|---------|----------|----------------|
| `ClaimRouter` | `domain/services/claim_router.py` | Classify claims by domain, recommend sources |
| `SmartMCPSelector` | `domain/services/mcp_selector.py` | Select relevant MCP sources per claim |
| `AdaptiveEvidenceCollector` | `domain/services/evidence_collector.py` | Tiered evidence collection with early-exit |
| `HybridVerificationOracle` | `domain/services/verification_oracle.py` | Apply verification strategies |
| `LLMClaimDecomposer` | `domain/services/claim_decomposer.py` | LLM-powered claim extraction |

### Coding Standards
- **Python Version**: 3.14+ features allowed.
- **Typing**: STRICT type hints required. Use `typing.Annotated` for dependency injection.
- **Async**: Use `async/await` for all I/O bound operations (DB, LLM calls).
- **Libraries**:
  - `fastapi` for API.
  - `pydantic` v2 for data models.
  - `neo4j`, `qdrant-client`, `redis` for DBs.
  - `httpx` for async HTTP.

### Testing
- Use `pytest` with `pytest-asyncio`.
- **API Tests**: `pytest api/tests/`
- **Benchmark Tests**: `pytest benchmark/`

---

## 🔌 Knowledge Sources Integration

### Local Sources (Tier 1 - Fast)
| Source | Adapter | Data |
|--------|---------|------|
| **Neo4j** | `Neo4jGraphAdapter` | Wikipedia entities + relationships (10+ edge types) |
| **Qdrant** | `QdrantVectorAdapter` | Dense + sparse (BM25) vectors, 384-dim (all-MiniLM-L12-v2) |

### MCP Sources (Tier 2 - External)
All accessed via unified `OHIMCPAdapter` → `ohi-mcp-server` (port 8083):

| Source | Domain | MCP Tool |
|--------|--------|----------|
| Wikipedia/Wikidata/DBpedia | General knowledge | `ohi_wikipedia`, `ohi_wikidata` |
| OpenAlex, Crossref | Academic papers | `ohi_openalex`, `ohi_crossref` |
| PubMed, EuropePMC | Medical/biomedical | `ohi_pubmed`, `ohi_europepmc` |
| ClinicalTrials.gov | Clinical research | `ohi_clinicaltrials` |
| GDELT | News & events | `ohi_gdelt` |
| World Bank | Economic data | `ohi_worldbank` |
| OSV | Security vulnerabilities | `ohi_osv` |
| Context7 | Technical documentation | `context7` |

### Evidence Quality Weights (in `EvidenceQuality.assess()`)
```python
source_weights = {
    EvidenceSource.GRAPH_EXACT: 1.0,      # Highest trust
    EvidenceSource.PUBMED: 0.95,
    EvidenceSource.GRAPH_INFERRED: 0.9,
    EvidenceSource.ACADEMIC: 0.9,
    EvidenceSource.VECTOR_SEMANTIC: 0.85,
    EvidenceSource.WIKIPEDIA: 0.8,
    EvidenceSource.NEWS: 0.65,            # Lower trust
}
```

---

## ⚛️ Frontend / React (`frontend/`)

### Tech Stack
- **Framework**: Next.js 16 (App Router).
- **Styling**: Tailwind CSS v4.
- **State**: React Query (server state), URL search params (navigation state).
- **DB/Auth**: Supabase + Drizzle ORM.
- **Tests**: Vitest (Unit), Playwright (E2E).

### Guidelines
- **Server Components**: Default to Server Components. Use `"use client"` only for interactivity.
- **Data Fetching**: Use Server Actions or React Query.
- **UI Components**: Use `shadcn/ui` patterns (Radix UI + Tailwind).
- **Type Safety**: Generate Typescript types from DB schema (`npm run db:generate`).

---

## 🛠️ Workflows & Commands

### Setup & run
- **Full Stack**: `docker compose up -d`
- **Rebuild API**: `docker compose build ohi-api && docker compose up -d ohi-api`

### API Development (`api/`)
```bash
cd api
python -m venv venv && source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -e ".[dev]"                          # Install dev deps
pytest                                           # Run tests
uvicorn src.open_hallucination_index.api.main:app --reload  # Dev server
```

### Frontend Development (`frontend/`)
```bash
cd frontend
npm install
npm run dev      # Start dev server on :3000
npm run test     # Run Vitest
npm run db:push  # Push schema changes to Supabase/Postgres
```

### Ingestion (`ingestion/`)
```bash
python -m ingestion --limit 1000  # Run Wikipedia ingestion pipeline
```

### Benchmark (`benchmark/`)
```bash
pip install -e benchmark/
python -m benchmark                              # Run full suite
docker exec ohi-benchmark python -m benchmark   # Run inside container
```

---

## ⚠️ Critical Project Context

- **LLM is LOCAL**: vLLM runs on GPU (Baichuan2-13B-Chat-4bits). Do not assume OpenAI API.
- **MCP Protocol**: The `ohi-mcp-server` aggregates 12+ external sources. Use `MCPKnowledgeSource` port.
- **Embeddings are LOCAL**: `all-MiniLM-L12-v2` via `sentence-transformers` (384 dimensions).
- **Dual Storage**: MCP evidence can be persisted to both Neo4j (graph) and Qdrant (vector) for future local retrieval.
- **Dependencies**: `pyproject.toml` (API/Benchmark), `package.json` (Frontend/MCP Server).
