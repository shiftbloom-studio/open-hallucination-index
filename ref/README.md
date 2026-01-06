# Open Hallucination Index & Verification API

A middleware API for verifying LLM-generated text against trusted knowledge sources. Detects hallucinations by decomposing text into atomic claims and validating each against graph and vector knowledge stores.

## Architecture

This project follows **Hexagonal Architecture** (Ports and Adapters) for clean separation of concerns:

```
src/open_hallucination_index/
├── domain/           # Core entities and value objects (Claim, Evidence, TrustScore)
├── ports/            # Abstract interfaces (ClaimDecomposer, KnowledgeStore, LLMProvider)
├── application/      # Use-case orchestration (VerifyTextUseCase)
├── adapters/         # Concrete implementations for external services
│   └── outbound/     # Neo4j, Qdrant, Redis, OpenAI adapters
├── infrastructure/   # Config, DI, entrypoint
└── api/              # FastAPI routes and schemas
```

### Core Concepts

1. **Claim Decomposition**: Breaks unstructured text into atomic, verifiable claims (subject-predicate-object triplets)
2. **Verification Oracle**: Validates claims using pluggable strategies:
   - `GRAPH_EXACT`: Exact matching in Neo4j knowledge graph
   - `VECTOR_SEMANTIC`: Semantic similarity in Qdrant vector store
   - `HYBRID`: Combined graph + vector verification
   - `CASCADING`: Graph-first with vector fallback
3. **Trust Scoring**: Aggregates individual claim scores into a global trust score (0.0 - 1.0)

## Quick Start

### Prerequisites

- Python 3.11+
- Docker services running:
  - vLLM/OpenAI-compatible API on port 8000
  - Neo4j on ports 7474 (HTTP) / 7687 (Bolt)
  - Qdrant on port 6333
  - Redis on port 6379

### Installation

```bash
# Install in development mode
pip install -e ".[dev]"

# Or just install dependencies
pip install -e .
```

### Configuration

Create a `.env` file or set environment variables:

```env
# LLM
LLM_BASE_URL=http://localhost:8000/v1
LLM_MODEL=default

# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password

# Qdrant
QDRANT_HOST=localhost
QDRANT_PORT=6333

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379

# API
API_HOST=0.0.0.0
API_PORT=8080
```

### Running the API

```bash
# Using the CLI entrypoint
ohi-server

# Or directly with uvicorn
uvicorn open_hallucination_index.api.app:create_app --factory --reload
```

### API Endpoints

- `POST /api/v1/verify` - Verify text for factual accuracy
- `POST /api/v1/verify/batch` - Batch verification
- `GET /api/v1/strategies` - List available verification strategies
- `GET /health/live` - Liveness probe
- `GET /health/ready` - Readiness probe

## Project Status

This is the **scaffolding phase**. The architecture and interfaces are defined, but adapters contain stub implementations. Next steps:

1. [ ] Implement `OpenAILLMAdapter` with actual API calls
2. [ ] Implement `Neo4jGraphAdapter` with Cypher queries
3. [ ] Implement `QdrantVectorAdapter` with vector search
4. [ ] Implement `RedisCacheAdapter` with caching logic
5. [ ] Implement `ClaimDecomposer` (LLM-based extraction)
6. [ ] Implement `VerificationOracle` (strategy patterns)
7. [ ] Implement `Scorer` (weighted aggregation)
8. [ ] Add unit and integration tests
9. [ ] Wire adapters in `infrastructure/dependencies.py`

## Development

```bash
# Run tests
pytest

# Type checking
mypy src

# Linting
ruff check src tests

# Format
ruff format src tests
```

## License

MIT
