<p align="center">
  <img src="https://raw.githubusercontent.com/your-org/open-hallucination-index/main/docs/assets/logo.png" alt="Open Hallucination Index" width="120" />
</p>

<h1 align="center">Open Hallucination Index</h1>

<p align="center">
  <strong>🔍 Real-time fact-checking for LLM outputs</strong>
</p>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#api-reference">API</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#configuration">Config</a> •
  <a href="#contributing">Contributing</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11+-blue.svg" alt="Python 3.11+" />
  <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="MIT License" />
  <img src="https://img.shields.io/badge/docker-ready-2496ED.svg" alt="Docker Ready" />
  <a href="https://github.com/your-org/open-hallucination-index/actions"><img src="https://github.com/your-org/open-hallucination-index/workflows/CI/badge.svg" alt="CI Status" /></a>
</p>

---

**Open Hallucination Index (OHI)** is a high-performance middleware API that verifies LLM-generated text against trusted knowledge sources. It detects hallucinations by decomposing text into atomic claims and validating each against multiple knowledge bases in real-time.

## ✨ Features

| Feature | Description |
|---------|-------------|
| **🧠 Claim Decomposition** | Breaks text into verifiable atomic claims using LLM-powered extraction |
| **📊 Multi-Source Verification** | Validates against Neo4j graph, Qdrant vectors, Wikipedia, and Context7 |
| **⚡ High Performance** | Session pooling, batch processing, parallel verification, Redis caching |
| **🎯 Trust Scoring** | Evidence-ratio based scoring with confidence intervals (0.0 - 1.0) |
| **🔌 Pluggable Architecture** | Hexagonal design - easily swap knowledge sources and strategies |
| **🐳 Docker Ready** | One-command deployment with docker-compose |

## 🚀 Quick Start

### Prerequisites

- Docker & Docker Compose
- 16GB+ RAM recommended (for local LLM)
- NVIDIA GPU (optional, for faster inference)

### 1. Clone & Configure

```bash
git clone https://github.com/your-org/open-hallucination-index.git
cd open-hallucination-index

# Copy environment template
cp .env.example .env

# Edit with your settings (optional - defaults work for local development)
nano .env
```

### 2. Start with Docker Compose

```bash
# Build and start all services
docker compose up -d

# Check status
docker compose ps

# View logs
docker compose logs -f ohi-api
```

This starts:
- **ohi-api** (port 8080) - Main verification API
- **ohi-vllm** (port 8000) - Local LLM inference (vLLM)
- **ohi-neo4j** (port 7474/7687) - Graph database
- **ohi-qdrant** (port 6333) - Vector database
- **ohi-redis** (port 6379) - Caching layer
- **ohi-wikipedia-mcp** (port 3001) - Wikipedia knowledge source
- **ohi-context7-mcp** (port 3002) - Technical documentation source

### 3. Verify Installation

```bash
# Health check
curl http://localhost:8080/health

# Test verification (replace YOUR_API_KEY from .env)
curl -X POST http://localhost:8080/api/v1/verify \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_API_KEY" \
  -d '{"text": "The Eiffel Tower is located in Paris and was built in 1889."}'
```

**Expected Response:**
```json
{
  "id": "abc123...",
  "trust_score": {
    "overall": 0.988,
    "claims_total": 2,
    "claims_supported": 2,
    "claims_refuted": 0,
    "confidence": 0.92
  },
  "summary": "Analyzed 2 claim(s): 2 supported. Overall trust level: high (0.99).",
  "claims": [
    {
      "text": "The Eiffel Tower is located in Paris",
      "status": "supported",
      "confidence": 0.92,
      "reasoning": "Strongly supported: 12 supporting vs 1 contradicting (ratio 12.0:1)."
    }
  ]
}
```

## 📖 API Reference

### Verify Text

```http
POST /api/v1/verify
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `text` | string | ✅ | Text to verify (max 10,000 chars) |
| `strategy` | string | ❌ | `hybrid`, `mcp_enhanced`, `graph_exact`, `vector_semantic`, `cascading` |
| `use_cache` | boolean | ❌ | Use cached results (default: `true`) |

### Batch Verification

```http
POST /api/v1/verify/batch
```

Verify multiple texts in parallel. Max 10 texts per request.

```json
{
  "texts": ["Text 1 to verify", "Text 2 to verify"],
  "strategy": "mcp_enhanced"
}
```

### List Strategies

```http
GET /api/v1/strategies
```

Returns available verification strategies with descriptions.

### Health Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Full health status with component checks |
| `GET /health/live` | Kubernetes liveness probe |
| `GET /health/ready` | Kubernetes readiness probe |

## 🏗️ Architecture

OHI follows **Hexagonal Architecture** (Ports and Adapters) for maximum flexibility:

```
┌─────────────────────────────────────────────────────────────────────┐
│                          API Layer (FastAPI)                         │
│                    POST /verify, GET /health                         │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Application Layer                                 │
│              VerifyTextUseCase (Orchestration)                       │
└──────┬──────────────────┬───────────────────┬───────────────────────┘
       │                  │                   │
       ▼                  ▼                   ▼
┌─────────────┐   ┌─────────────────┐   ┌───────────────┐
│   Domain    │   │  Domain Services │   │    Ports      │
│  Entities   │   │  - Scorer        │   │  (Interfaces) │
│  - Claim    │   │  - Oracle        │   │  - LLMProvider│
│  - Evidence │   │  - Decomposer    │   │  - KnowledgeStore
│  - TrustScore│  └─────────────────┘   └───────────────┘
└─────────────┘                                 │
                                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         Adapters (Outbound)                          │
│  ┌───────────┐ ┌──────────┐ ┌───────────┐ ┌───────────┐ ┌─────────┐│
│  │ OpenAI/   │ │  Neo4j   │ │  Qdrant   │ │  Redis    │ │  MCP    ││
│  │ vLLM      │ │  Graph   │ │  Vector   │ │  Cache    │ │Wikipedia││
│  └───────────┘ └──────────┘ └───────────┘ └───────────┘ └─────────┘│
└─────────────────────────────────────────────────────────────────────┘
```

### Verification Strategies

| Strategy | Description | Use Case |
|----------|-------------|----------|
| `mcp_enhanced` | Query external sources (Wikipedia, Context7) + local stores | **Recommended** - Most comprehensive |
| `hybrid` | Parallel graph + vector search | Fast local-only verification |
| `cascading` | Graph first, vector fallback | When exact matches preferred |
| `graph_exact` | Neo4j only | Known entity verification |
| `vector_semantic` | Qdrant only | Semantic similarity matching |

## ⚙️ Configuration

### Environment Variables

```env
# API Settings
API_HOST=0.0.0.0
API_PORT=8080
API_API_KEY=your-secret-api-key    # Required for production

# LLM Configuration
LLM_BASE_URL=http://ohi-vllm:8000/v1
LLM_MODEL=Qwen/Qwen2.5-7B-Instruct
LLM_API_KEY=no-key-required        # For local vLLM

# Neo4j Graph Database
NEO4J_URI=bolt://ohi-neo4j:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your-neo4j-password

# Qdrant Vector Database
QDRANT_HOST=ohi-qdrant
QDRANT_PORT=6333

# Redis Cache
REDIS_HOST=ohi-redis
REDIS_PORT=6379
REDIS_ENABLED=true

# MCP Sources
MCP_WIKIPEDIA_ENABLED=true
MCP_WIKIPEDIA_URL=http://ohi-wikipedia-mcp:3001
MCP_CONTEXT7_ENABLED=true
MCP_CONTEXT7_URL=http://ohi-context7-mcp:3002

# Verification
VERIFICATION_DEFAULT_STRATEGY=mcp_enhanced
VERIFICATION_PERSIST_MCP_EVIDENCE=true
```

### Docker Compose Profiles

```bash
# Full stack (default)
docker compose up -d

# API only (connect to external services)
docker compose -f docker-compose.api.yml up -d

# With GPU acceleration
docker compose --profile gpu up -d
```

## 🧪 Development

### Local Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Type checking
mypy src

# Linting & formatting
ruff check src tests
ruff format src tests
```

### Project Structure

```
open-hallucination-index/
├── src/open_hallucination_index/
│   ├── domain/           # Core entities (Claim, Evidence, TrustScore)
│   ├── ports/            # Abstract interfaces
│   ├── application/      # Use-case orchestration
│   ├── adapters/         # External service implementations
│   │   └── outbound/     # Neo4j, Qdrant, Redis, MCP adapters
│   ├── infrastructure/   # Config, DI, lifecycle
│   └── api/              # FastAPI routes
├── tests/                # Unit & integration tests
├── docker/               # Docker configurations
├── scripts/              # Utility scripts
└── docs/                 # Documentation
```

## 🤝 Contributing

Contributions are welcome! Please read our [Contributing Guide](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [vLLM](https://github.com/vllm-project/vllm) - High-throughput LLM serving
- [Neo4j](https://neo4j.com/) - Graph database
- [Qdrant](https://qdrant.tech/) - Vector search engine
- [FastAPI](https://fastapi.tiangolo.com/) - Modern Python web framework
- [MCP](https://modelcontextprotocol.io/) - Model Context Protocol

---

<p align="center">
  Made with ❤️ by the OHI Team
</p>
