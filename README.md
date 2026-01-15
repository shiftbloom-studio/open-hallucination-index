<p align="center">
  <img src="https://raw.githubusercontent.com/shiftbloom-studio/open-hallucination-index/main/docs/logo.jpg" alt="Open Hallucination Index" width="765" />
</p>

<h1 align="center">Open Hallucination Index</h1>

<p align="center">
  <strong>🔍 Wissenschaftlich fundiertes Fact-Checking für LLM-Ausgaben in Echtzeit</strong>
</p>

<p align="center">
  <a href="#project-overview">Projektüberblick</a> •
  <a href="#documentation">Dokumentation</a> •
  <a href="#project-structure">Struktur</a> •
  <a href="#getting-started">Start</a> •
  <a href="#contributing">Mitwirkung</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.14+-blue.svg" alt="Python 3.14+" />
  <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="MIT License" />
  <a href="https://github.com/shiftbloom-studio/open-hallucination-index/actions"><img src="https://github.com/shiftbloom-studio/open-hallucination-index/workflows/CI/badge.svg" alt="CI Status" /></a>
</p>

---

**Open Hallucination Index (OHI)** ist eine hochperformante Middleware- und Analyseplattform, die LLM‑Ausgaben in atomic Claims zerlegt, diese gegen kuratierte Wissensquellen verifiziert und eine nachvollziehbare Vertrauensbewertung in Echtzeit berechnet. Der Fokus liegt auf reproduzierbarer, evidenzbasierter Halluzinationsdetektion mit klaren Schnittstellen für Forschung, Produktivbetrieb und Auditierbarkeit.

## 🧭 Projektüberblick

OHI verbindet **Claim‑Decomposition**, **Multi‑Source‑Evidenzsuche** und **quantitative Trust‑Scoring‑Modelle**. Die Architektur folgt einem hexagonalen Design, sodass Wissensquellen, Scoring‑Strategien und Retrieval‑Pipelines austauschbar bleiben. Das System besteht aus:

- **API (FastAPI):** Orchestriert Verifikation, Evidence‑Aggregation und Scoring.
- **Frontend (Next.js):** Wissenschaftlich orientierte UI für Analyse, Nachvollziehbarkeit und Reporting.
- **Infrastruktur‑Layer:** Neo4j, Qdrant und optionale MCP‑Quellen für externe Evidenz.

## 📚 Dokumentation

Die detaillierte Dokumentation ist im Ordner docs abgelegt:

- [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md) – Beitragspraxis, Konventionen und Review‑Prozess
- [docs/CODE_OF_CONDUCT.md](docs/CODE_OF_CONDUCT.md) – Community‑Standards
- [docs/PUBLIC_ACCESS.md](docs/PUBLIC_ACCESS.md) – Öffentlicher Zugriff und Nutzungsrahmen
- [docs/API.md](docs/API.md) – Vollständige API‑Spezifikation, Modelle, Beispiele
- [docs/FRONTEND.md](docs/FRONTEND.md) – UI‑Architektur, Seitenstruktur, Designprinzipien

## ✨ Features

| Feature | Description |
|---------|-------------|
| **🧠 Claim Decomposition** | Breaks text into verifiable atomic claims using LLM-powered extraction |
| **📊 Multi-Source Verification** | Validates against Neo4j graph, Qdrant vectors, Wikipedia, and Context7 |
| **⚡ High Performance** | Session pooling, batch processing, parallel verification, Redis caching |
| **🎯 Trust Scoring** | Evidence-ratio based scoring with confidence intervals (0.0 - 1.0) |
| **🔌 Pluggable Architecture** | Hexagonal design - easily swap knowledge sources and strategies |

## 📁 Project Structure

```
open-hallucination-index/
├── api/                    # Python FastAPI Backend
│   ├── src/                # Main source code
│   │   └── open_hallucination_index/
│   │       ├── domain/     # Core entities (Claim, Evidence, TrustScore)
│   │       ├── ports/      # Abstract interfaces
│   │       ├── application/# Use-case orchestration
│   │       ├── adapters/   # External service implementations
│   │       ├── infrastructure/ # Config, DI, lifecycle
│   │       └── api/        # FastAPI routes
│   ├── tests/              # Unit & integration tests
│   ├── scripts/            # Utility scripts
│   └── pyproject.toml      # Python dependencies
├── frontend/               # Next.js Frontend Application
│   ├── src/                # React/Next.js source code
│   ├── e2e/                # Playwright E2E tests
│   └── package.json        # Node.js dependencies
├── docs/                   # Documentation
│   ├── CONTRIBUTING.md     # Contribution guidelines
│   ├── CODE_OF_CONDUCT.md  # Community standards
│   └── PUBLIC_ACCESS.md    # Public access documentation
├── .github/                # GitHub configuration
│   ├── workflows/          # CI/CD pipelines
│   └── ISSUE_TEMPLATE/     # Issue templates
├── README.md               # This file
├── LICENSE                 # MIT License
└── SECURITY.md             # Security policy
```

## 🚀 Getting Started

### Prerequisites

- **Python 3.14+** for the API
- **Node.js 22+** for the frontend
- **Your own infrastructure** for knowledge sources (see [Infrastructure](#infrastructure))

### API Setup

```bash
cd api

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install with development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Start the API server
ohi-server
```

### Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Run development server
npm run dev

# Run tests
npm run test
```

## 🏗️ Infrastructure

> **⚠️ Important:** This repository contains only the **API and Frontend source code**. Users are responsible for deploying and managing their own infrastructure.

### Required Services

The OHI API requires the following external services:

| Service | Purpose | Documentation |
|---------|---------|---------------|
| **Neo4j** | Graph database for structured knowledge | [neo4j.com](https://neo4j.com/) |
| **Qdrant** | Vector database for semantic search | [qdrant.tech](https://qdrant.tech/) |
| **Redis** | Caching layer (optional) | [redis.io](https://redis.io/) |
| **LLM Service** | For claim decomposition (OpenAI, vLLM, etc.) | [vllm.ai](https://vllm.ai/) |

### Configuration

Create a `.env` file in the `api/` directory:

```env
# API Settings
API_HOST=0.0.0.0
API_PORT=8080
API_API_KEY=your-secret-api-key

# LLM Configuration
LLM_BASE_URL=http://your-llm-service:8000/v1
LLM_MODEL=mistralai/Mistral-7B-Instruct-v0.2
LLM_API_KEY=your-llm-api-key

# Neo4j Graph Database
NEO4J_URI=bolt://your-neo4j-host:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your-neo4j-password

# Qdrant Vector Database
QDRANT_HOST=your-qdrant-host
QDRANT_PORT=6333

# Redis Cache (optional)
REDIS_HOST=your-redis-host
REDIS_PORT=6379
REDIS_ENABLED=true

# MCP Sources (optional)
MCP_WIKIPEDIA_ENABLED=true
MCP_CONTEXT7_ENABLED=true
```

### Deployment Options

You can deploy these services using:

- **Docker Compose** (create your own compose file)
- **Kubernetes** (Helm charts recommended)
- **Managed Services** (Neo4j Aura, Qdrant Cloud, Redis Cloud)
- **Self-hosted** on bare metal or VMs

## 📖 API Reference

Die vollständige API‑Dokumentation mit Request/Response‑Schemas, Beispielaufrufen, Error‑Konzept und Strategien befindet sich in [docs/API.md](docs/API.md).

## 🧪 Development

### Running Tests

**API:**
```bash
cd api
pytest tests/ -v
mypy src
ruff check src tests
```

**Frontend:**
```bash
cd frontend
npm run test
npm run lint
npm run test:e2e
```

### Verification Strategies

| Strategy | Description | Use Case |
|----------|-------------|----------|
| `mcp_enhanced` | Query external sources (Wikipedia, Context7) + local stores | **Recommended** - Most comprehensive |
| `hybrid` | Parallel graph + vector search | Fast local-only verification |
| `cascading` | Graph first, vector fallback | When exact matches preferred |
| `graph_exact` | Neo4j only | Known entity verification |
| `vector_semantic` | Qdrant only | Semantic similarity matching |

## 🤝 Contributing

Contributions are welcome! Please read our [Contributing Guide](docs/CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Please also review our [Code of Conduct](docs/CODE_OF_CONDUCT.md).

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [FastAPI](https://fastapi.tiangolo.com/) - Modern Python web framework
- [Next.js](https://nextjs.org/) - React framework for the frontend
- [Neo4j](https://neo4j.com/) - Graph database
- [Qdrant](https://qdrant.tech/) - Vector search engine
- [MCP](https://modelcontextprotocol.io/) - Model Context Protocol

---

<p align="center">
  Made with ❤️ by the OHI Team
</p>
