"""
OHI - Open Hallucination Index
==============================

High-performance middleware for detecting LLM hallucinations by decomposing
claims and verifying them against trusted knowledge sources.

Package Structure:
- models/: Domain entities (Claim, Evidence, TrustScore)
- interfaces/: Abstract contracts (ports)
- pipeline/: Verification flow stages
- adapters/: Infrastructure integrations (Neo4j, Qdrant, Redis, MCP)
- server/: FastAPI application
- config/: Settings, DI, entrypoint
- services/: High-level application services
"""

__version__ = "0.1.0"
