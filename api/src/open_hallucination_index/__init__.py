"""
Open Hallucination Index & Verification API
============================================

A hexagonal-architecture middleware for verifying LLM-generated text
against trusted knowledge sources (graph DB + vector store).

Layers:
- domain: Core entities and value objects (Claim, Evidence, TrustScore)
- ports: Abstract interfaces (ClaimDecomposer, KnowledgeStore, LLMProvider)
- application: Use-case orchestration (VerifyTextUseCase)
- adapters: Concrete implementations for external services
- infrastructure: Config, DI wiring, entrypoint
- api: FastAPI routes and request/response schemas
"""

__version__ = "0.1.0"
