"""
API Layer
=========

FastAPI routes, request/response schemas, and HTTP-specific logic.
This is the primary (driving) adapter in hexagonal architecture.
"""

from open_hallucination_index.api.app import create_app

__all__ = [
    "create_app",
]
