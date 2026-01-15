"""
Infrastructure Layer
====================

Cross-cutting concerns: configuration, dependency injection,
application bootstrap, and entrypoint.
"""

from open_hallucination_index.infrastructure.config import Settings, get_settings

__all__ = [
    "Settings",
    "get_settings",
]
