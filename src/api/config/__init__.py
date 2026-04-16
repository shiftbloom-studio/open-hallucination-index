"""
OHI Config - Settings and Dependencies
=======================================

Configuration, dependency injection, and application bootstrap.
"""

from config.dependencies import get_llm_provider
from config.entrypoint import main
from config.settings import Settings, get_settings

__all__ = [
    "Settings",
    "get_settings",
    "get_llm_provider",
    "main",
]
