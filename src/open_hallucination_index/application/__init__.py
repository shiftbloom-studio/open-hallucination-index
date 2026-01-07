"""
Application Layer
=================

Use-case orchestration. This layer coordinates domain logic and ports
to fulfill business requirements. No infrastructure details leak here.
"""

from open_hallucination_index.application.verify_text import VerifyTextUseCase

__all__ = [
    "VerifyTextUseCase",
]
