"""OHI Server Routes.

v1 routes (admin, track, verify) have been removed; v2 routes land in
Phase 1 tasks 1.8-1.12.
"""

from server.routes.health import router as health_router

__all__ = ["health_router"]
