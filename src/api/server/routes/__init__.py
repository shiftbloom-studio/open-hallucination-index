"""OHI Server Routes.

v1 routes (admin, track, v1 verify) have been removed.
v2 routes mounted so far: health, verify. More added in Phase 1 Tasks
1.9 (stream), 1.11 (retention middleware is not a route), 1.12 (deep),
and Phase 4 Tasks 4.2 (feedback) / 4.8 (calibration report).
"""

from server.routes.health import router as health_router
from server.routes.verify import router as verify_router

__all__ = ["health_router", "verify_router"]
