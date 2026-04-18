"""OHI Server Routes.

v1 routes (admin, track, v1 verify) have been removed.
v2 routes mounted so far: health, verify, async_verify (internal).
More added in Phase 1 Tasks 1.11 (retention middleware is not a route),
1.12 (deep), and Phase 4 Tasks 4.2 (feedback) / 4.8 (calibration
report). Stream D2 replaced the SSE streaming plan with polling
(``/api/v2/verify`` + ``/api/v2/verify/status/{id}``) and added the
internal ``/_internal/async-verify`` route.
"""

from server.routes.async_verify import router as async_verify_router
from server.routes.health import router as health_router
from server.routes.verify import router as verify_router

__all__ = ["async_verify_router", "health_router", "verify_router"]
