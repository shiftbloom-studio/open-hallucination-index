"""OHI Server Routes."""

from server.routes.health import router as health_router
from server.routes.track import router as track_router
from server.routes.verify import router as verify_router

__all__ = ["health_router", "track_router", "verify_router"]
