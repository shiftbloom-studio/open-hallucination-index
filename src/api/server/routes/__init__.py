"""OHI Server Routes."""

from server.routes.admin import router as admin_router
from server.routes.health import router as health_router
from server.routes.track import router as track_router
from server.routes.verify import router as verify_router

__all__ = ["admin_router", "health_router", "track_router", "verify_router"]
