"""HTTP middleware for the v2 API.

Phase 1 ships the retention middleware (no raw-text persistence by
default). Rate limiting + cost ceiling middlewares land in Task 1.10;
internal-auth middleware in Task 1.10.
"""

from server.middleware.retention import RetentionMiddleware

__all__ = ["RetentionMiddleware"]
