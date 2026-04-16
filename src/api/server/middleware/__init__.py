"""HTTP middleware for the v2 API.

Phase 1 ships the retention middleware (no raw-text persistence by
default). Rate limiting + cost ceiling middlewares land in Task 1.10;
internal-auth middleware in Task 1.10. Infra sub-project adds the
EdgeSecretMiddleware which enforces the CF-injected shared secret on
every request (see spec §7.5).
"""

from server.middleware.edge_secret import EdgeSecretMiddleware
from server.middleware.retention import RetentionMiddleware

__all__ = ["EdgeSecretMiddleware", "RetentionMiddleware"]
