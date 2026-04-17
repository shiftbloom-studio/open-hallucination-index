"""Enforce the X-OHI-Edge-Secret header set by Cloudflare's Transform Rule.

Traffic path: User → CF → (CF adds X-OHI-Edge-Secret: <shared>) → Lambda Function URL.
Lambda Function URL has `auth_type = NONE`, so without this middleware any caller
who knows the Function URL could bypass CF. This middleware closes that gap.

Design:
- Secret value is fetched lazily via a caller-supplied `get_expected_secret` callable.
  In prod, that callable reads from `SecretsLoader.get(infra_env.edge_secret_arn())`.
- `/health/live` is exempt so Lambda's own runtime health checks don't need
  the header. Other /health/* routes still require it.
- Comparison uses `hmac.compare_digest` (constant-time).
"""

from __future__ import annotations

import hmac
import logging
from collections.abc import Awaitable, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp

HEADER_NAME = "X-OHI-Edge-Secret"
EXEMPT_PATHS = frozenset({"/health/live"})

logger = logging.getLogger("ohi.middleware.edge_secret")


class EdgeSecretMiddleware(BaseHTTPMiddleware):
    def __init__(
        self, app: ASGIApp, *, get_expected_secret: Callable[[], str]
    ) -> None:
        super().__init__(app)
        self._get_expected = get_expected_secret

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        # CORS preflight (OPTIONS) is sent by the browser BEFORE any Cloudflare
        # Transform Rule fires on the actual request, so the edge-secret header
        # is not present on preflight. CORSMiddleware (registered separately)
        # owns the OPTIONS response; this middleware gets out of the way.
        if request.method == "OPTIONS":
            return await call_next(request)

        if request.url.path in EXEMPT_PATHS:
            return await call_next(request)

        provided = request.headers.get(HEADER_NAME, "")
        if not provided:
            logger.warning(
                "rate_limit_triggered",
                extra={"reason": "missing_edge_secret", "path": request.url.path},
            )
            return JSONResponse({"detail": "missing_edge_secret"}, status_code=403)

        try:
            expected = self._get_expected()
        except Exception:
            logger.exception("Failed to load edge secret")
            return JSONResponse({"detail": "edge_secret_unavailable"}, status_code=503)

        if not hmac.compare_digest(provided, expected):
            logger.warning(
                "rate_limit_triggered",
                extra={"reason": "invalid_edge_secret", "path": request.url.path},
            )
            return JSONResponse({"detail": "invalid_edge_secret"}, status_code=403)

        return await call_next(request)
