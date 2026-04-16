"""Raw-text retention policy middleware. Spec §11.

OHI v2 is open-access; unauthenticated callers submit arbitrary text.
To minimise the PII/GDPR surface, raw input text is NOT persisted by
default — only the ``sha256(text)`` cache key and the resulting
``DocumentVerdict`` hit Postgres / Redis.

Retention opt-in lives in the ``?retain=true`` query parameter. The
middleware reads it on every request, stamps ``request.state.retain_text``,
and downstream storage (verdict_store, feedback_store) respects the flag.

This middleware deliberately does no other work on the request path —
the actual enforcement happens at the storage boundary, not here. Its
sole purpose is to surface the retention decision explicitly so policy
doesn't leak into route handlers.
"""

from __future__ import annotations

import logging

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger(__name__)


class RetentionMiddleware(BaseHTTPMiddleware):
    """Tag ``request.state.retain_text`` from the ``?retain=true`` query param.

    Default: False. No other side effects.
    """

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        raw = request.query_params.get("retain", "false").lower()
        request.state.retain_text = raw in ("true", "1", "yes")
        if request.state.retain_text:
            logger.debug("Retention opt-in for %s %s", request.method, request.url.path)
        return await call_next(request)
