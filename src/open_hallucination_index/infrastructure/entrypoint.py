"""
Application Entrypoint
======================

CLI entrypoint for running the API server.
"""

from __future__ import annotations

import logging
import sys
import uvicorn
import uvloop


def main() -> None:
    """Run the API server using uvicorn."""
    uvloop.install()
    
    from open_hallucination_index.infrastructure.config import get_settings

    settings = get_settings()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, settings.log_level),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    from open_hallucination_index.infrastructure.logging import HealthLiveAccessFilter

    logger = logging.getLogger(__name__)
    logger.info(f"Starting {settings.api.title} v{settings.api.version}")
    logger.info(f"Environment: {settings.environment}")

    access_logger = logging.getLogger("uvicorn.access")
    access_logger.addFilter(HealthLiveAccessFilter(min_interval_seconds=120.0))

    # Determine host: use 127.0.0.1 for IPv4 when 0.0.0.0 is configured
    api_host = settings.api.host
    if api_host == "0.0.0.0":
        # 0.0.0.0 binds to all interfaces (IPv4 only)
        pass
    elif api_host == "::":
        # Convert IPv6 all-interfaces to IPv4
        api_host = "0.0.0.0"

    uvicorn.run(
        "open_hallucination_index.api.app:create_app",
        factory=True,
        host=api_host,
        port=settings.api.port,
        workers=settings.api.workers,
        reload=settings.api.debug,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    sys.exit(main() or 0)
