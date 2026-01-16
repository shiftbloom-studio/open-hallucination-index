"""
Application Entrypoint
======================

CLI entrypoint for running the API server.
"""

from __future__ import annotations

import sys

import uvicorn
import uvloop


def main() -> None:
    """Run the API server using uvicorn."""
    uvloop.install()

    from config.settings import get_settings

    settings = get_settings()

    # NOTE: Logging is now configured in the lifespan manager to avoid
    # duplicate logs when workers are spawned or app factory is called multiple times.
    # Only log startup messages here (before uvicorn takes over).

    # Basic console output for startup (uvicorn will configure its own logging)
    print(f"Starting {settings.api.title} v{settings.api.version}")
    print(f"Environment: {settings.environment}")

    # Determine host: use 127.0.0.1 for IPv4 when 0.0.0.0 is configured
    api_host = settings.api.host
    if api_host == "0.0.0.0":
        # 0.0.0.0 binds to all interfaces (IPv4 only)
        pass
    elif api_host == "::":
        # Convert IPv6 all-interfaces to IPv4
        api_host = "0.0.0.0"

    uvicorn.run(
        "server.app:create_app",
        factory=True,
        host=api_host,
        port=settings.api.port,
        workers=settings.api.workers,
        reload=settings.api.debug,
        log_level=settings.log_level.lower(),
        access_log=False,
    )


if __name__ == "__main__":
    main()
    sys.exit(0)
