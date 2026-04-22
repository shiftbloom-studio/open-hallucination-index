#!/bin/sh
# OHI Lambda container entrypoint. Starts the tsnet-backed bolt proxy
# in the background, then hands off to uvicorn. The Lambda Web Adapter
# extension (loaded from /opt/extensions/lambda-adapter) proxies Lambda
# events to uvicorn on PORT=8080.
#
# Design notes:
# - tsproxy itself exits(0) cleanly if TS_AUTHKEY is unset, so the
#   container remains launchable even before Tailscale is provisioned
#   (Phase 2 handoff state). The Python app will just see Neo4j as
#   unreachable and come up in degraded mode (MCP-only evidence).
# - We bounded-wait for tsproxy's local listener before starting
#   uvicorn. Lambda init has a 10-second ceiling; tsnet ephemeral
#   bring-up is typically ~1-3 s, and our wait is capped at 6 s.
# - `exec` on the final uvicorn call so Lambda signals go to uvicorn.

set -u

if [ -n "${TS_AUTHKEY:-}" ] && [ -n "${TS_UPSTREAM:-}" ]; then
    echo "[entrypoint] Starting tsproxy (TS_UPSTREAM=${TS_UPSTREAM})"
    /opt/tsproxy &
    TSPID=$!

    # Wait up to 6 s for the local listener. Each iteration is ~0.25 s so
    # the total is bounded regardless of python-startup jitter.
    LISTEN_HOST_PORT="${TS_LISTEN:-127.0.0.1:7687}"
    LISTEN_PORT="${LISTEN_HOST_PORT##*:}"
    LISTEN_HOST="${LISTEN_HOST_PORT%:*}"
    for _ in 1 2 3 4 5 6 7 8 9 10 11 12; do
        if python3 -c "import socket,sys; socket.create_connection(('${LISTEN_HOST}', ${LISTEN_PORT}), timeout=0.25); sys.exit(0)" 2>/dev/null; then
            echo "[entrypoint] tsproxy listening on ${LISTEN_HOST_PORT}"
            break
        fi
        # Also bail early if tsproxy died so we don't waste the budget.
        if ! kill -0 "${TSPID}" 2>/dev/null; then
            echo "[entrypoint] WARN: tsproxy exited before the local listener was ready; Neo4j will be unreachable this cold start"
            break
        fi
        sleep 0.5
    done
else
    echo "[entrypoint] TS_AUTHKEY and/or TS_UPSTREAM unset; skipping tsproxy. Neo4j will be unreachable (degraded mode)."
fi

exec python -m uvicorn server.app:app --host 0.0.0.0 --port 8080 --workers 1
