# Wikipedia MCP Server
# ====================
# Sidecar container providing Wikipedia access via MCP protocol

FROM python:3.12-slim

LABEL maintainer="ShiftBloom <dev@shiftbloom.com>"
LABEL description="Wikipedia MCP Server for Open Hallucination Index"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install wikipedia-mcp
RUN pip install --no-cache-dir wikipedia-mcp

# Create non-root user
RUN useradd --create-home --uid 10001 mcpuser
USER mcpuser
WORKDIR /home/mcpuser

# Health check endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Expose MCP port
EXPOSE 8080

# Run wikipedia-mcp with SSE transport and minimal logging
CMD ["wikipedia-mcp", "--transport", "sse", "--host", "0.0.0.0", "--port", "8080", "--language", "en", "--enable-cache", "--log-level", "WARNING"]
