# Context7 MCP Server
# ====================
# Sidecar container providing library documentation via MCP protocol

FROM node:20-slim

LABEL maintainer="ShiftBloom <dev@shiftbloom.com>"
LABEL description="Context7 MCP Server for Open Hallucination Index"

# Install curl for health checks and install context7-mcp globally (as root)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && npm install -g @upstash/context7-mcp

# Create non-root user
RUN useradd --create-home --uid 10001 mcpuser
USER mcpuser
WORKDIR /home/mcpuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:3000/health || exit 1

# Expose MCP port
EXPOSE 3000

# Environment variable for API key (set via docker-compose)
ENV CONTEXT7_API_KEY=""

# Run context7-mcp with HTTP transport
CMD ["context7-mcp", "--transport", "http", "--port", "3000"]
