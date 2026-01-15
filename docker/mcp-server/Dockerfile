# OHI MCP Server - Unified Knowledge Sources
# ===========================================
#
# High-performance MCP server aggregating 13+ knowledge sources.

FROM node:24-alpine AS builder

WORKDIR /app

# Install dependencies
COPY src/ohi-mcp-server/package*.json ./
RUN npm ci

# Copy source and build
COPY src/ohi-mcp-server/tsconfig.json ./
COPY src/ohi-mcp-server/src ./src
RUN npm run build

# Production image
FROM node:24-alpine AS production

WORKDIR /app

ENV NODE_ENV=production

# Healthcheck dependency
RUN apk add --no-cache curl

# Install production dependencies only
COPY src/ohi-mcp-server/package*.json ./
RUN npm ci --omit=dev && npm cache clean --force

# Copy built application
COPY --from=builder /app/dist ./dist

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -fsS http://localhost:8080/health || exit 1

# Run as non-root user
RUN addgroup -g 1001 -S nodejs && \
    adduser -S nodejs -u 1001
USER nodejs

EXPOSE 8080

CMD ["node", "dist/index.js"]
