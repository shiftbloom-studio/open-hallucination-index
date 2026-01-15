# syntax=docker/dockerfile:1

FROM node:24 AS builder

WORKDIR /app

# Accept build arguments
ARG NEXT_PUBLIC_SUPABASE_URL
ARG NEXT_PUBLIC_SUPABASE_ANON_KEY
ARG NEXT_PUBLIC_APP_URL
ARG DEFAULT_API_URL
ARG DEFAULT_API_KEY

# Set them as environment variables for the build
ENV NEXT_PUBLIC_SUPABASE_URL=$NEXT_PUBLIC_SUPABASE_URL
ENV NEXT_PUBLIC_SUPABASE_ANON_KEY=$NEXT_PUBLIC_SUPABASE_ANON_KEY
ENV NEXT_PUBLIC_APP_URL=$NEXT_PUBLIC_APP_URL
ENV DEFAULT_API_URL=$DEFAULT_API_URL
ENV DEFAULT_API_KEY=$DEFAULT_API_KEY

COPY package*.json ./
# Use BuildKit cache mount to avoid re-downloading npm packages
RUN --mount=type=cache,target=/root/.npm \
    npm ci

COPY . .
RUN npm run build

FROM node:24-slim

WORKDIR /app

ENV NODE_ENV=production

COPY --from=builder /app/next.config.ts ./
COPY --from=builder /app/public ./public
COPY --from=builder /app/.next ./.next
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/package.json ./package.json

EXPOSE 3000

CMD ["npm", "start"]