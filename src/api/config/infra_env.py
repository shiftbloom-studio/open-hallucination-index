"""Infra environment-variable accessor.

This module is deliberately separate from `settings.py` (which is user WIP and
off-limits). It reads environment variables populated by the Lambda runtime
(set in Terraform via `aws_lambda_function.environment.variables`).

Every accessor raises `KeyError` with a clear message if the var is missing —
fail-fast at Lambda cold start is intentional.
"""

from __future__ import annotations

import os


def _require(name: str) -> str:
    v = os.environ.get(name)
    if not v:
        raise KeyError(
            f"Required environment variable {name} is unset. "
            f"This value is written by Terraform's compute/ layer; check Lambda config."
        )
    return v


# --- Secret ARNs (from Secrets Manager) ---
def gemini_api_key_secret_arn() -> str:
    return _require("OHI_GEMINI_KEY_SECRET_ARN")


def internal_bearer_secret_arn() -> str:
    return _require("OHI_INTERNAL_BEARER_SECRET_ARN")


def edge_secret_arn() -> str:
    return _require("OHI_CF_EDGE_SECRET_ARN")


def cf_access_service_token_secret_arn() -> str:
    return _require("OHI_CF_ACCESS_SERVICE_TOKEN_SECRET_ARN")


def cloudflared_tunnel_token_secret_arn() -> str:
    return _require("OHI_CLOUDFLARED_TUNNEL_TOKEN_SECRET_ARN")


def labeler_tokens_secret_arn() -> str:
    return _require("OHI_LABELER_TOKENS_SECRET_ARN")


def pc_origin_credentials_secret_arn() -> str:
    return _require("OHI_PC_ORIGIN_CREDENTIALS_SECRET_ARN")


def neo4j_credentials_secret_arn() -> str:
    return _require("OHI_NEO4J_CREDENTIALS_SECRET_ARN")


# --- Tunnel hostnames ---
def tunnel_neo4j_host() -> str:
    return _require("OHI_CF_TUNNEL_HOSTNAME_NEO4J")


def tunnel_qdrant_host() -> str:
    return _require("OHI_CF_TUNNEL_HOSTNAME_QDRANT")


def tunnel_pg_rest_host() -> str:
    return _require("OHI_CF_TUNNEL_HOSTNAME_PG_REST")


def tunnel_webdis_host() -> str:
    return _require("OHI_CF_TUNNEL_HOSTNAME_WEBDIS")


def tunnel_embed_host() -> str:
    """Hostname of the PC-side embedding service reached via the CF tunnel.

    Set by Terraform's compute/ layer. Consumed indirectly via the
    OHI_EMBEDDING_REMOTE_URL env var (which is built from this hostname
    at Terraform-apply time).
    """
    return _require("OHI_CF_TUNNEL_HOSTNAME_EMBED")


# --- Embedding backend selection ---
def embedding_backend() -> str:
    """Which embedding implementation to wire up. 'local' (in-process
    sentence-transformers) or 'remote' (HTTP to the pc-embed service).

    Default 'local' preserves pre-Lambda dev behavior; Lambda is set to
    'remote' via Terraform so torch doesn't have to live in the image.
    """
    return os.environ.get("OHI_EMBEDDING_BACKEND", "local").lower()


def embedding_remote_url() -> str | None:
    """Base URL of the pc-embed service, e.g. https://embed.ohi.shiftbloom.studio.
    Only required when embedding_backend() == 'remote'.
    """
    v = os.environ.get("OHI_EMBEDDING_REMOTE_URL", "").strip()
    return v or None


# --- Runtime config ---
def gemini_model() -> str:
    return os.environ.get("OHI_GEMINI_MODEL", "gemini-3-flash-preview")


def gemini_daily_ceiling_eur() -> float:
    return float(os.environ.get("OHI_GEMINI_DAILY_CEILING_EUR", "0"))  # 0 = unlimited


def cors_origins() -> list[str]:
    """Comma-separated list of allowed CORS origins from env.

    Empty list means "fall back to app settings defaults". Set by the compute
    Terraform layer to the production frontend origin(s).

    Example: OHI_CORS_ORIGINS=https://ohi.shiftbloom.studio,https://staging.ohi.shiftbloom.studio
    """
    raw = os.environ.get("OHI_CORS_ORIGINS", "").strip()
    if not raw:
        return []
    return [origin.strip() for origin in raw.split(",") if origin.strip()]
