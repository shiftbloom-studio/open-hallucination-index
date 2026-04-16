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


# --- Runtime config ---
def gemini_model() -> str:
    return os.environ.get("OHI_GEMINI_MODEL", "gemini-3-flash-preview")


def gemini_daily_ceiling_eur() -> float:
    return float(os.environ.get("OHI_GEMINI_DAILY_CEILING_EUR", "0"))  # 0 = unlimited
