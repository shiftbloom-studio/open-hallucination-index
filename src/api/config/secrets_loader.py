"""Thin wrapper around boto3 SecretsManager with TTL cache + bootstrap-grace.

Two failure modes:
- `ConfigurationError` — critical secret missing; Lambda cold-start fails loudly.
- `BootstrapGraceSecret` — specific secrets that are legitimately empty between
  `secrets/` apply and `cloudflare/` apply (spec §4.3); callers that raise this
  should return 503 `{"status":"bootstrapping"}` rather than 500.

Importable lazily; do NOT instantiate at module import time, because Lambda
cold-start imports happen before AWS_REGION is set in some test harnesses.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

import boto3  # type: ignore[import-untyped]
from botocore.exceptions import ClientError  # type: ignore[import-untyped]


class ConfigurationError(RuntimeError):
    """Critical secret unavailable."""


class BootstrapGraceSecret(RuntimeError):
    """Bootstrap-grace secret unavailable; caller should 503 not 500."""


@dataclass
class _CacheEntry:
    value: str
    fetched_at: float


class SecretsLoader:
    """Lazy, TTL-cached accessor for AWS Secrets Manager values.

    Usage:
        loader = SecretsLoader()
        key = loader.get(os.environ["OHI_GEMINI_KEY_SECRET_ARN"])
    """

    def __init__(self, client: Any | None = None, ttl_seconds: int = 600) -> None:
        self._client = client or boto3.client("secretsmanager")
        self._ttl = ttl_seconds
        self._cache: dict[str, _CacheEntry] = {}

    def get(self, secret_arn: str, *, bootstrap_grace: bool = False) -> str:
        """Return the secret's string value, possibly from cache."""
        cached = self._cache.get(secret_arn)
        if cached is not None and (time.monotonic() - cached.fetched_at) < self._ttl:
            return cached.value
        try:
            resp = self._client.get_secret_value(SecretId=secret_arn)
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            if bootstrap_grace and code == "ResourceNotFoundException":
                raise BootstrapGraceSecret(
                    f"Secret {secret_arn} not yet populated; bootstrap-grace tolerated."
                ) from exc
            raise ConfigurationError(f"Cannot read secret {secret_arn}: {code}") from exc
        value = resp.get("SecretString") or ""
        if bootstrap_grace and not value:
            raise BootstrapGraceSecret(
                f"Secret {secret_arn} exists but is empty; bootstrap-grace tolerated."
            )
        self._cache[secret_arn] = _CacheEntry(value=value, fetched_at=time.monotonic())
        return value

    def get_json(self, secret_arn: str, *, bootstrap_grace: bool = False) -> Any:
        return json.loads(self.get(secret_arn, bootstrap_grace=bootstrap_grace))


# Module-level singleton for app-wide reuse. Lazily created on first attribute access.
_loader: SecretsLoader | None = None


def get_loader() -> SecretsLoader:
    global _loader
    if _loader is None:
        _loader = SecretsLoader()
    return _loader
