"""Tests for SecretsLoader."""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
from botocore.exceptions import ClientError

from config.secrets_loader import (
    BootstrapGraceSecret,
    ConfigurationError,
    SecretsLoader,
)


@pytest.fixture
def fake_boto_client():
    """A MagicMock standing in for boto3's SecretsManager client."""
    client = MagicMock()
    client.get_secret_value.return_value = {"SecretString": "abc123"}
    return client


def test_loader_fetches_secret_and_caches(fake_boto_client):
    loader = SecretsLoader(client=fake_boto_client, ttl_seconds=600)
    v1 = loader.get("arn:aws:secretsmanager:eu-central-1:111:secret:ohi/gemini-api-key-xx")
    v2 = loader.get("arn:aws:secretsmanager:eu-central-1:111:secret:ohi/gemini-api-key-xx")
    assert v1 == "abc123" == v2
    fake_boto_client.get_secret_value.assert_called_once()


def test_loader_json_parses_when_requested(fake_boto_client):
    fake_boto_client.get_secret_value.return_value = {
        "SecretString": json.dumps({"client_id": "cid", "client_secret": "csec"})
    }
    loader = SecretsLoader(client=fake_boto_client, ttl_seconds=600)
    v = loader.get_json("arn:aws:secretsmanager:eu-central-1:111:secret:ohi/cf-access-token-xx")
    assert v == {"client_id": "cid", "client_secret": "csec"}


def test_loader_raises_on_resource_not_found_for_critical(fake_boto_client):
    err = {"Error": {"Code": "ResourceNotFoundException", "Message": "nope"}}
    fake_boto_client.get_secret_value.side_effect = ClientError(err, "GetSecretValue")
    loader = SecretsLoader(client=fake_boto_client, ttl_seconds=600)
    with pytest.raises(ConfigurationError):
        loader.get("arn:aws:secretsmanager:eu-central-1:111:secret:ohi/gemini-api-key-xx")


def test_loader_tolerates_missing_bootstrap_grace_secret(fake_boto_client):
    err = {"Error": {"Code": "ResourceNotFoundException", "Message": "nope"}}
    fake_boto_client.get_secret_value.side_effect = ClientError(err, "GetSecretValue")
    loader = SecretsLoader(client=fake_boto_client, ttl_seconds=600)
    with pytest.raises(BootstrapGraceSecret):
        loader.get(
            "arn:aws:secretsmanager:eu-central-1:111:secret:ohi/cf-access-service-token-xx",
            bootstrap_grace=True,
        )


def test_loader_tolerates_empty_bootstrap_grace_value(fake_boto_client):
    fake_boto_client.get_secret_value.return_value = {"SecretString": ""}
    loader = SecretsLoader(client=fake_boto_client, ttl_seconds=600)
    with pytest.raises(BootstrapGraceSecret):
        loader.get(
            "arn:aws:secretsmanager:eu-central-1:111:secret:ohi/cloudflared-tunnel-token-xx",
            bootstrap_grace=True,
        )


def test_loader_cache_expires_after_ttl(fake_boto_client, monkeypatch):
    """After TTL elapses, the next get() re-fetches."""
    from config import secrets_loader as sl

    now = [1000.0]
    monkeypatch.setattr(sl.time, "monotonic", lambda: now[0])
    loader = SecretsLoader(client=fake_boto_client, ttl_seconds=600)
    loader.get("arn:aws:...:secret:x")
    now[0] = 2000.0  # past TTL
    loader.get("arn:aws:...:secret:x")
    assert fake_boto_client.get_secret_value.call_count == 2
