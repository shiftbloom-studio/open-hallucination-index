from __future__ import annotations

import io
import json

import boto3
import botocore.config
import pytest

from adapters.embeddings import LocalEmbeddingAdapter
from config.settings import EmbeddingSettings


class _FakeBedrockClient:
    def __init__(self, response_payload: dict[str, object]) -> None:
        self._response_payload = response_payload
        self.calls: list[dict[str, object]] = []

    def invoke_model(self, **kwargs: object) -> dict[str, io.BytesIO]:
        self.calls.append(kwargs)
        return {"body": io.BytesIO(json.dumps(self._response_payload).encode("utf-8"))}


class _FakeConfig:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs


@pytest.mark.asyncio
async def test_bedrock_embedding_adapter_invokes_titan_with_expected_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = _FakeBedrockClient({"embedding": [0.1, 0.2, 0.3, 0.4]})
    seen: dict[str, object] = {}

    def _fake_boto3_client(
        service_name: str, region_name: str, config: _FakeConfig
    ) -> _FakeBedrockClient:
        seen["service_name"] = service_name
        seen["region_name"] = region_name
        seen["config"] = config
        return fake_client

    monkeypatch.setattr(boto3, "client", _fake_boto3_client)
    monkeypatch.setattr(botocore.config, "Config", _FakeConfig)
    monkeypatch.setenv("OHI_EMBEDDING_BACKEND", "bedrock")

    settings = EmbeddingSettings(
        bedrock_model_id="amazon.titan-embed-text-v2:0",
        bedrock_dimension=256,
        bedrock_region="eu-central-1",
        bedrock_batch_concurrency=2,
        bedrock_timeout_seconds=12.0,
        normalize=False,
    )
    adapter = LocalEmbeddingAdapter(settings)

    vector = await adapter.generate_embedding("hello from titan")

    assert vector == [0.1, 0.2, 0.3, 0.4]
    assert seen["service_name"] == "bedrock-runtime"
    assert seen["region_name"] == "eu-central-1"
    assert isinstance(seen["config"], _FakeConfig)
    assert seen["config"].kwargs == {
        "connect_timeout": 5,
        "read_timeout": 12.0,
        "retries": {"max_attempts": 3, "mode": "standard"},
    }

    assert len(fake_client.calls) == 1
    assert fake_client.calls[0]["modelId"] == "amazon.titan-embed-text-v2:0"
    assert fake_client.calls[0]["accept"] == "application/json"
    assert fake_client.calls[0]["contentType"] == "application/json"
    assert json.loads(fake_client.calls[0]["body"]) == {
        "inputText": "hello from titan",
        "dimensions": 256,
        "normalize": False,
        "embeddingTypes": ["float"],
    }


@pytest.mark.asyncio
async def test_bedrock_embedding_adapter_accepts_embeddings_by_type_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = _FakeBedrockClient(
        {"embeddingsByType": {"float": [0.9, 0.8, 0.7, 0.6]}}
    )

    monkeypatch.setattr(
        boto3,
        "client",
        lambda service_name, region_name, config: fake_client,
    )
    monkeypatch.setattr(botocore.config, "Config", _FakeConfig)
    monkeypatch.setenv("OHI_EMBEDDING_BACKEND", "bedrock")

    settings = EmbeddingSettings(bedrock_dimension=256)
    adapter = LocalEmbeddingAdapter(settings)

    vector = await adapter.generate_embedding("fallback shape")

    assert vector == [0.9, 0.8, 0.7, 0.6]
