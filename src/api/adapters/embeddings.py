"""
Embedding adapter — tri-mode (local sentence-transformers, remote HTTP, or Bedrock).

The public name `LocalEmbeddingAdapter` is kept unchanged so the DI
container in `config/dependencies.py` doesn't need to know which mode is
active. Mode is selected by env var at construction time:

    OHI_EMBEDDING_BACKEND=local   (default) → in-process sentence-transformers
    OHI_EMBEDDING_BACKEND=remote            → HTTP POST to OHI_EMBEDDING_REMOTE_URL
    OHI_EMBEDDING_BACKEND=bedrock           → Amazon Bedrock Titan Text Embeddings V2

Remote mode exists so Lambda doesn't have to carry torch + a sentence
transformer model — those live on the PC embed container, reached via the
CF tunnel (embed.ohi.shiftbloom.studio).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from threading import Lock
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from config.settings import EmbeddingSettings

logger = logging.getLogger(__name__)

_executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="embedding")
_model_load_lock = Lock()


# --- In-process (legacy, used when OHI_EMBEDDING_BACKEND=local) -------------


@lru_cache(maxsize=1)
def _get_model(model_name: str):
    """Load and cache the sentence transformer model."""
    import torch
    from sentence_transformers import SentenceTransformer

    logger.info(f"Loading embedding model in-process: {model_name}")
    device = "cpu"
    with _model_load_lock:
        model = SentenceTransformer(
            model_name,
            device=device,
            trust_remote_code=False,
            model_kwargs={
                "low_cpu_mem_usage": False,
                "device_map": None,
                "dtype": torch.float32,
            },
        )
    model.eval()
    dim = model.get_sentence_embedding_dimension()
    logger.info(f"In-process embedding model ready on {device}, dim={dim}")
    return model


class _InProcessEmbeddingAdapter:
    def __init__(self, settings: EmbeddingSettings) -> None:
        self._model_name = settings.model_name
        self._batch_size = settings.batch_size
        self._normalize = settings.normalize
        self._model = _get_model(self._model_name)

    @property
    def embedding_dimension(self) -> int:
        return self._model.get_sentence_embedding_dimension()

    def _embed_sync(self, text: str) -> list[float]:
        vec = self._model.encode(
            text, normalize_embeddings=self._normalize, show_progress_bar=False
        )
        return vec.tolist()

    def _embed_batch_sync(self, texts: list[str]) -> list[list[float]]:
        vecs = self._model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=self._normalize,
            show_progress_bar=False,
        )
        return vecs.tolist()

    async def generate_embedding(self, text: str) -> list[float]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(_executor, self._embed_sync, text)

    async def generate_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(_executor, self._embed_batch_sync, texts)

    async def health_check(self) -> bool:
        try:
            v = await self.generate_embedding("test")
            return len(v) > 0
        except Exception as e:
            logger.warning(f"In-process embedding health check failed: {e}")
            return False


# --- Remote (used when OHI_EMBEDDING_BACKEND=remote) ------------------------


class _RemoteEmbeddingAdapter:
    """
    HTTP client for the PC-side pc-embed service.

    Sends CF Access service-token headers if OHI_CF_ACCESS_CLIENT_ID and
    OHI_CF_ACCESS_CLIENT_SECRET are set (same pattern the other tunnel
    adapters will adopt); omits them otherwise.
    """

    def __init__(self, settings: EmbeddingSettings) -> None:
        import httpx  # local import — keeps the module importable on bare images

        base_url = os.environ.get("OHI_EMBEDDING_REMOTE_URL")
        if not base_url:
            raise RuntimeError(
                "OHI_EMBEDDING_BACKEND=remote requires OHI_EMBEDDING_REMOTE_URL "
                "(e.g. https://embed.ohi.shiftbloom.studio)"
            )
        self._base_url = base_url.rstrip("/")
        self._model_name = settings.model_name

        timeout = float(os.environ.get("OHI_EMBEDDING_REMOTE_TIMEOUT_S", "10"))
        headers: dict[str, str] = {}
        access_id = os.environ.get("OHI_CF_ACCESS_CLIENT_ID")
        access_secret = os.environ.get("OHI_CF_ACCESS_CLIENT_SECRET")
        if access_id and access_secret:
            headers["CF-Access-Client-Id"] = access_id
            headers["CF-Access-Client-Secret"] = access_secret

        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(timeout, connect=5.0),
            headers=headers,
        )
        self._dim_cache: int | None = None
        logger.info(
            f"Remote embedding adapter configured: {self._base_url} "
            f"(access_headers={'yes' if access_id and access_secret else 'no'})"
        )

    @property
    def embedding_dimension(self) -> int:
        if self._dim_cache is None:
            raise RuntimeError(
                "Embedding dimension unknown until first call — call "
                "generate_embedding() once or await warmup()."
            )
        return self._dim_cache

    async def warmup(self) -> None:
        await self.generate_embedding("warmup")

    async def generate_embedding(self, text: str) -> list[float]:
        resp = await self._client.post("/encode", json={"text": text})
        resp.raise_for_status()
        payload = resp.json()
        vec = payload["vector"]
        self._dim_cache = payload.get("dim", len(vec))
        return vec

    async def generate_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        resp = await self._client.post("/encode/batch", json={"texts": texts})
        resp.raise_for_status()
        payload = resp.json()
        vecs = payload["vectors"]
        if vecs:
            self._dim_cache = payload.get("dim", len(vecs[0]))
        return vecs

    async def health_check(self) -> bool:
        try:
            resp = await self._client.get("/health/live")
            return resp.status_code == 200
        except Exception as e:
            logger.warning(f"Remote embedding health check failed: {e}")
            return False


# --- Bedrock (used when OHI_EMBEDDING_BACKEND=bedrock) ----------------------


class _BedrockEmbeddingAdapter:
    """Amazon Bedrock Titan Text Embeddings V2 client."""

    def __init__(self, settings: EmbeddingSettings) -> None:
        import boto3
        from botocore.config import Config

        self._region = settings.bedrock_region
        self._model_id = settings.bedrock_model_id
        self._dim = settings.bedrock_dimension
        self._normalize = settings.normalize
        self._batch_concurrency = settings.bedrock_batch_concurrency

        config = Config(
            connect_timeout=5,
            read_timeout=settings.bedrock_timeout_seconds,
            retries={"max_attempts": 3, "mode": "standard"},
        )
        self._client = boto3.client(
            service_name="bedrock-runtime",
            region_name=self._region,
            config=config,
        )
        logger.info(
            "Bedrock embedding adapter configured: model=%s region=%s dim=%d normalize=%s",
            self._model_id,
            self._region,
            self._dim,
            self._normalize,
        )

    @property
    def embedding_dimension(self) -> int:
        return self._dim

    def _embed_sync(self, text: str) -> list[float]:
        if not text.strip():
            raise ValueError("Bedrock embedding input must be non-empty")

        request = json.dumps(
            {
                "inputText": text,
                "dimensions": self._dim,
                "normalize": self._normalize,
                "embeddingTypes": ["float"],
            }
        )
        response = self._client.invoke_model(
            body=request,
            modelId=self._model_id,
            accept="application/json",
            contentType="application/json",
        )
        body = response.get("body")
        if body is None:
            raise RuntimeError("Bedrock embedding response missing body")
        payload = json.loads(body.read())
        vector = payload.get("embedding")
        if vector is None:
            by_type = payload.get("embeddingsByType") or {}
            vector = by_type.get("float")
        if not isinstance(vector, list) or not vector:
            raise RuntimeError("Bedrock embedding response missing float embedding")

        out = [float(v) for v in vector]
        if len(out) != self._dim:
            logger.warning(
                "Bedrock embedding dimension mismatch: expected=%d actual=%d",
                self._dim,
                len(out),
            )
        return out

    async def generate_embedding(self, text: str) -> list[float]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(_executor, self._embed_sync, text)

    async def generate_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        semaphore = asyncio.Semaphore(self._batch_concurrency)

        async def _run(text: str) -> list[float]:
            async with semaphore:
                return await self.generate_embedding(text)

        return await asyncio.gather(*(_run(text) for text in texts))

    async def health_check(self) -> bool:
        try:
            vec = await self.generate_embedding("test")
            return len(vec) > 0
        except Exception as e:
            logger.warning(f"Bedrock embedding health check failed: {e}")
            return False


# --- Public facade ----------------------------------------------------------


class LocalEmbeddingAdapter:
    """
    Public name preserved for backward-compat with `config/dependencies.py`.
    At construction time, delegates to either the in-process or remote
    implementation based on the `OHI_EMBEDDING_BACKEND` env var.
    """

    def __init__(self, settings: EmbeddingSettings) -> None:
        backend = os.environ.get("OHI_EMBEDDING_BACKEND", "local").lower()
        if backend == "remote":
            self._impl: (
                _InProcessEmbeddingAdapter
                | _RemoteEmbeddingAdapter
                | _BedrockEmbeddingAdapter
            ) = (
                _RemoteEmbeddingAdapter(settings)
            )
            logger.info("LocalEmbeddingAdapter facade → remote backend")
        elif backend == "bedrock":
            self._impl = _BedrockEmbeddingAdapter(settings)
            logger.info("LocalEmbeddingAdapter facade → bedrock backend")
        elif backend == "local":
            self._impl = _InProcessEmbeddingAdapter(settings)
            logger.info("LocalEmbeddingAdapter facade → in-process backend")
        else:
            raise RuntimeError(f"Unsupported OHI_EMBEDDING_BACKEND={backend!r}")

    @property
    def embedding_dimension(self) -> int:
        return self._impl.embedding_dimension

    async def generate_embedding(self, text: str) -> list[float]:
        return await self._impl.generate_embedding(text)

    async def generate_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        return await self._impl.generate_embeddings_batch(texts)

    async def health_check(self) -> bool:
        return await self._impl.health_check()
