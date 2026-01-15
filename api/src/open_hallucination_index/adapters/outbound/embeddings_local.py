"""
Local Embedding Adapter
=======================

Local embedding generation using sentence-transformers.
Runs on CPU or GPU without external API calls.
"""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from open_hallucination_index.infrastructure.config import EmbeddingSettings

logger = logging.getLogger(__name__)

# Thread pool for running sync model inference
_executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="embedding")


@lru_cache(maxsize=1)
def _get_model(model_name: str):
    """Load and cache the sentence transformer model."""
    from sentence_transformers import SentenceTransformer

    logger.info(f"Loading embedding model: {model_name}")

    # Explicitly determine device - CPU for API containers (GPU is for vLLM)
    device = "cpu"

    # Load model without problematic kwargs that can cause meta tensor issues
    # Force CPU to avoid GPU memory conflicts with vLLM
    model = SentenceTransformer(
        model_name,
        device=device,
        trust_remote_code=False,
    )

    # Ensure all parameters are on the correct device (not meta)
    model = model.to(device)

    # Verify model is ready
    model.eval()

    dim = model.get_sentence_embedding_dimension()
    logger.info(f"Loaded embedding model on {device}, dim={dim}")
    return model


class LocalEmbeddingAdapter:
    """
    Adapter for local embedding generation using sentence-transformers.

    Uses all-MiniLM-L6-v2 by default (384 dimensions, fast, good quality).
    For higher quality, use all-mpnet-base-v2 (768 dimensions).
    """

    def __init__(self, settings: EmbeddingSettings) -> None:
        """
        Initialize the local embedding adapter.

        Args:
            settings: Embedding configuration.
        """
        self._model_name = settings.model_name
        self._batch_size = settings.batch_size
        self._normalize = settings.normalize
        self._model = None

    def _load_model(self):
        """Lazy load the model."""
        if self._model is None:
            self._model = _get_model(self._model_name)
        return self._model

    @property
    def embedding_dimension(self) -> int:
        """Return the embedding dimension for the loaded model."""
        model = self._load_model()
        return model.get_sentence_embedding_dimension()

    def _embed_sync(self, text: str) -> list[float]:
        """Synchronous embedding generation."""
        model = self._load_model()
        embedding = model.encode(
            text,
            normalize_embeddings=self._normalize,
            show_progress_bar=False,
        )
        return embedding.tolist()

    def _embed_batch_sync(self, texts: list[str]) -> list[list[float]]:
        """Synchronous batch embedding generation."""
        model = self._load_model()
        embeddings = model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=self._normalize,
            show_progress_bar=False,
        )
        return embeddings.tolist()

    async def generate_embedding(self, text: str) -> list[float]:
        """
        Generate embedding vector for text.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector as list of floats.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(_executor, self._embed_sync, text)

    async def generate_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embedding vectors for multiple texts efficiently.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.
        """
        if not texts:
            return []
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(_executor, self._embed_batch_sync, texts)

    async def health_check(self) -> bool:
        """Check if the embedding model is loaded and working."""
        try:
            embedding = await self.generate_embedding("test")
            return len(embedding) > 0
        except Exception as e:
            logger.warning(f"Embedding health check failed: {e}")
            return False
