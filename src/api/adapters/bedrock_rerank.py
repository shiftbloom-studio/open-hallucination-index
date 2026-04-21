"""
Bedrock rerank adapter.

Wraps the Agents-for-Bedrock Runtime `Rerank` operation so retrieval can
rerank ANN candidates without coupling pipeline code to boto3 payload shapes.
"""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from config.settings import RetrievalSettings

logger = logging.getLogger(__name__)
_rerank_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="bedrock-rerank")


@dataclass(frozen=True)
class RerankScore:
    """Single reranked document score."""

    index: int
    relevance_score: float


class BedrockRerankAdapter:
    """Thin async facade over Bedrock agent-runtime `rerank`."""

    def __init__(self, settings: "RetrievalSettings") -> None:
        self._enabled = settings.bedrock_rerank_enabled
        self._region = settings.bedrock_rerank_region
        self._model_id = settings.bedrock_rerank_model_id
        self._model_arn = settings.bedrock_rerank_model_arn or (
            f"arn:aws:bedrock:{self._region}::foundation-model/{self._model_id}"
        )
        self._default_top_k = settings.final_top_k

        if not self._enabled:
            self._client = None
            logger.info("Bedrock rerank disabled by config")
            return

        import boto3
        from botocore.config import Config

        self._client = boto3.client(
            "bedrock-agent-runtime",
            region_name=self._region,
            config=Config(
                connect_timeout=5,
                read_timeout=settings.bedrock_rerank_timeout_seconds,
                retries={"max_attempts": 3, "mode": "standard"},
            ),
        )
        logger.info(
            "Bedrock rerank configured: model=%s region=%s",
            self._model_arn,
            self._region,
        )

    @property
    def enabled(self) -> bool:
        return self._enabled and self._client is not None

    async def rerank(
        self,
        *,
        query: str,
        documents: list[str],
        top_k: int | None = None,
    ) -> list[RerankScore]:
        if not documents:
            return []
        if not self.enabled:
            return [
                RerankScore(index=i, relevance_score=0.0) for i in range(len(documents))
            ]

        wanted = min(top_k or self._default_top_k, len(documents))
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            _rerank_executor,
            self._rerank_sync,
            query,
            documents,
            wanted,
        )

    def _rerank_sync(
        self,
        query: str,
        documents: list[str],
        top_k: int,
    ) -> list[RerankScore]:
        if self._client is None:
            return [RerankScore(index=i, relevance_score=0.0) for i in range(len(documents))]

        response = self._client.rerank(
            queries=[
                {
                    "type": "TEXT",
                    "textQuery": {"text": query},
                }
            ],
            rerankingConfiguration={
                "type": "BEDROCK_RERANKING_MODEL",
                "bedrockRerankingConfiguration": {
                    "modelConfiguration": {"modelArn": self._model_arn},
                    "numberOfResults": top_k,
                },
            },
            sources=[
                {
                    "type": "INLINE",
                    "inlineDocumentSource": {
                        "type": "TEXT",
                        "textDocument": {"text": doc},
                    },
                }
                for doc in documents
            ],
        )
        results = response.get("results") or []
        ranked: list[RerankScore] = []
        for item in results:
            idx = item.get("index")
            score = item.get("relevanceScore")
            if isinstance(idx, int):
                ranked.append(
                    RerankScore(index=idx, relevance_score=float(score or 0.0))
                )
        if not ranked:
            raise RuntimeError("Bedrock rerank returned no results")
        ranked.sort(key=lambda row: row.relevance_score, reverse=True)
        return ranked

