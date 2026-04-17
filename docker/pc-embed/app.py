"""
OHI PC-side embedding service.

Runs sentence-transformers/all-MiniLM-L12-v2 in a dedicated long-lived
container. Lambda's RemoteEmbeddingAdapter POSTs text here instead of
loading torch in-process. The contract is deliberately tiny.
"""

from __future__ import annotations

import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

MODEL_NAME = os.environ.get("OHI_EMBED_MODEL", "all-MiniLM-L12-v2")
NORMALIZE = os.environ.get("OHI_EMBED_NORMALIZE", "true").lower() in ("1", "true", "yes")
BATCH_SIZE = int(os.environ.get("OHI_EMBED_BATCH_SIZE", "32"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("ohi-embed")

_executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="embed")
_model: SentenceTransformer | None = None


def _load_model() -> SentenceTransformer:
    global _model
    if _model is None:
        logger.info(f"Loading embedding model: {MODEL_NAME}")
        _model = SentenceTransformer(MODEL_NAME, device="cpu")
        _model.eval()
        dim = _model.get_sentence_embedding_dimension()
        logger.info(f"Model ready, dim={dim}, normalize={NORMALIZE}, batch_size={BATCH_SIZE}")
    return _model


app = FastAPI(title="ohi-embed", version="1")


@app.on_event("startup")
def _startup() -> None:
    _load_model()


class EncodeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=32_768)


class EncodeBatchRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=512)


class EncodeResponse(BaseModel):
    vector: list[float]
    dim: int
    model: str


class EncodeBatchResponse(BaseModel):
    vectors: list[list[float]]
    dim: int
    model: str


def _encode_sync(text: str) -> list[float]:
    model = _load_model()
    vec = model.encode(text, normalize_embeddings=NORMALIZE, show_progress_bar=False)
    return vec.tolist()


def _encode_batch_sync(texts: list[str]) -> list[list[float]]:
    model = _load_model()
    vecs = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        normalize_embeddings=NORMALIZE,
        show_progress_bar=False,
    )
    return vecs.tolist()


@app.post("/encode", response_model=EncodeResponse)
async def encode(req: EncodeRequest) -> EncodeResponse:
    loop = asyncio.get_running_loop()
    try:
        vector = await loop.run_in_executor(_executor, _encode_sync, req.text)
    except Exception as exc:
        logger.exception("encode failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return EncodeResponse(vector=vector, dim=len(vector), model=MODEL_NAME)


@app.post("/encode/batch", response_model=EncodeBatchResponse)
async def encode_batch(req: EncodeBatchRequest) -> EncodeBatchResponse:
    loop = asyncio.get_running_loop()
    try:
        vectors = await loop.run_in_executor(_executor, _encode_batch_sync, req.texts)
    except Exception as exc:
        logger.exception("encode_batch failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    dim = len(vectors[0]) if vectors else 0
    return EncodeBatchResponse(vectors=vectors, dim=dim, model=MODEL_NAME)


@app.get("/health/live")
async def health_live() -> dict[str, Any]:
    return {"status": "ok", "model": MODEL_NAME}


@app.get("/health/ready")
async def health_ready() -> JSONResponse:
    try:
        vec = await asyncio.get_running_loop().run_in_executor(_executor, _encode_sync, "ready")
        return JSONResponse({"status": "ok", "dim": len(vec), "model": MODEL_NAME})
    except Exception as exc:
        logger.warning(f"readiness failed: {exc}")
        return JSONResponse({"status": "error", "detail": str(exc)}, status_code=503)
