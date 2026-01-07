"""API request schemas - re-exported from routes for shared use."""

from typing import Literal

from pydantic import BaseModel, Field


class VerifyTextRequest(BaseModel):
    """Request body for text verification."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=100_000,
        description="Text content to verify for factual accuracy.",
    )
    context: str | None = Field(
        default=None,
        max_length=10_000,
        description="Optional context to help with claim disambiguation.",
    )
    strategy: Literal["graph_exact", "vector_semantic", "hybrid", "cascading"] | None = Field(
        default=None,
        description="Verification strategy override.",
    )
    use_cache: bool = Field(default=True)


class BatchVerifyRequest(BaseModel):
    """Request for batch verification."""

    texts: list[str] = Field(..., min_length=1, max_length=50)
    strategy: Literal["graph_exact", "vector_semantic", "hybrid", "cascading"] | None = None
    use_cache: bool = True
