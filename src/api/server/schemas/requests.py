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
    strategy: (
        Literal[
            "graph_exact",
            "vector_semantic",
            "hybrid",
            "cascading",
            "mcp_enhanced",
            "adaptive",
        ]
        | None
    ) = Field(
        default=None,
        description="Verification strategy override.",
    )
    use_cache: bool = Field(default=True)
    target_sources: int | None = Field(
        default=None,
        ge=1,
        le=20,
        description="Preferred number of sources to query during verification.",
    )
    return_evidence: bool = Field(
        default=True,
        description="Whether to include full evidence traces in the response.",
    )
    tier: Literal["local", "default", "max"] | None = Field(
        default=None,
        description="Evidence collection tier.",
    )
    skip_decomposition: bool = Field(
        default=False,
        description="Whether to skip claim decomposition.",
    )


class BatchVerifyRequest(BaseModel):
    """Request for batch verification."""

    texts: list[str] = Field(..., min_length=1, max_length=50)
    strategy: (
        Literal[
            "graph_exact",
            "vector_semantic",
            "hybrid",
            "cascading",
            "mcp_enhanced",
            "adaptive",
        ]
        | None
    ) = None
    use_cache: bool = True
