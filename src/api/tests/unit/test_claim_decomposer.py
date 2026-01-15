"""
Tests for LLM-Based Claim Decomposer
====================================
"""

from unittest.mock import AsyncMock

import pytest

from open_hallucination_index.domain.entities import ClaimType
from open_hallucination_index.domain.services.claim_decomposer import (
    DecompositionError,
    LLMClaimDecomposer,
)
from open_hallucination_index.ports.llm_provider import LLMResponse


class TestLLMClaimDecomposer:
    """Tests for the LLMClaimDecomposer."""

    @pytest.fixture
    def mock_llm_provider(self) -> AsyncMock:
        """Create a mock LLM provider."""
        mock = AsyncMock()
        mock.health_check.return_value = True
        return mock

    @pytest.fixture
    def decomposer(self, mock_llm_provider: AsyncMock) -> LLMClaimDecomposer:
        """Create a decomposer instance with mocked LLM."""
        return LLMClaimDecomposer(
            llm_provider=mock_llm_provider,
            max_claims=50,
        )

    @pytest.mark.asyncio
    async def test_decompose_empty_text(self, decomposer: LLMClaimDecomposer) -> None:
        """Test decomposing empty text returns empty list."""
        claims = await decomposer.decompose("")

        assert claims == []

    @pytest.mark.asyncio
    async def test_decompose_whitespace_only(self, decomposer: LLMClaimDecomposer) -> None:
        """Test decomposing whitespace-only text returns empty list."""
        claims = await decomposer.decompose("   \n\t  ")

        assert claims == []

    @pytest.mark.asyncio
    async def test_decompose_valid_response(
        self,
        decomposer: LLMClaimDecomposer,
        mock_llm_provider: AsyncMock,
    ) -> None:
        """Test decomposing with valid LLM response."""
        # Set up mock response
        json_response = """[
            {
                "text": "Paris is the capital of France.",
                "subject": "Paris",
                "predicate": "is the capital of",
                "object": "France",
                "claim_type": "subject_predicate_object",
                "confidence": 0.95
            },
            {
                "text": "The Eiffel Tower is located in Paris.",
                "subject": "Eiffel Tower",
                "predicate": "is located in",
                "object": "Paris",
                "claim_type": "subject_predicate_object",
                "confidence": 0.9
            }
        ]"""

        mock_llm_provider.complete.return_value = LLMResponse(
            content=json_response,
            model="test-model",
            usage={"prompt_tokens": 100, "completion_tokens": 50},
        )

        text = "Paris is the capital of France. The Eiffel Tower is located in Paris."
        claims = await decomposer.decompose(text)

        assert len(claims) == 2
        assert claims[0].text == "Paris is the capital of France."
        assert claims[0].subject == "Paris"
        assert claims[0].object == "France"
        assert claims[0].claim_type == ClaimType.SUBJECT_PREDICATE_OBJECT
        assert claims[0].confidence == 0.95

    @pytest.mark.asyncio
    async def test_decompose_with_markdown_code_block(
        self,
        decomposer: LLMClaimDecomposer,
        mock_llm_provider: AsyncMock,
    ) -> None:
        """Test parsing response with markdown code block."""
        json_response = """Here are the claims:

```json
[
    {
        "text": "Water boils at 100 degrees Celsius.",
        "subject": "Water",
        "predicate": "boils at",
        "object": "100 degrees Celsius",
        "claim_type": "quantitative",
        "confidence": 0.98
    }
]
```
"""

        mock_llm_provider.complete.return_value = LLMResponse(
            content=json_response,
            model="test-model",
            usage={},
        )

        claims = await decomposer.decompose("Water boils at 100 degrees Celsius.")

        assert len(claims) == 1
        assert claims[0].claim_type == ClaimType.QUANTITATIVE

    @pytest.mark.asyncio
    async def test_decompose_with_context(
        self,
        decomposer: LLMClaimDecomposer,
        mock_llm_provider: AsyncMock,
    ) -> None:
        """Test decomposing with additional context."""
        mock_llm_provider.complete.return_value = LLMResponse(
            content="[]",
            model="test-model",
            usage={},
        )

        await decomposer.decompose_with_context(
            text="He founded the company in 2020.",
            context="Article about Elon Musk",
        )

        # Check that complete was called
        mock_llm_provider.complete.assert_called_once()
        call_args = mock_llm_provider.complete.call_args
        messages = call_args[0][0]

        # Context should be included in user message
        user_message = messages[-1].content
        assert "Elon Musk" in user_message

    @pytest.mark.asyncio
    async def test_decompose_max_claims_limit(
        self,
        decomposer: LLMClaimDecomposer,
        mock_llm_provider: AsyncMock,
    ) -> None:
        """Test that claims are limited to max_claims."""
        # Return more claims than max
        claims_json = [
            {
                "text": f"Claim {i}",
                "claim_type": "unclassified",
                "confidence": 0.8,
            }
            for i in range(100)  # More than default max of 50
        ]

        import json

        mock_llm_provider.complete.return_value = LLMResponse(
            content=json.dumps(claims_json),
            model="test-model",
            usage={},
        )

        claims = await decomposer.decompose("Long text with many claims...")

        assert len(claims) <= 50  # Should be limited

    @pytest.mark.asyncio
    async def test_decompose_llm_error(
        self,
        decomposer: LLMClaimDecomposer,
        mock_llm_provider: AsyncMock,
    ) -> None:
        """Test handling of LLM errors."""
        mock_llm_provider.complete.side_effect = Exception("LLM unavailable")

        with pytest.raises(DecompositionError) as exc_info:
            await decomposer.decompose("Test text")

        assert "Failed to decompose text" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_decompose_invalid_json_fallback(
        self,
        decomposer: LLMClaimDecomposer,
        mock_llm_provider: AsyncMock,
    ) -> None:
        """Test fallback when LLM returns invalid JSON."""
        mock_llm_provider.complete.return_value = LLMResponse(
            content="This is not valid JSON at all!",
            model="test-model",
            usage={},
        )

        text = "The sky is blue. Water is wet."
        claims = await decomposer.decompose(text)

        # Should use fallback sentence splitting
        assert len(claims) >= 1
        # Fallback claims have lower confidence
        assert all(c.confidence == 0.5 for c in claims)

    @pytest.mark.asyncio
    async def test_health_check(
        self,
        decomposer: LLMClaimDecomposer,
        mock_llm_provider: AsyncMock,
    ) -> None:
        """Test health check delegates to LLM provider."""
        result = await decomposer.health_check()

        assert result is True
        mock_llm_provider.health_check.assert_called_once()
