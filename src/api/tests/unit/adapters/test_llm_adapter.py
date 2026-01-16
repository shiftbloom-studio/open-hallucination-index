"""Unit tests for LLM adapter."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from open_hallucination_index.adapters.llm_client import LLMClient
from open_hallucination_index.domain.entities import Claim


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_message = MagicMock()
    mock_message.content = '["Python was created in 1991", "Python emphasizes readability"]'
    mock_choice.message = mock_message
    mock_response.choices = [mock_choice]
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
    return mock_client


@pytest.fixture
def llm_client(mock_openai_client):
    """LLM client with mocked OpenAI client."""
    with patch("openai.AsyncOpenAI", return_value=mock_openai_client):
        client = LLMClient(
            api_key="test_key",
            model="gpt-4",
            base_url="http://localhost:8000/v1",
        )
        client.client = mock_openai_client
        return client


class TestLLMClient:
    """Test LLMClient adapter."""

    def test_initialization(self, llm_client: LLMClient):
        """Test client initialization."""
        assert llm_client is not None
        assert llm_client.client is not None

    @pytest.mark.asyncio
    async def test_decompose_claims_basic(self, llm_client: LLMClient):
        """Test claim decomposition."""
        text = "Python was created in 1991 and emphasizes readability."
        
        claims = await llm_client.decompose_claims(text)
        
        assert len(claims) > 0
        assert all(isinstance(claim, Claim) for claim in claims)

    @pytest.mark.asyncio
    async def test_decompose_claims_single(self, llm_client: LLMClient):
        """Test decomposing single claim."""
        text = "Python is a programming language."
        
        # Mock single claim response
        mock_message = MagicMock()
        mock_message.content = '["Python is a programming language"]'
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        llm_client.client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        claims = await llm_client.decompose_claims(text)
        
        assert len(claims) == 1

    @pytest.mark.asyncio
    async def test_decompose_claims_empty(self, llm_client: LLMClient):
        """Test decomposing empty text."""
        text = ""
        
        claims = await llm_client.decompose_claims(text)
        
        # Should handle empty input gracefully
        assert isinstance(claims, list)

    @pytest.mark.asyncio
    async def test_generate_response(self, llm_client: LLMClient):
        """Test generating LLM response."""
        prompt = "What is Python?"
        
        mock_message = MagicMock()
        mock_message.content = "Python is a programming language."
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        llm_client.client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        response = await llm_client.generate_response(prompt)
        
        assert response is not None
        assert isinstance(response, str)


class TestLLMErrorHandling:
    """Test error handling scenarios."""

    @pytest.mark.asyncio
    async def test_api_error_handling(self, llm_client: LLMClient):
        """Test handling of API errors."""
        llm_client.client.chat.completions.create = AsyncMock(
            side_effect=Exception("API error")
        )
        
        with pytest.raises(Exception):
            await llm_client.decompose_claims("test text")

    @pytest.mark.asyncio
    async def test_invalid_json_response(self, llm_client: LLMClient):
        """Test handling invalid JSON response."""
        mock_message = MagicMock()
        mock_message.content = "Invalid JSON"
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        llm_client.client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        # Should handle invalid JSON gracefully
        claims = await llm_client.decompose_claims("test text")
        assert isinstance(claims, list)

    @pytest.mark.asyncio
    async def test_timeout_handling(self, llm_client: LLMClient):
        """Test handling of timeout errors."""
        llm_client.client.chat.completions.create = AsyncMock(
            side_effect=TimeoutError("Request timeout")
        )
        
        with pytest.raises(TimeoutError):
            await llm_client.decompose_claims("test text")
