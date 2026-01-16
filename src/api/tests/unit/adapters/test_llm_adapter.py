"""Unit tests for OpenAI LLM adapter."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from adapters.openai import LLMProviderError, OpenAILLMAdapter
from interfaces.llm import LLMMessage, LLMResponse


@pytest.fixture
def mock_settings():
    """Mock LLM settings."""
    settings = MagicMock()
    settings.base_url = "http://localhost:8000/v1"
    settings.api_key = MagicMock()
    settings.api_key.get_secret_value.return_value = "test_key"
    settings.timeout_seconds = 30.0
    settings.max_retries = 2
    settings.model = "gpt-4"
    return settings


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_message = MagicMock()
    mock_message.content = "Test response content"
    mock_choice.message = mock_message
    mock_choice.finish_reason = "stop"
    mock_response.choices = [mock_choice]
    mock_response.model = "gpt-4"
    mock_response.usage = MagicMock()
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 5
    mock_response.usage.total_tokens = 15
    mock_response.model_dump.return_value = {"id": "test"}
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
    # Mock models.list for health check
    mock_client.models = MagicMock()
    mock_client.models.list = AsyncMock(return_value=MagicMock())
    return mock_client


@pytest.fixture
def llm_adapter(mock_settings, mock_openai_client):
    """LLM adapter with mocked client."""
    with patch("adapters.openai.AsyncOpenAI", return_value=mock_openai_client):
        with patch("adapters.openai.httpx.AsyncClient"):
            with patch("adapters.openai.httpx.AsyncHTTPTransport"):
                adapter = OpenAILLMAdapter(mock_settings)
                adapter._client = mock_openai_client
                return adapter


class TestOpenAILLMAdapter:
    """Test OpenAILLMAdapter."""

    def test_initialization(self, llm_adapter: OpenAILLMAdapter):
        """Test adapter initialization."""
        assert llm_adapter is not None
        assert llm_adapter.model_name == "gpt-4"

    @pytest.mark.asyncio
    async def test_complete(self, llm_adapter: OpenAILLMAdapter, mock_openai_client):
        """Test completing a message."""
        messages = [
            LLMMessage(role="user", content="Hello, world!")
        ]

        response = await llm_adapter.complete(messages)

        assert isinstance(response, LLMResponse)
        assert response.content == "Test response content"
        assert response.model == "gpt-4"
        mock_openai_client.chat.completions.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_complete_with_temperature(self, llm_adapter: OpenAILLMAdapter, mock_openai_client):
        """Test completing with custom temperature."""
        messages = [
            LLMMessage(role="system", content="You are helpful."),
            LLMMessage(role="user", content="Hello!")
        ]

        await llm_adapter.complete(messages, temperature=0.7)

        call_args = mock_openai_client.chat.completions.create.call_args
        assert call_args.kwargs["temperature"] == 0.7


class TestLLMAdapterErrorHandling:
    """Test error handling scenarios."""

    @pytest.mark.asyncio
    async def test_api_error(self, llm_adapter: OpenAILLMAdapter, mock_openai_client):
        """Test handling of API errors."""
        from openai import APIStatusError

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_openai_client.chat.completions.create.side_effect = APIStatusError(
            message="Server error",
            response=mock_response,
            body=None,
        )

        with pytest.raises(LLMProviderError):
            await llm_adapter.complete([LLMMessage(role="user", content="test")])

    @pytest.mark.asyncio
    async def test_health_check(self, llm_adapter: OpenAILLMAdapter, mock_openai_client):
        """Test health check."""
        result = await llm_adapter.health_check()

        # Health check should call the API and return True on success
        assert result is True
