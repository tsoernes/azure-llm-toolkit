"""Tests for GPT-5 parameter conversion."""

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from azure_llm_toolkit.client import AzureLLMClient, _prepare_gpt5_kwargs
from azure_llm_toolkit.config import AzureConfig


class TestGPT5ParameterConversion:
    """Test automatic parameter conversion for GPT-5 models."""

    def test_prepare_gpt5_kwargs_converts_max_tokens(self, caplog):
        """Test that max_tokens is converted to max_completion_tokens for GPT-5 models."""
        kwargs = {
            "model": "gpt-5-mini",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 100,
        }

        with caplog.at_level(logging.WARNING):
            result = _prepare_gpt5_kwargs(kwargs, "gpt-5-mini")

        assert "max_tokens" not in result
        assert result["max_completion_tokens"] == 100
        assert "Converted max_tokens=100 to max_completion_tokens=100" in caplog.text

    def test_prepare_gpt5_kwargs_removes_temperature(self, caplog):
        """Test that temperature is removed for GPT-5 models."""
        kwargs = {
            "model": "gpt-5",
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 0.7,
        }

        with caplog.at_level(logging.WARNING):
            result = _prepare_gpt5_kwargs(kwargs, "gpt-5")

        assert "temperature" not in result
        assert "Removed temperature=0.7 parameter" in caplog.text

    def test_prepare_gpt5_kwargs_converts_both_parameters(self, caplog):
        """Test that both max_tokens and temperature are handled for GPT-5 models."""
        kwargs = {
            "model": "gpt-5-turbo",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 200,
            "temperature": 0.5,
        }

        with caplog.at_level(logging.WARNING):
            result = _prepare_gpt5_kwargs(kwargs, "gpt-5-turbo")

        assert "max_tokens" not in result
        assert "temperature" not in result
        assert result["max_completion_tokens"] == 200
        assert "Converted max_tokens=200" in caplog.text
        assert "Removed temperature=0.5" in caplog.text

    def test_prepare_gpt5_kwargs_preserves_other_parameters(self):
        """Test that other parameters are preserved during conversion."""
        kwargs = {
            "model": "gpt-5",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 150,
            "temperature": 0.8,
            "reasoning_effort": "high",
            "response_format": {"type": "json_object"},
        }

        result = _prepare_gpt5_kwargs(kwargs, "gpt-5")

        assert result["model"] == "gpt-5"
        assert result["messages"] == [{"role": "user", "content": "Hello"}]
        assert result["reasoning_effort"] == "high"
        assert result["response_format"] == {"type": "json_object"}
        assert result["max_completion_tokens"] == 150

    def test_prepare_gpt5_kwargs_case_insensitive(self, caplog):
        """Test that GPT-5 detection is case-insensitive."""
        kwargs = {
            "model": "GPT-5-MINI",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 100,
        }

        with caplog.at_level(logging.WARNING):
            result = _prepare_gpt5_kwargs(kwargs, "GPT-5-MINI")

        assert "max_completion_tokens" in result
        assert "max_tokens" not in result

    def test_prepare_gpt5_kwargs_no_conversion_for_non_gpt5(self, caplog):
        """Test that non-GPT-5 models are not affected."""
        kwargs = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 100,
            "temperature": 0.7,
        }

        with caplog.at_level(logging.WARNING):
            result = _prepare_gpt5_kwargs(kwargs, "gpt-4o")

        assert result["max_tokens"] == 100
        assert result["temperature"] == 0.7
        assert "max_completion_tokens" not in result
        assert len(caplog.records) == 0

    def test_prepare_gpt5_kwargs_handles_gpt4_and_gpt35(self):
        """Test that GPT-4 and GPT-3.5 models are not affected."""
        for model in ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo", "gpt-35-turbo"]:
            kwargs = {
                "model": model,
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 100,
                "temperature": 0.7,
            }

            result = _prepare_gpt5_kwargs(kwargs, model)

            assert result["max_tokens"] == 100
            assert result["temperature"] == 0.7
            assert "max_completion_tokens" not in result

    def test_prepare_gpt5_kwargs_no_max_tokens(self):
        """Test conversion when max_tokens is not provided."""
        kwargs = {
            "model": "gpt-5",
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 0.7,
        }

        result = _prepare_gpt5_kwargs(kwargs, "gpt-5")

        assert "max_tokens" not in result
        assert "max_completion_tokens" not in result
        assert "temperature" not in result

    def test_prepare_gpt5_kwargs_no_temperature(self):
        """Test conversion when temperature is not provided."""
        kwargs = {
            "model": "gpt-5",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 100,
        }

        result = _prepare_gpt5_kwargs(kwargs, "gpt-5")

        assert "max_tokens" not in result
        assert result["max_completion_tokens"] == 100

    def test_prepare_gpt5_kwargs_does_not_modify_original(self):
        """Test that the original kwargs dict is not modified."""
        original_kwargs = {
            "model": "gpt-5",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 100,
            "temperature": 0.7,
        }

        # Make a copy to compare
        kwargs_copy = original_kwargs.copy()

        _prepare_gpt5_kwargs(original_kwargs, "gpt-5")

        # Original should be unchanged
        assert original_kwargs == kwargs_copy

    @pytest.mark.asyncio
    async def test_chat_completion_applies_gpt5_conversion(self, caplog):
        """Test that chat_completion applies GPT-5 conversion automatically."""
        config = AzureConfig(
            api_key="test-key",
            endpoint="https://test.openai.azure.com",
            chat_deployment="gpt-5-mini",
        )

        # Mock the OpenAI client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Hello!"))]
        mock_response.usage = MagicMock(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            completion_tokens_details=MagicMock(reasoning_tokens=0),
        )
        mock_response.model = "gpt-5-mini"

        with patch("azure_llm_toolkit.client.AsyncAzureOpenAI") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client

            client = AzureLLMClient(
                config=config,
                enable_rate_limiting=False,
                enable_cache=False,
            )
            client.client = mock_client

            with caplog.at_level(logging.WARNING):
                result = await client.chat_completion(
                    messages=[{"role": "user", "content": "Hello"}],
                    max_tokens=100,
                    temperature=0.7,
                )

            # Verify the conversion warnings were logged
            assert "Converted max_tokens=100 to max_completion_tokens=100" in caplog.text
            assert "Removed temperature=0.7 parameter" in caplog.text

            # Verify the API was called with converted parameters
            call_kwargs = mock_client.chat.completions.create.call_args[1]
            assert "max_completion_tokens" in call_kwargs
            assert call_kwargs["max_completion_tokens"] == 100
            assert "temperature" not in call_kwargs
            assert "max_tokens" not in call_kwargs

    @pytest.mark.asyncio
    async def test_chat_completion_stream_applies_gpt5_conversion(self, caplog):
        """Test that chat_completion_stream applies GPT-5 conversion automatically."""
        config = AzureConfig(
            api_key="test-key",
            endpoint="https://test.openai.azure.com",
            chat_deployment="gpt-5-turbo",
        )

        # Mock streaming response
        mock_chunk = MagicMock()
        mock_chunk.choices = [MagicMock(delta=MagicMock(content="Hello"))]

        async def mock_stream():
            yield mock_chunk

        mock_stream_context = AsyncMock()
        mock_stream_context.__aenter__ = AsyncMock(return_value=mock_stream())
        mock_stream_context.__aexit__ = AsyncMock(return_value=None)

        with patch("azure_llm_toolkit.client.AsyncAzureOpenAI") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.chat.completions.stream = MagicMock(return_value=mock_stream_context)
            mock_client_class.return_value = mock_client

            client = AzureLLMClient(
                config=config,
                enable_rate_limiting=False,
                enable_cache=False,
            )
            client.client = mock_client

            with caplog.at_level(logging.WARNING):
                chunks = []
                async for chunk in client.chat_completion_stream(
                    messages=[{"role": "user", "content": "Hello"}],
                    max_tokens=150,
                    temperature=0.5,
                ):
                    chunks.append(chunk)

            # Verify the conversion warnings were logged
            assert "Converted max_tokens=150 to max_completion_tokens=150" in caplog.text
            assert "Removed temperature=0.5 parameter" in caplog.text

            # Verify the API was called with converted parameters
            call_kwargs = mock_client.chat.completions.stream.call_args[1]
            assert "max_completion_tokens" in call_kwargs
            assert call_kwargs["max_completion_tokens"] == 150
            assert "temperature" not in call_kwargs
            assert "max_tokens" not in call_kwargs
