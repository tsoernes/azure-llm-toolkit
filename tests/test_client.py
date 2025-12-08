"""Tests for AzureLLMClient."""

import pytest

from azure_llm_toolkit import AzureLLMClient


@pytest.mark.asyncio
async def test_embed_text(client, sample_texts):
    """Test single text embedding."""
    text = sample_texts[0]
    embedding = await client.embed_text(text)

    assert isinstance(embedding, list)
    assert len(embedding) > 0
    assert all(isinstance(x, float) for x in embedding)


@pytest.mark.asyncio
async def test_embed_text_with_cache(client_with_cache, sample_texts):
    """Test single text embedding with caching."""
    text = sample_texts[0]

    # First call - cache miss
    embedding1 = await client_with_cache.embed_text(text, use_cache=True)
    assert isinstance(embedding1, list)

    # Second call - cache hit
    embedding2 = await client_with_cache.embed_text(text, use_cache=True)
    assert embedding1 == embedding2

    # Verify cache stats
    stats = client_with_cache.cache_manager.get_stats()
    assert stats["embeddings"]["file_count"] > 0


@pytest.mark.asyncio
async def test_chat_completion(client, sample_messages):
    """Test chat completion."""
    result = await client.chat_completion(
        messages=sample_messages,
        system_prompt="You are a helpful assistant.",
    )

    assert result.content is not None
    assert len(result.content) > 0
    assert result.usage.total_tokens > 0
    assert result.model is not None


@pytest.mark.asyncio
async def test_chat_completion_with_cache(client_with_cache, sample_messages):
    """Test chat completion with caching."""
    # First call - cache miss
    result1 = await client_with_cache.chat_completion(
        messages=sample_messages,
        use_cache=True,
    )
    assert result1.content is not None

    # Second call - cache hit
    result2 = await client_with_cache.chat_completion(
        messages=sample_messages,
        use_cache=True,
    )
    assert result1.content == result2.content

    # Verify cache stats
    stats = client_with_cache.cache_manager.get_stats()
    assert stats["chat"]["file_count"] > 0


@pytest.mark.asyncio
async def test_chat_completion_with_max_tokens(client):
    """Test chat completion with max_tokens parameter."""
    result = await client.chat_completion(
        messages=[{"role": "user", "content": "Say hello"}],
        max_tokens=10,
    )

    assert result.content is not None
    assert result.usage.completion_tokens <= 10


def test_count_tokens(client):
    """Test token counting."""
    text = "This is a test sentence."
    count = client.count_tokens(text)

    assert isinstance(count, int)
    assert count > 0


def test_count_message_tokens(client, sample_messages):
    """Test message token counting."""
    count = client.count_message_tokens(sample_messages)

    assert isinstance(count, int)
    assert count > 0


def test_estimate_embedding_cost(client):
    """Test embedding cost estimation."""
    text = "Sample text for cost estimation"
    cost = client.estimate_embedding_cost(text)

    assert isinstance(cost, float)
    assert cost >= 0


def test_estimate_chat_cost(client, sample_messages):
    """Test chat cost estimation."""
    cost = client.estimate_chat_cost(sample_messages)

    assert isinstance(cost, float)
    assert cost >= 0


@pytest.mark.asyncio
async def test_cache_disabled(client, sample_texts):
    """Test that caching can be disabled."""
    text = sample_texts[0]

    # Client has cache disabled
    assert client.enable_cache is False

    # Even with use_cache=True, it should not cache
    embedding1 = await client.embed_text(text, use_cache=True)
    embedding2 = await client.embed_text(text, use_cache=True)

    # Both should work, but no cache manager
    assert isinstance(embedding1, list)
    assert isinstance(embedding2, list)
