"""Tests for caching functionality."""

import pytest
from pathlib import Path
import numpy as np

from azure_llm_toolkit import CacheManager, EmbeddingCache, ChatCache


@pytest.fixture
def embedding_cache(tmp_path):
    """Create EmbeddingCache in temporary directory."""
    return EmbeddingCache(cache_dir=tmp_path / "embeddings", enabled=True)


@pytest.fixture
def chat_cache(tmp_path):
    """Create ChatCache in temporary directory."""
    return ChatCache(cache_dir=tmp_path / "chat", enabled=True)


@pytest.fixture
def cache_manager(tmp_path):
    """Create CacheManager in temporary directory."""
    return CacheManager(cache_dir=tmp_path / "cache", enabled=True)


def test_embedding_cache_set_and_get(embedding_cache):
    """Test setting and getting cached embeddings."""
    text = "Test text for embedding"
    model = "text-embedding-3-large"
    embedding = np.array([1.0, 2.0, 3.0])

    # Set cache
    embedding_cache.set(text, model, embedding)

    # Get cache
    cached = embedding_cache.get(text, model)

    assert cached is not None
    assert np.allclose(cached, embedding)


def test_embedding_cache_miss(embedding_cache):
    """Test cache miss returns None."""
    text = "Non-existent text"
    model = "text-embedding-3-large"

    cached = embedding_cache.get(text, model)
    assert cached is None


def test_embedding_cache_different_models(embedding_cache):
    """Test that different models have separate cache entries."""
    text = "Same text"
    model1 = "text-embedding-3-large"
    model2 = "text-embedding-3-small"
    embedding1 = np.array([1.0, 2.0, 3.0])
    embedding2 = np.array([4.0, 5.0, 6.0])

    embedding_cache.set(text, model1, embedding1)
    embedding_cache.set(text, model2, embedding2)

    cached1 = embedding_cache.get(text, model1)
    cached2 = embedding_cache.get(text, model2)

    assert np.allclose(cached1, embedding1)
    assert np.allclose(cached2, embedding2)


def test_embedding_cache_batch(embedding_cache):
    """Test batch cache operations."""
    texts = ["Text 1", "Text 2", "Text 3"]
    model = "text-embedding-3-large"
    embeddings = [
        np.array([1.0, 2.0]),
        np.array([3.0, 4.0]),
        np.array([5.0, 6.0]),
    ]

    # Cache first two texts
    embedding_cache.set_batch(texts[:2], model, embeddings[:2])

    # Get batch - should have partial hits
    cached_embeddings, missing_indices = embedding_cache.get_batch(texts, model)

    assert len(cached_embeddings) == 3
    assert cached_embeddings[0] is not None
    assert cached_embeddings[1] is not None
    assert cached_embeddings[2] is None
    assert missing_indices == [2]


def test_chat_cache_set_and_get(chat_cache):
    """Test setting and getting cached chat responses."""
    messages = [{"role": "user", "content": "Hello"}]
    model = "gpt-4o"
    response = {
        "content": "Hi there!",
        "usage": {"total_tokens": 10},
        "model": model,
    }

    # Set cache
    chat_cache.set(messages, model, response)

    # Get cache
    cached = chat_cache.get(messages, model)

    assert cached is not None
    assert cached["content"] == response["content"]
    assert cached["usage"] == response["usage"]


def test_chat_cache_miss(chat_cache):
    """Test chat cache miss returns None."""
    messages = [{"role": "user", "content": "Non-existent"}]
    model = "gpt-4o"

    cached = chat_cache.get(messages, model)
    assert cached is None


def test_chat_cache_different_parameters(chat_cache):
    """Test that different parameters create separate cache entries."""
    messages = [{"role": "user", "content": "Hello"}]
    model = "gpt-4o"
    response1 = {"content": "Response 1"}
    response2 = {"content": "Response 2"}

    # Cache with different temperatures
    chat_cache.set(messages, model, response1, temperature=0.7)
    chat_cache.set(messages, model, response2, temperature=0.0)

    cached1 = chat_cache.get(messages, model, temperature=0.7)
    cached2 = chat_cache.get(messages, model, temperature=0.0)

    assert cached1["content"] == "Response 1"
    assert cached2["content"] == "Response 2"


def test_cache_clear(embedding_cache):
    """Test clearing cache."""
    text = "Test text"
    model = "text-embedding-3-large"
    embedding = np.array([1.0, 2.0, 3.0])

    embedding_cache.set(text, model, embedding)
    assert embedding_cache.get(text, model) is not None

    # Clear cache
    count = embedding_cache.clear()
    assert count > 0

    # Verify cache is empty
    assert embedding_cache.get(text, model) is None


def test_cache_stats(embedding_cache):
    """Test cache statistics."""
    texts = ["Text 1", "Text 2", "Text 3"]
    model = "text-embedding-3-large"
    embeddings = [np.array([float(i), float(i + 1)]) for i in range(3)]

    # Add some cache entries
    embedding_cache.set_batch(texts, model, embeddings)

    stats = embedding_cache.get_stats()

    assert stats["enabled"] is True
    assert stats["file_count"] == 3
    assert stats["size_bytes"] > 0
    assert stats["size_mb"] > 0


def test_cache_manager_embedding(cache_manager):
    """Test CacheManager embedding cache."""
    text = "Test text"
    model = "text-embedding-3-large"
    embedding = np.array([1.0, 2.0, 3.0])

    cache_manager.embedding_cache.set(text, model, embedding)
    cached = cache_manager.embedding_cache.get(text, model)

    assert cached is not None
    assert np.allclose(cached, embedding)


def test_cache_manager_chat(cache_manager):
    """Test CacheManager chat cache."""
    messages = [{"role": "user", "content": "Hello"}]
    model = "gpt-4o"
    response = {"content": "Hi!"}

    cache_manager.chat_cache.set(messages, model, response)
    cached = cache_manager.chat_cache.get(messages, model)

    assert cached is not None
    assert cached["content"] == "Hi!"


def test_cache_manager_clear_all(cache_manager):
    """Test clearing all caches."""
    # Add some data
    cache_manager.embedding_cache.set("text", "model", np.array([1.0]))
    cache_manager.chat_cache.set([{"role": "user", "content": "hi"}], "model", {"content": "hello"})

    # Clear all
    result = cache_manager.clear_all()

    assert "embeddings" in result
    assert "chat" in result
    assert result["embeddings"] > 0 or result["chat"] > 0


def test_cache_manager_stats(cache_manager):
    """Test CacheManager statistics."""
    # Add some data
    cache_manager.embedding_cache.set("text", "model", np.array([1.0, 2.0]))
    cache_manager.chat_cache.set([{"role": "user", "content": "hi"}], "model", {"content": "hello"})

    stats = cache_manager.get_stats()

    assert "total_size_bytes" in stats
    assert "total_size_mb" in stats
    assert "total_files" in stats
    assert "embeddings" in stats
    assert "chat" in stats
    assert stats["total_files"] >= 2


def test_cache_disabled(tmp_path):
    """Test that caching can be disabled."""
    cache = EmbeddingCache(cache_dir=tmp_path / "cache", enabled=False)

    text = "Test text"
    model = "text-embedding-3-large"
    embedding = np.array([1.0, 2.0, 3.0])

    # Set should not cache
    cache.set(text, model, embedding)

    # Get should return None
    cached = cache.get(text, model)
    assert cached is None

    # Stats should show disabled
    stats = cache.get_stats()
    assert stats["enabled"] is False


def test_cache_custom_directory(tmp_path):
    """Test using custom cache directory."""
    custom_dir = tmp_path / "my_custom_cache"
    cache = EmbeddingCache(cache_dir=custom_dir, enabled=True)

    text = "Test"
    model = "model"
    embedding = np.array([1.0])

    cache.set(text, model, embedding)

    # Verify file was created in custom directory
    assert custom_dir.exists()
    assert len(list(custom_dir.iterdir())) > 0


def test_embedding_cache_overwrites(embedding_cache):
    """Test that setting the same key overwrites previous value."""
    text = "Test text"
    model = "model"
    embedding1 = np.array([1.0, 2.0, 3.0])
    embedding2 = np.array([4.0, 5.0, 6.0])

    embedding_cache.set(text, model, embedding1)
    embedding_cache.set(text, model, embedding2)

    cached = embedding_cache.get(text, model)
    assert np.allclose(cached, embedding2)


def test_chat_cache_different_messages(chat_cache):
    """Test that different messages create separate cache entries."""
    model = "gpt-4o"
    messages1 = [{"role": "user", "content": "Hello"}]
    messages2 = [{"role": "user", "content": "Goodbye"}]
    response1 = {"content": "Hi!"}
    response2 = {"content": "Bye!"}

    chat_cache.set(messages1, model, response1)
    chat_cache.set(messages2, model, response2)

    cached1 = chat_cache.get(messages1, model)
    cached2 = chat_cache.get(messages2, model)

    assert cached1["content"] == "Hi!"
    assert cached2["content"] == "Bye!"
