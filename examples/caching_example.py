"""Example of disk-based caching for LLM calls to save costs and improve performance.

This example demonstrates how to use the caching functionality to avoid
redundant API calls for embeddings and chat completions.
"""

import asyncio
from pathlib import Path

from dotenv import load_dotenv

from azure_llm_toolkit import AzureConfig, AzureLLMClient, CacheManager

# Load environment variables
load_dotenv()


async def basic_embedding_cache():
    """Basic example: Cache embeddings to avoid redundant API calls."""
    print("\n=== Basic Embedding Cache ===")

    config = AzureConfig()

    # Create client with caching enabled (default)
    client = AzureLLMClient(config=config, enable_cache=True)

    texts = ["Hello world", "Machine learning is powerful", "Azure OpenAI"]

    # First call - hits the API
    print("First call (cache miss)...")
    result1 = await client.embed_texts(texts, use_cache=True)
    print(f"Generated {len(result1.embeddings)} embeddings")
    print(f"Total tokens: {result1.usage.total_tokens}")

    # Second call - retrieves from cache
    print("\nSecond call (cache hit)...")
    result2 = await client.embed_texts(texts, use_cache=True)
    print(f"Retrieved {len(result2.embeddings)} embeddings from cache")
    print(f"Total tokens: {result2.usage.total_tokens}")

    # Verify embeddings are identical
    import numpy as np

    for i, (emb1, emb2) in enumerate(zip(result1.embeddings, result2.embeddings)):
        assert np.allclose(emb1, emb2), f"Embeddings differ at index {i}"
    print("\n✓ Cached embeddings match original embeddings")


async def basic_chat_cache():
    """Basic example: Cache chat completions."""
    print("\n=== Basic Chat Cache ===")

    config = AzureConfig()
    client = AzureLLMClient(config=config, enable_cache=True)

    messages = [{"role": "user", "content": "What is 2+2?"}]

    # First call - hits the API
    print("First call (cache miss)...")
    result1 = await client.chat_completion(messages, use_cache=True)
    print(f"Response: {result1.content}")
    print(f"Tokens: {result1.usage.total_tokens}")

    # Second call - retrieves from cache
    print("\nSecond call (cache hit)...")
    result2 = await client.chat_completion(messages, use_cache=True)
    print(f"Response: {result2.content}")
    print(f"Tokens: {result2.usage.total_tokens}")

    # Verify responses are identical
    assert result1.content == result2.content, "Cached response differs from original"
    print("\n✓ Cached response matches original response")


async def partial_cache_hits():
    """Example: Partial cache hits when some texts are cached and others aren't."""
    print("\n=== Partial Cache Hits ===")

    config = AzureConfig()
    client = AzureLLMClient(config=config, enable_cache=True)

    # First batch - all cache misses
    texts1 = ["Document 1", "Document 2", "Document 3"]
    print(f"Embedding first batch ({len(texts1)} texts)...")
    result1 = await client.embed_texts(texts1, use_cache=True)
    print(f"Cache misses: {len(texts1)}")

    # Second batch - partial overlap
    texts2 = ["Document 2", "Document 3", "Document 4", "Document 5"]
    print(f"\nEmbedding second batch ({len(texts2)} texts)...")
    result2 = await client.embed_texts(texts2, use_cache=True)
    print(f"Cache hits: 2, Cache misses: 2")
    print(f"Total embeddings: {len(result2.embeddings)}")


async def cache_statistics():
    """Example: Get cache statistics."""
    print("\n=== Cache Statistics ===")

    config = AzureConfig()

    # Create cache manager with custom directory
    cache_manager = CacheManager(cache_dir=".llm_cache_example")
    client = AzureLLMClient(config=config, cache_manager=cache_manager)

    # Generate some cached data
    texts = [f"Sample text {i}" for i in range(20)]
    await client.embed_texts(texts, use_cache=True)

    messages_list = [[{"role": "user", "content": f"Question {i}?"}] for i in range(10)]
    for messages in messages_list:
        await client.chat_completion(messages, use_cache=True)

    # Get cache statistics
    stats = cache_manager.get_stats()
    print("\nCache Statistics:")
    print(f"Total size: {stats['total_size_mb']:.2f} MB")
    print(f"Total files: {stats['total_files']}")
    print(f"\nEmbedding cache:")
    print(f"  Files: {stats['embeddings']['file_count']}")
    print(f"  Size: {stats['embeddings']['size_mb']:.2f} MB")
    print(f"\nChat cache:")
    print(f"  Files: {stats['chat']['file_count']}")
    print(f"  Size: {stats['chat']['size_mb']:.2f} MB")


async def selective_caching():
    """Example: Selectively enable/disable caching per call."""
    print("\n=== Selective Caching ===")

    config = AzureConfig()
    client = AzureLLMClient(config=config, enable_cache=True)

    text = "Sample text for embedding"

    # First call without cache
    print("First call (cache disabled)...")
    result1 = await client.embed_texts([text], use_cache=False)
    print(f"Generated embedding (not cached)")

    # Second call with cache enabled - still a cache miss
    print("\nSecond call (cache enabled but still miss)...")
    result2 = await client.embed_texts([text], use_cache=True)
    print(f"Generated embedding (now cached)")

    # Third call - cache hit
    print("\nThird call (cache hit)...")
    result3 = await client.embed_texts([text], use_cache=True)
    print(f"Retrieved from cache")


async def cache_with_parameters():
    """Example: Cache is sensitive to model parameters."""
    print("\n=== Cache with Different Parameters ===")

    config = AzureConfig()
    client = AzureLLMClient(config=config, enable_cache=True)

    messages = [{"role": "user", "content": "Tell me a fact"}]

    # Same messages but different parameters = different cache entries
    print("Call 1 (temperature=0.7)...")
    result1 = await client.chat_completion(messages, temperature=0.7, use_cache=True)
    print(f"Response: {result1.content[:50]}...")

    print("\nCall 2 (temperature=0.0) - different cache entry...")
    result2 = await client.chat_completion(messages, temperature=0.0, use_cache=True)
    print(f"Response: {result2.content[:50]}...")

    print("\nCall 3 (temperature=0.7) - cache hit from call 1...")
    result3 = await client.chat_completion(messages, temperature=0.7, use_cache=True)
    print(f"Response: {result3.content[:50]}...")

    assert result1.content == result3.content, "Should match cached result"
    print("\n✓ Cache correctly handles different parameters")


async def clear_cache():
    """Example: Clear cache."""
    print("\n=== Clear Cache ===")

    config = AzureConfig()
    cache_manager = CacheManager(cache_dir=".llm_cache_example")
    client = AzureLLMClient(config=config, cache_manager=cache_manager)

    # Add some cached data
    texts = ["Text 1", "Text 2", "Text 3"]
    await client.embed_texts(texts, use_cache=True)

    # Check cache stats
    stats_before = cache_manager.get_stats()
    print(f"Cache files before clear: {stats_before['total_files']}")

    # Clear cache
    cleared = cache_manager.clear_all()
    print(f"\nCleared cache:")
    print(f"  Embeddings: {cleared['embeddings']} files")
    print(f"  Chat: {cleared['chat']} files")

    # Check stats after clear
    stats_after = cache_manager.get_stats()
    print(f"\nCache files after clear: {stats_after['total_files']}")

    # Verify cache is empty
    result = await client.embed_texts(texts, use_cache=True)
    print(f"\n✓ Cache cleared - next call hits API again")


async def custom_cache_directory():
    """Example: Use custom cache directory."""
    print("\n=== Custom Cache Directory ===")

    config = AzureConfig()

    # Create cache in custom location
    custom_dir = Path("./my_custom_cache")
    cache_manager = CacheManager(cache_dir=custom_dir)
    client = AzureLLMClient(config=config, cache_manager=cache_manager)

    # Use the cache
    texts = ["Cached text 1", "Cached text 2"]
    await client.embed_texts(texts, use_cache=True)

    print(f"Cache directory: {custom_dir.absolute()}")
    print(f"Embedding cache: {custom_dir / 'embeddings'}")
    print(f"Chat cache: {custom_dir / 'chat'}")

    stats = cache_manager.get_stats()
    print(f"\nCached {stats['total_files']} files in custom directory")


async def disable_caching():
    """Example: Completely disable caching."""
    print("\n=== Disable Caching ===")

    config = AzureConfig()

    # Create client with caching disabled
    client = AzureLLMClient(config=config, enable_cache=False)

    texts = ["Text A", "Text B"]

    # Call multiple times - no caching
    print("Call 1 (no cache)...")
    await client.embed_texts(texts, use_cache=True)  # use_cache is ignored

    print("Call 2 (no cache)...")
    await client.embed_texts(texts, use_cache=True)  # Still hits API

    print("\n✓ Caching disabled - all calls hit API")


async def cache_cost_savings():
    """Example: Demonstrate cost savings from caching."""
    print("\n=== Cache Cost Savings ===")

    config = AzureConfig()
    cache_manager = CacheManager(cache_dir=".llm_cache_example")
    client = AzureLLMClient(config=config, cache_manager=cache_manager)

    # Clear cache first
    cache_manager.clear_all()

    # Large batch of texts
    texts = [f"Document {i} with some content to embed" for i in range(100)]

    # First pass - all cache misses
    print("First pass (all cache misses)...")
    result1 = await client.embed_texts(texts, use_cache=True)
    tokens_first = result1.usage.total_tokens
    cost_first = client.cost_estimator.estimate_cost(
        model=config.embedding_deployment,
        tokens_input=tokens_first,
    )
    print(f"Tokens: {tokens_first:,}")
    print(f"Cost: {cost_first:.4f} kr")

    # Second pass - all cache hits (no API calls, no cost!)
    print("\nSecond pass (all cache hits)...")
    result2 = await client.embed_texts(texts, use_cache=True)
    print(f"Tokens: 0 (from cache)")
    print(f"Cost: 0.0000 kr (from cache)")

    # Third pass with some new texts
    texts_mixed = texts[:50] + [f"New document {i}" for i in range(50)]
    print(f"\nThird pass (50% cache hits)...")
    result3 = await client.embed_texts(texts_mixed, use_cache=True)
    # Only the new 50 texts cost money
    tokens_third = sum(client.count_tokens(t) for t in texts_mixed[50:])
    cost_third = client.cost_estimator.estimate_cost(
        model=config.embedding_deployment,
        tokens_input=tokens_third,
    )
    print(f"Tokens: {tokens_third:,} (50 new texts)")
    print(f"Cost: {cost_third:.4f} kr")

    print(f"\n✓ Total cost savings: {cost_first:.4f} kr")


async def main():
    """Run all caching examples."""
    print("Azure LLM Toolkit - Caching Examples")
    print("=" * 60)

    try:
        await basic_embedding_cache()
        await basic_chat_cache()
        await partial_cache_hits()
        await cache_statistics()
        await selective_caching()
        await cache_with_parameters()
        await clear_cache()
        await custom_cache_directory()
        await disable_caching()
        await cache_cost_savings()

        print("\n" + "=" * 60)
        print("All caching examples completed successfully!")
        print("\nNote: Caching can significantly reduce costs for repeated queries.")
        print("Use cache_manager.clear_all() to clear cache when needed.")

    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
