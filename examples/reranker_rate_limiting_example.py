"""
Example demonstrating rate limiting in the logprob-based reranker.

This example shows how the reranker handles parallel document scoring with
built-in rate limiting to prevent hitting Azure OpenAI quotas.
"""

import asyncio
import time
from dotenv import load_dotenv

from azure_llm_toolkit import AzureConfig, AzureLLMClient, RateLimiter
from azure_llm_toolkit.reranker import LogprobReranker, create_reranker


async def basic_rate_limiting():
    """Basic rate limiting with default settings."""
    print("=" * 80)
    print("Basic Rate Limiting Example")
    print("=" * 80)

    load_dotenv()

    config = AzureConfig()
    client = AzureLLMClient(config=config)

    # Create reranker with default rate limits (2700 RPM, 450k TPM)
    reranker = LogprobReranker(client=client)

    print(f"\nDefault Rate Limits:")
    print(f"  RPM: {reranker.rate_limiter.rpm_limit:,}")
    print(f"  TPM: {reranker.rate_limiter.tpm_limit:,}")

    query = "What is machine learning?"
    documents = [
        "Machine learning is a subset of AI that learns from data.",
        "Python is a programming language.",
        "Deep learning uses neural networks.",
        "Cloud computing provides on-demand resources.",
        "Supervised learning uses labeled training data.",
    ]

    print(f"\nScoring {len(documents)} documents in parallel...")
    start_time = time.time()

    results = await reranker.rerank(query, documents)

    elapsed = time.time() - start_time

    print(f"\nCompleted in {elapsed:.2f} seconds")
    print("\nTop 3 Results:")
    for i, result in enumerate(results[:3], 1):
        print(f"{i}. Score: {result.score:.3f} - {result.document[:50]}...")

    # Show rate limiter stats
    stats = reranker.rate_limiter.get_stats()
    print(f"\nRate Limiter Stats:")
    print(f"  Total requests: {stats['total_requests']}")
    print(f"  Total tokens: {stats['total_tokens']:,.0f}")
    print(f"  Total wait time: {stats['total_wait_time_seconds']:.3f}s")
    print(f"  RPM utilization: {stats['rpm_utilization_pct']:.1f}%")
    print(f"  TPM utilization: {stats['tpm_utilization_pct']:.1f}%")


async def custom_rate_limits():
    """Example with custom rate limits."""
    print("\n" + "=" * 80)
    print("Custom Rate Limits Example")
    print("=" * 80)

    load_dotenv()

    config = AzureConfig()
    client = AzureLLMClient(config=config)

    # Create reranker with higher limits for production use
    reranker = create_reranker(
        client=client,
        model="gpt-4o-east-US",
        rpm_limit=3000,
        tpm_limit=500000,
    )

    print(f"\nCustom Rate Limits:")
    print(f"  RPM: {reranker.rate_limiter.rpm_limit:,}")
    print(f"  TPM: {reranker.rate_limiter.tpm_limit:,}")

    query = "How do neural networks work?"
    documents = [
        f"Document {i}: Neural networks are computing systems inspired by biological neural networks."
        for i in range(10)
    ]

    print(f"\nScoring {len(documents)} documents...")
    start_time = time.time()

    results = await reranker.rerank(query, documents)

    elapsed = time.time() - start_time
    print(f"Completed in {elapsed:.2f} seconds")

    stats = reranker.rate_limiter.get_stats()
    print(f"\nRate Limiter Stats:")
    print(f"  Requests: {stats['total_requests']}")
    print(f"  Tokens: {stats['total_tokens']:,.0f}")
    print(f"  Wait time: {stats['total_wait_time_seconds']:.3f}s")


async def shared_rate_limiter():
    """Example using shared rate limiter across multiple rerankers."""
    print("\n" + "=" * 80)
    print("Shared Rate Limiter Example")
    print("=" * 80)

    load_dotenv()

    config = AzureConfig()
    client = AzureLLMClient(config=config)

    # Create shared rate limiter
    shared_limiter = RateLimiter(rpm_limit=5000, tpm_limit=600000)

    print(f"\nShared Rate Limiter:")
    print(f"  RPM: {shared_limiter.rpm_limit:,}")
    print(f"  TPM: {shared_limiter.tpm_limit:,}")

    # Create two rerankers sharing the same rate limiter
    reranker1 = LogprobReranker(client=client, rate_limiter=shared_limiter)
    reranker2 = LogprobReranker(client=client, rate_limiter=shared_limiter)

    query1 = "What is artificial intelligence?"
    query2 = "Explain quantum computing"

    docs1 = [
        "AI is the simulation of human intelligence in machines.",
        "Machine learning is a subset of AI.",
        "Deep learning uses neural networks.",
    ]

    docs2 = [
        "Quantum computing uses quantum mechanics.",
        "Qubits can be in superposition.",
        "Quantum computers solve complex problems.",
    ]

    print(f"\nReranking with two rerankers sharing rate limiter...")
    start_time = time.time()

    # Run both rerankers concurrently - they share the rate limiter
    results1, results2 = await asyncio.gather(
        reranker1.rerank(query1, docs1),
        reranker2.rerank(query2, docs2),
    )

    elapsed = time.time() - start_time
    print(f"Completed in {elapsed:.2f} seconds")

    # Both rerankers contribute to the same rate limiter stats
    stats = shared_limiter.get_stats()
    print(f"\nShared Rate Limiter Stats:")
    print(f"  Total requests: {stats['total_requests']} (from both rerankers)")
    print(f"  Total tokens: {stats['total_tokens']:,.0f}")
    print(f"  Total wait time: {stats['total_wait_time_seconds']:.3f}s")

    print(f"\nReranker 1 - Top result:")
    print(f"  {results1[0].document[:60]}... (score: {results1[0].score:.3f})")

    print(f"\nReranker 2 - Top result:")
    print(f"  {results2[0].document[:60]}... (score: {results2[0].score:.3f})")


async def parallel_heavy_load():
    """Example demonstrating rate limiting under heavy parallel load."""
    print("\n" + "=" * 80)
    print("Heavy Parallel Load Example")
    print("=" * 80)

    load_dotenv()

    config = AzureConfig()
    client = AzureLLMClient(config=config)

    # Use conservative limits to show rate limiting in action
    reranker = create_reranker(
        client=client,
        rpm_limit=1000,  # Lower limit to demonstrate throttling
        tpm_limit=100000,
    )

    print(f"\nRate Limits (conservative):")
    print(f"  RPM: {reranker.rate_limiter.rpm_limit:,}")
    print(f"  TPM: {reranker.rate_limiter.tpm_limit:,}")

    # Create many documents to test rate limiting
    query = "What is machine learning?"
    documents = [
        f"Document {i}: Machine learning enables computers to learn from data without explicit programming. "
        f"It uses algorithms to identify patterns and make decisions."
        for i in range(20)
    ]

    print(f"\nScoring {len(documents)} documents in parallel...")
    print("Rate limiter will throttle requests to stay within limits...")

    start_time = time.time()
    results = await reranker.rerank(query, documents, top_k=5)
    elapsed = time.time() - start_time

    print(f"\nCompleted in {elapsed:.2f} seconds")

    stats = reranker.rate_limiter.get_stats()
    print(f"\nRate Limiter Stats:")
    print(f"  Total requests: {stats['total_requests']}")
    print(f"  Total tokens: {stats['total_tokens']:,.0f}")
    print(f"  Total wait time: {stats['total_wait_time_seconds']:.3f}s")
    print(f"  Average wait per request: {stats['total_wait_time_seconds'] / stats['total_requests']:.3f}s")

    if stats["total_wait_time_seconds"] > 0:
        print(f"\n✓ Rate limiting worked! Requests were throttled to respect limits.")
    else:
        print(f"\n  No throttling needed - load was within limits.")

    print(f"\nTop 5 Results:")
    for i, result in enumerate(results, 1):
        print(f"{i}. Score: {result.score:.3f}")


async def progressive_scoring():
    """Example showing progressive document scoring with rate limiting."""
    print("\n" + "=" * 80)
    print("Progressive Scoring Example")
    print("=" * 80)

    load_dotenv()

    config = AzureConfig()
    client = AzureLLMClient(config=config)

    reranker = LogprobReranker(client=client)

    query = "Explain deep learning"

    # Simulate progressive batches of documents
    batches = [
        ["Deep learning uses neural networks with many layers."],
        ["Machine learning is a broader field than deep learning.", "Python is popular for AI."],
        [
            "CNNs are used for image recognition.",
            "RNNs process sequential data.",
            "Transformers revolutionized NLP.",
        ],
    ]

    print(f"\nScoring documents in {len(batches)} progressive batches...")

    all_results = []
    total_start = time.time()

    for batch_num, batch in enumerate(batches, 1):
        print(f"\n--- Batch {batch_num} ({len(batch)} documents) ---")

        batch_start = time.time()
        results = await reranker.rerank(query, batch)
        batch_elapsed = time.time() - batch_start

        all_results.extend(results)

        print(f"Completed in {batch_elapsed:.2f}s")

        stats = reranker.rate_limiter.get_stats()
        print(f"  Cumulative requests: {stats['total_requests']}")
        print(f"  Cumulative tokens: {stats['total_tokens']:,.0f}")
        print(f"  Cumulative wait time: {stats['total_wait_time_seconds']:.3f}s")

    total_elapsed = time.time() - total_start

    print(f"\n{'=' * 60}")
    print(f"Total time for all batches: {total_elapsed:.2f}s")

    # Sort all results
    all_results.sort(key=lambda x: x.score, reverse=True)

    print(f"\nTop 3 Overall:")
    for i, result in enumerate(all_results[:3], 1):
        print(f"{i}. Score: {result.score:.3f} - {result.document[:50]}...")


async def rate_limiter_recovery():
    """Example showing how rate limiter recovers over time."""
    print("\n" + "=" * 80)
    print("Rate Limiter Recovery Example")
    print("=" * 80)

    load_dotenv()

    config = AzureConfig()
    client = AzureLLMClient(config=config)

    # Use low limits to demonstrate recovery
    reranker = create_reranker(
        client=client,
        rpm_limit=500,
        tpm_limit=50000,
    )

    print(f"\nRate Limits:")
    print(f"  RPM: {reranker.rate_limiter.rpm_limit:,}")
    print(f"  TPM: {reranker.rate_limiter.tpm_limit:,}")

    query = "What is AI?"
    documents = [f"Document {i} about artificial intelligence." for i in range(10)]

    # First burst
    print(f"\n--- First burst: {len(documents)} documents ---")
    start = time.time()
    await reranker.rerank(query, documents)
    elapsed = time.time() - start

    stats1 = reranker.rate_limiter.get_stats()
    print(f"Time: {elapsed:.2f}s")
    print(f"Wait time: {stats1['total_wait_time_seconds']:.3f}s")
    print(f"RPM available: {stats1['rpm_available']:.0f}/{reranker.rate_limiter.rpm_limit}")
    print(f"TPM available: {stats1['tpm_available']:.0f}/{reranker.rate_limiter.tpm_limit:,}")

    # Wait for recovery
    wait_time = 5
    print(f"\n--- Waiting {wait_time} seconds for rate limiter recovery ---")
    await asyncio.sleep(wait_time)

    # Check recovery
    stats2 = reranker.rate_limiter.get_stats()
    print(f"After recovery:")
    print(f"  RPM available: {stats2['rpm_available']:.0f}/{reranker.rate_limiter.rpm_limit}")
    print(f"  TPM available: {stats2['tpm_available']:.0f}/{reranker.rate_limiter.tpm_limit:,}")

    # Second burst
    print(f"\n--- Second burst: {len(documents)} documents ---")
    start = time.time()
    await reranker.rerank(query, documents)
    elapsed = time.time() - start

    stats3 = reranker.rate_limiter.get_stats()
    print(f"Time: {elapsed:.2f}s")
    print(f"Wait time since start: {stats3['total_wait_time_seconds']:.3f}s")
    print(f"Second burst wait: {stats3['total_wait_time_seconds'] - stats1['total_wait_time_seconds']:.3f}s")


async def main():
    """Run all examples."""
    examples = [
        ("Basic Rate Limiting", basic_rate_limiting),
        ("Custom Rate Limits", custom_rate_limits),
        ("Shared Rate Limiter", shared_rate_limiter),
        ("Heavy Parallel Load", parallel_heavy_load),
        ("Progressive Scoring", progressive_scoring),
        ("Rate Limiter Recovery", rate_limiter_recovery),
    ]

    print("\n" + "=" * 80)
    print("Reranker Rate Limiting Examples")
    print("=" * 80)
    print("\nThese examples demonstrate the built-in rate limiting capabilities")
    print("that prevent hitting Azure OpenAI quotas during parallel scoring.\n")

    for name, example_func in examples:
        try:
            await example_func()
            await asyncio.sleep(1)  # Brief pause between examples
        except Exception as e:
            print(f"\n❌ Error in {name}: {e}")
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 80)
    print("All examples completed!")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
