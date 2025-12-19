"""Basic usage examples for azure-llm-toolkit."""

import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv

from azure_llm_toolkit import (
    AzureConfig,
    AzureLLMClient,
    CostEstimator,
    InMemoryCostTracker,
)

# Load environment variables from .env file
load_dotenv()


async def example_embeddings():
    """Example: Generate embeddings for texts."""
    print("\n=== Embeddings Example ===")

    config = AzureConfig()
    client = AzureLLMClient(config=config)

    texts = [
        "Azure OpenAI provides powerful language models.",
        "Machine learning is transforming many industries.",
        "Python is a great language for AI development.",
    ]

    result = await client.embed_texts(texts)

    print(f"Generated {len(result.embeddings)} embeddings")
    print(f"Embedding dimension: {len(result.embeddings[0])}")
    print(f"Total tokens used: {result.usage.total_tokens}")
    print(f"Model: {result.model}")


async def example_chat_completion():
    """Example: Chat completion with usage tracking."""
    print("\n=== Chat Completion Example ===")

    config = AzureConfig()
    client = AzureLLMClient(config=config)

    messages = [{"role": "user", "content": "What is machine learning in one sentence?"}]

    result = await client.chat_completion(
        messages=messages,
        system_prompt="You are a helpful AI assistant. Be concise.",
        max_tokens=100,
    )

    print(f"Response: {result.content}")
    print(f"Tokens - Input: {result.usage.prompt_tokens}, Output: {result.usage.completion_tokens}")
    print(f"Total tokens: {result.usage.total_tokens}")
    print(f"Finish reason: {result.finish_reason}")


async def example_cost_tracking():
    """Example: Track costs for API operations."""
    print("\n=== Cost Tracking Example ===")

    # Create cost tracker
    cost_tracker = InMemoryCostTracker(currency="kr")
    cost_estimator = CostEstimator(currency="kr")

    config = AzureConfig()
    client = AzureLLMClient(
        config=config,
        cost_tracker=cost_tracker,
        cost_estimator=cost_estimator,
    )

    # Perform some operations
    await client.embed_texts(["Sample text for embedding"], track_cost=True)

    await client.chat_completion(
        messages=[{"role": "user", "content": "Hello!"}],
        track_cost=True,
    )

    # Get cost summary
    summary = cost_tracker.get_summary()

    print(f"Total cost: {summary['total_cost']:.4f} {summary['currency']}")
    print(f"Total entries: {summary['total_entries']}")
    print(f"Total input tokens: {summary['total_tokens_input']}")
    print(f"Total output tokens: {summary['total_tokens_output']}")
    print("\nBy category:")
    for category, amount in summary["by_category"].items():
        print(f"  {category}: {amount:.4f} {summary['currency']}")
    print("\nBy model:")
    for model, amount in summary["by_model"].items():
        print(f"  {model}: {amount:.4f} {summary['currency']}")


async def example_rate_limiting():
    """Example: Rate limiting in action."""
    print("\n=== Rate Limiting Example ===")

    config = AzureConfig()
    client = AzureLLMClient(
        config=config,
        enable_rate_limiting=True,
    )

    print("Embedding 10 texts with rate limiting enabled...")
    texts = [f"Document number {i}" for i in range(10)]

    import time

    start = time.time()

    result = await client.embed_texts(texts, batch_size=5)

    elapsed = time.time() - start

    print(f"Completed in {elapsed:.2f} seconds")
    print(f"Generated {len(result.embeddings)} embeddings")
    print(f"Total tokens: {result.usage.total_tokens}")

    # Get rate limiter stats
    if client.rate_limiter_pool:
        stats = client.rate_limiter_pool.get_all_stats()
        for model, model_stats in stats.items():
            print(f"\nRate limiter stats for {model}:")
            print(f"  Total requests: {model_stats['total_requests']}")
            print(f"  Total tokens: {model_stats['total_tokens']}")
            print(f"  Total wait time: {model_stats['total_wait_time_seconds']:.2f}s")


# Query rewriting example removed (function intentionally deleted).
# If you previously relied on query rewriting, implement your own
# rewriting logic or use an external service; this project no longer
# provides a built-in `rewrite_query` API.


async def example_metadata_extraction():
    """Example: Extract metadata from filenames and content."""
    print("\n=== Metadata Extraction Example ===")

    config = AzureConfig()
    client = AzureLLMClient(config=config)

    # From filename
    filename = "2024-Q4-Financial-Report-Final.pdf"
    metadata = await client.extract_metadata_from_filename(filename)

    print(f"\nMetadata from filename '{filename}':")
    for key, value in metadata.items():
        print(f"  {key}: {value}")

    # From content
    content = """
    Machine Learning Best Practices
    Author: Jane Doe
    Date: 2024-12-01

    This document provides comprehensive guidelines for implementing
    machine learning systems in production environments.
    """

    metadata = await client.extract_metadata_from_content(content=content, filename="ml-best-practices.md")

    print(f"\nMetadata from content:")
    for key, value in metadata.items():
        print(f"  {key}: {value}")


async def example_rag_answer():
    """Example: RAG-style question answering."""
    print("\n=== RAG Answer Example ===")

    config = AzureConfig()
    client = AzureLLMClient(config=config)

    context = """
    Azure OpenAI Service provides REST API access to OpenAI's powerful language models
    including GPT-4, GPT-4 Turbo, GPT-3.5-Turbo, and Embeddings model series. These models
    can be easily adapted to your specific task including but not limited to content generation,
    summarization, image understanding, semantic search, and natural language to code translation.
    """

    question = "What models are available in Azure OpenAI Service?"

    result = await client.generate_answer(
        question=question,
        context=context,
    )

    print(f"Question: {question}")
    print(f"Answer: {result.content}")
    print(f"Tokens used: {result.usage.total_tokens}")


async def example_token_counting():
    """Example: Count tokens and estimate costs."""
    print("\n=== Token Counting Example ===")

    config = AzureConfig()
    client = AzureLLMClient(config=config)

    text = "This is a sample text for token counting and cost estimation."

    token_count = client.count_tokens(text)
    print(f"Text: '{text}'")
    print(f"Token count: {token_count}")

    # Estimate embedding cost
    embedding_cost = client.estimate_embedding_cost(text)
    print(f"Estimated embedding cost: {embedding_cost:.6f} kr")

    # Estimate chat cost
    messages = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": text}]

    message_tokens = client.count_message_tokens(messages)
    chat_cost = client.estimate_chat_cost(messages, estimated_output_tokens=100)

    print(f"Message tokens: {message_tokens}")
    print(f"Estimated chat cost (with ~100 output tokens): {chat_cost:.6f} kr")


async def main():
    """Run all examples."""
    print("Azure LLM Toolkit - Basic Usage Examples")
    print("=" * 50)

    # Check if credentials are set
    if not os.getenv("AZURE_OPENAI_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        print("\nError: Azure OpenAI API key not set!")
        print("Please set AZURE_OPENAI_API_KEY or OPENAI_API_KEY environment variable.")
        return

    if not os.getenv("AZURE_ENDPOINT") and not os.getenv("AZURE_OPENAI_ENDPOINT"):
        print("\nError: Azure endpoint not set!")
        print("Please set AZURE_ENDPOINT or AZURE_OPENAI_ENDPOINT environment variable.")
        return

    try:
        # Run examples (comment out any you don't want to run)
        await example_embeddings()
        await example_chat_completion()
        await example_cost_tracking()
        await example_rate_limiting()
        # example_query_rewriting() removed — query rewriting no longer provided by this toolkit
        await example_metadata_extraction()
        await example_rag_answer()
        await example_token_counting()

        print("\n" + "=" * 50)
        print("All examples completed successfully!")

    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
