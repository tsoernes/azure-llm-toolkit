"""
Comprehensive example demonstrating PolarsBatchEmbedder with various input types.

This example shows:
1. Embedding a list of texts
2. Embedding a Polars DataFrame
3. Using with rate limiting integration
4. Cost tracking and metrics
5. Handling large datasets efficiently
"""

import asyncio

import polars as pl

from azure_llm_toolkit import (
    AzureConfig,
    InMemoryCostTracker,
    PolarsBatchEmbedder,
    RateLimiter,
)


async def example_1_embed_text_list():
    """Example 1: Embed a simple list of texts."""
    print("\n" + "=" * 60)
    print("Example 1: Embedding a List of Texts")
    print("=" * 60)

    config = AzureConfig()
    embedder = PolarsBatchEmbedder(
        config=config,
        max_tokens_per_minute=450_000,  # Rate limit
        max_lists_per_query=1024,  # Batch size
    )

    # Sample texts
    texts = [
        "Azure OpenAI provides powerful language models",
        "Embeddings convert text into numerical vectors",
        "Semantic search uses embeddings to find similar content",
        "Vector databases store and search embeddings efficiently",
        "RAG systems combine retrieval with generation",
    ]

    print(f"\nEmbedding {len(texts)} texts...")
    embeddings, metadata = await embedder.embed_texts(texts, show_progress=True)

    print(f"\n✅ Success!")
    print(f"  - Generated {len(embeddings)} embeddings")
    print(f"  - Embedding dimension: {len(embeddings[0])}")
    print(f"  - Total tokens: {metadata['total_tokens']:,}")
    print(f"  - Estimated cost: {metadata['estimated_cost']:.4f} {metadata['currency']}")
    print(f"  - Number of splits: {metadata['num_splits']}")
    print(f"  - Number of batches: {metadata['num_batches']}")


async def example_2_embed_dataframe():
    """Example 2: Embed texts in a Polars DataFrame."""
    print("\n" + "=" * 60)
    print("Example 2: Embedding a Polars DataFrame")
    print("=" * 60)

    config = AzureConfig()
    embedder = PolarsBatchEmbedder(config=config)

    # Create sample DataFrame
    df = pl.DataFrame(
        {
            "id": [f"doc_{i}" for i in range(100)],
            "title": [f"Document {i}" for i in range(100)],
            "content": [
                f"This is the content of document {i}. It contains information about topic {i % 10}."
                for i in range(100)
            ],
        }
    )

    print(f"\nDataFrame shape: {df.shape}")
    print(f"Columns: {df.columns}")
    print(f"\nFirst few rows:")
    print(df.head(3))

    # Embed the content column
    print(f"\nEmbedding 'content' column...")
    result_df = await embedder.embed_dataframe(df, text_column="content", verbose=True)

    print(f"\n✅ Success!")
    print(f"  - Result DataFrame shape: {result_df.shape}")
    print(f"  - New columns: {[col for col in result_df.columns if col not in df.columns]}")
    print(f"\nEmbedding column info:")
    print(f"  - Column name: content.embedding")
    print(f"  - Embedding dimension: {len(result_df['content.embedding'][0])}")
    print(f"\nSample row with embedding:")
    print(result_df.select(["id", "title", "content.token_count"]).head(3))


async def example_3_with_rate_limiter():
    """Example 3: Use PolarsBatchEmbedder with RateLimiter integration."""
    print("\n" + "=" * 60)
    print("Example 3: Batch Embedder with RateLimiter Integration")
    print("=" * 60)

    config = AzureConfig()

    # Create a rate limiter for the embedding deployment
    rate_limiter = RateLimiter(
        deployment_name=config.embedding_deployment,
        requests_per_minute=3000,
        tokens_per_minute=1_000_000,
    )

    # Create embedder with rate limiter integration
    embedder = PolarsBatchEmbedder(
        config=config,
        max_tokens_per_minute=450_000,
        use_rate_limiting=True,  # Enable integration
        rate_limiter=rate_limiter,  # Provide rate limiter instance
    )

    # Create DataFrame with more texts
    df = pl.DataFrame(
        {
            "id": range(500),
            "text": [f"Sample document number {i} with some content about topic {i % 20}." for i in range(500)],
        }
    )

    print(f"\nEmbedding {len(df)} texts with rate limiter coordination...")
    result_df = await embedder.embed_dataframe(df, text_column="text", verbose=True)

    print(f"\n✅ Success!")
    print(f"  - Embedded {len(result_df)} texts")
    print(f"  - Rate limiter ensured proper throttling")


async def example_4_cost_tracking():
    """Example 4: Track costs while embedding."""
    print("\n" + "=" * 60)
    print("Example 4: Embedding with Cost Tracking")
    print("=" * 60)

    config = AzureConfig()
    cost_tracker = InMemoryCostTracker()

    embedder = PolarsBatchEmbedder(config=config, cost_estimator=cost_tracker.estimator)

    # Large batch of texts
    texts = [f"Document {i}: This is a sample text for embedding." for i in range(1000)]

    print(f"\nEmbedding {len(texts)} texts...")
    embeddings, metadata = await embedder.embed_texts(texts, show_progress=True)

    # Record the cost
    cost_tracker.record_embedding(
        model=config.embedding_deployment,
        tokens=metadata["total_tokens"],
        cost=metadata["estimated_cost"],
    )

    print(f"\n✅ Success!")
    print(f"  - Total tokens: {metadata['total_tokens']:,}")
    print(f"  - Estimated cost: {metadata['estimated_cost']:.4f} {metadata['currency']}")
    print(f"\nCost Tracker Summary:")
    summary = cost_tracker.get_summary()
    print(f"  - Total embedding calls: {summary['embedding_calls']}")
    print(f"  - Total embedding tokens: {summary['embedding_tokens']:,}")
    print(f"  - Total cost: {summary['total_cost']:.4f} {metadata['currency']}")


async def example_5_incremental_embedding():
    """Example 5: Incrementally embed new data in a DataFrame."""
    print("\n" + "=" * 60)
    print("Example 5: Incremental DataFrame Embedding")
    print("=" * 60)

    config = AzureConfig()
    embedder = PolarsBatchEmbedder(config=config)

    # Initial DataFrame
    df = pl.DataFrame(
        {
            "id": range(100),
            "text": [f"Document {i}" for i in range(100)],
        }
    )

    print(f"\n--- First embedding pass ---")
    print(f"Embedding {len(df)} texts...")
    df = await embedder.embed_dataframe(df, text_column="text", verbose=True)

    # Add new rows
    new_rows = pl.DataFrame(
        {
            "id": range(100, 150),
            "text": [f"Document {i}" for i in range(100, 150)],
        }
    )

    # Concatenate
    df = pl.concat([df, new_rows])

    print(f"\n--- Second embedding pass (incremental) ---")
    print(f"DataFrame now has {len(df)} rows")
    print(f"Only new rows (without embeddings) will be processed...")

    # re_embed=False (default) will only embed new rows
    df = await embedder.embed_dataframe(df, text_column="text", re_embed=False, verbose=True)

    print(f"\n✅ Success!")
    print(f"  - Total rows: {len(df)}")
    print(f"  - Rows with embeddings: {df.filter(pl.col('text.embedding').is_not_null()).height}")


async def example_6_handling_long_texts():
    """Example 6: Handle texts that exceed token limits."""
    print("\n" + "=" * 60)
    print("Example 6: Handling Long Texts with Automatic Splitting")
    print("=" * 60)

    config = AzureConfig()
    embedder = PolarsBatchEmbedder(
        config=config,
        max_tokens_per_row=8190,  # Token limit per text
    )

    # Create texts of varying lengths, including some very long ones
    short_text = "Short document."
    medium_text = "Medium length document. " * 100
    long_text = "Very long document with lots of content. " * 500  # Will exceed limit

    texts = [short_text, medium_text, long_text]

    print(f"\nText lengths (characters):")
    for i, text in enumerate(texts):
        print(f"  Text {i}: {len(text):,} chars")

    print(f"\nEmbedding texts (long texts will be automatically split)...")
    embeddings, metadata = await embedder.embed_texts(texts, show_progress=True)

    print(f"\n✅ Success!")
    print(f"  - Input texts: {metadata['num_texts']}")
    print(f"  - Split segments: {metadata['num_splits']}")
    print(f"  - Output embeddings: {len(embeddings)}")
    print(f"\nNote: Long texts were split and embeddings were averaged using weighted averaging")


async def example_7_large_dataset_streaming():
    """Example 7: Process a large dataset efficiently."""
    print("\n" + "=" * 60)
    print("Example 7: Processing Large Datasets")
    print("=" * 60)

    config = AzureConfig()
    embedder = PolarsBatchEmbedder(
        config=config,
        max_tokens_per_minute=1_000_000,  # Higher rate limit
        max_lists_per_query=2048,  # Larger batch size
    )

    # Simulate a large dataset
    num_docs = 5000
    df = pl.DataFrame(
        {
            "doc_id": range(num_docs),
            "text": [f"Document {i}: Content about topic {i % 50}. " * 10 for i in range(num_docs)],
            "category": [f"category_{i % 10}" for i in range(num_docs)],
        }
    )

    print(f"\nDataset info:")
    print(f"  - Total documents: {len(df):,}")
    print(f"  - Estimated tokens: ~{len(df) * 100:,}")

    print(f"\nEmbedding large dataset...")
    result_df = await embedder.embed_dataframe(df, text_column="text", verbose=True)

    print(f"\n✅ Success!")
    print(f"  - Processed {len(result_df):,} documents")
    print(f"  - DataFrame memory size: {result_df.estimated_size() / 1024 / 1024:.2f} MB")


async def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("PolarsBatchEmbedder Comprehensive Examples")
    print("=" * 60)

    try:
        await example_1_embed_text_list()
        await example_2_embed_dataframe()
        await example_3_with_rate_limiter()
        await example_4_cost_tracking()
        await example_5_incremental_embedding()
        await example_6_handling_long_texts()
        await example_7_large_dataset_streaming()

        print("\n" + "=" * 60)
        print("All examples completed successfully! ✅")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
