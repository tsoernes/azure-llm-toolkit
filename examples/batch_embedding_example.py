"""Example of high-performance batch embedding using Polars DataFrames.

This example demonstrates the PolarsBatchEmbedder for efficiently embedding
large datasets with intelligent batching and rate limiting.

Requirements:
    pip install azure-llm-toolkit[polars]
"""

import asyncio
from pathlib import Path

import polars as pl
from dotenv import load_dotenv

from azure_llm_toolkit import AzureConfig, CostEstimator, PolarsBatchEmbedder

# Load environment variables
load_dotenv()


async def basic_batch_embedding():
    """Basic example: Embed a DataFrame of texts."""
    print("\n=== Basic Batch Embedding ===")

    # Create sample data
    texts = [
        "Azure OpenAI provides powerful language models.",
        "Machine learning is transforming many industries.",
        "Python is a great language for AI development.",
        "Embeddings represent text as high-dimensional vectors.",
        "Vector databases enable semantic search capabilities.",
    ]

    df = pl.DataFrame({"id": range(len(texts)), "text": texts})

    print(f"Input DataFrame:\n{df}\n")

    # Configure and create embedder
    config = AzureConfig()
    embedder = PolarsBatchEmbedder(config)

    # Embed the DataFrame
    result_df = await embedder.embed_dataframe(df, text_column="text")

    print(f"Result columns: {result_df.columns}")
    print(f"Number of rows: {len(result_df)}")
    print(f"Embedding shape: {result_df['text.embedding'][0].shape}")
    print(f"\nToken counts:\n{result_df.select(['id', 'text.token_count'])}")


async def large_dataset_example():
    """Example: Process a large dataset with batching."""
    print("\n=== Large Dataset Batch Embedding ===")

    # Create a larger dataset
    n_docs = 1000
    texts = [f"This is document number {i} with some sample content about various topics." for i in range(n_docs)]

    df = pl.DataFrame({"doc_id": range(n_docs), "content": texts})

    print(f"Processing {len(df)} documents...")

    # Configure embedder with custom rate limits
    config = AzureConfig()
    embedder = PolarsBatchEmbedder(
        config=config,
        max_tokens_per_minute=450_000,  # Adjust based on your quota
        max_lists_per_query=1000,  # How many texts per API call
        sleep_sec=20,  # Initial sleep on rate limit error
    )

    # Embed
    result_df = await embedder.embed_dataframe(df, text_column="content", verbose=True)

    print(f"\nEmbedded {len(result_df)} documents")
    print(f"Total tokens: {result_df['content.token_count'].sum():,}")
    print(f"Mean tokens per doc: {result_df['content.token_count'].mean():.1f}")


async def incremental_embedding():
    """Example: Only embed new documents, skip already embedded ones."""
    print("\n=== Incremental Embedding ===")

    # Create DataFrame with some existing embeddings (simulated as None)
    df = pl.DataFrame(
        {
            "id": range(10),
            "text": [f"Document {i}" for i in range(10)],
            "text.embedding": [None] * 10,  # Start with no embeddings
        }
    )

    print("Initial DataFrame:")
    print(df.select(["id", "text"]))

    # First embedding pass
    config = AzureConfig()
    embedder = PolarsBatchEmbedder(config)

    df = await embedder.embed_dataframe(df, text_column="text", re_embed=False)
    print(f"\nAfter first pass: {df['text.embedding'].null_count()} null embeddings")

    # Add new documents
    new_docs = pl.DataFrame(
        {
            "id": range(10, 15),
            "text": [f"Document {i}" for i in range(10, 15)],
            "text.embedding": [None] * 5,
        }
    )

    df = pl.concat([df, new_docs])
    print(f"\nAfter adding new docs: {len(df)} total documents")

    # Second pass - only new documents will be embedded
    df = await embedder.embed_dataframe(df, text_column="text", re_embed=False)
    print(f"After second pass: {df['text.embedding'].null_count()} null embeddings")
    print(f"All {len(df)} documents now have embeddings!")


async def batch_with_long_texts():
    """Example: Handle texts that exceed token limits."""
    print("\n=== Batch Embedding with Long Texts ===")

    # Create texts of varying lengths, including some very long ones
    short_text = "This is a short document."
    medium_text = " ".join(["This is a medium length document."] * 50)
    long_text = " ".join(["This is a very long document that exceeds token limits."] * 200)

    df = pl.DataFrame(
        {
            "id": [1, 2, 3],
            "text": [short_text, medium_text, long_text],
            "description": ["Short", "Medium", "Long"],
        }
    )

    config = AzureConfig()
    embedder = PolarsBatchEmbedder(
        config=config,
        max_tokens_per_row=512,  # Low limit to demonstrate splitting
    )

    # Embed - long texts will be split, embedded separately, then averaged
    result_df = await embedder.embed_dataframe(df, text_column="text")

    print("Results:")
    print(result_df.select(["id", "description", "text.token_count"]))
    print("\nNote: Long texts exceeding max_tokens_per_row are automatically")
    print("split, embedded in parts, and combined using weighted averaging.")


async def cost_tracking_example():
    """Example: Track embedding costs."""
    print("\n=== Cost Tracking Example ===")

    # Create sample data
    df = pl.DataFrame({"text": [f"Sample text number {i} for embedding" for i in range(100)]})

    # Create embedder with cost estimator
    config = AzureConfig()
    cost_estimator = CostEstimator(currency="kr")

    # Set custom pricing if needed
    cost_estimator.set_model_pricing(
        model="text-embedding-3-large",
        input_price=1.03,  # NOK per 1M tokens
        output_price=0.0,
        cached_input_price=0.0,
    )

    embedder = PolarsBatchEmbedder(config=config, cost_estimator=cost_estimator)

    # Embed and get cost estimate
    result_df = await embedder.embed_dataframe(df, text_column="text")

    total_tokens = result_df["text.token_count"].sum()
    estimated_cost = cost_estimator.estimate_cost(
        model=config.embedding_deployment,
        tokens_input=total_tokens,
    )

    print(f"Embedded {len(result_df)} texts")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Estimated cost: {estimated_cost:.4f} {cost_estimator.currency}")


async def save_and_load_embeddings():
    """Example: Save embeddings to Parquet and reload."""
    print("\n=== Save and Load Embeddings ===")

    # Create and embed data
    df = pl.DataFrame({"text": [f"Document {i}" for i in range(20)]})

    config = AzureConfig()
    embedder = PolarsBatchEmbedder(config)

    result_df = await embedder.embed_dataframe(df, text_column="text")

    # Save to Parquet
    output_path = Path("embeddings_output.parquet")
    result_df.write_parquet(output_path)
    print(f"Saved embeddings to {output_path}")

    # Load back
    loaded_df = pl.read_parquet(output_path)
    print(f"\nLoaded {len(loaded_df)} documents from disk")
    print(f"Columns: {loaded_df.columns}")
    print(f"Embedding shape: {loaded_df['text.embedding'][0].shape}")

    # Clean up
    output_path.unlink()
    print(f"\nCleaned up {output_path}")


async def advanced_configuration():
    """Example: Advanced embedder configuration."""
    print("\n=== Advanced Configuration ===")

    config = AzureConfig(
        embedding_deployment="text-embedding-3-large",
        timeout_seconds=120,  # Longer timeout for large batches
    )

    embedder = PolarsBatchEmbedder(
        config=config,
        max_tokens_per_minute=1_000_000,  # High throughput
        max_tokens_per_row=8190,  # Maximum tokens per text
        max_lists_per_query=2048,  # Maximum texts per API call
        sleep_sec=60,  # Initial sleep on rate limit
        sleep_inc=10,  # Increment sleep on repeated rate limits
    )

    # Create test data
    df = pl.DataFrame({"text": [f"Test document {i}" for i in range(50)]})

    print("Configuration:")
    print(f"  Model: {config.embedding_deployment}")
    print(f"  Max TPM: {embedder.max_tokens_per_minute:,}")
    print(f"  Max tokens per row: {embedder.max_tokens_per_row}")
    print(f"  Max lists per query: {embedder.max_lists_per_query}")

    result_df = await embedder.embed_dataframe(df, text_column="text")
    print(f"\nEmbedded {len(result_df)} documents with advanced configuration")


async def main():
    """Run all examples."""
    print("Azure LLM Toolkit - Batch Embedding Examples")
    print("=" * 60)

    try:
        # Run examples
        await basic_batch_embedding()
        await large_dataset_example()
        await incremental_embedding()
        await batch_with_long_texts()
        await cost_tracking_example()
        await save_and_load_embeddings()
        await advanced_configuration()

        print("\n" + "=" * 60)
        print("All examples completed successfully!")

    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
