"""Tests for PolarsBatchEmbedder."""

import polars as pl
import pytest

from azure_llm_toolkit import PolarsBatchEmbedder


@pytest.mark.asyncio
async def test_embed_dataframe(batch_embedder, sample_texts):
    """Test embedding a DataFrame."""
    df = pl.DataFrame({"id": range(len(sample_texts)), "text": sample_texts})

    result_df = await batch_embedder.embed_dataframe(df, text_column="text")

    # Check columns exist
    assert "text.tokens" in result_df.columns
    assert "text.token_count" in result_df.columns
    assert "text.embedding" in result_df.columns

    # Check all rows have embeddings
    assert result_df["text.embedding"].null_count() == 0

    # Check embedding dimensions
    first_embedding = result_df["text.embedding"][0]
    assert len(first_embedding) > 0


@pytest.mark.asyncio
async def test_embed_dataframe_incremental(batch_embedder):
    """Test incremental embedding (skip already embedded texts)."""
    # Create DataFrame with some null embeddings
    df = pl.DataFrame(
        {
            "id": range(5),
            "text": [f"Document {i}" for i in range(5)],
            "text.embedding": [None] * 5,
        }
    )

    # First pass - embed all
    df = await batch_embedder.embed_dataframe(df, text_column="text", re_embed=False)
    assert df["text.embedding"].null_count() == 0

    # Add new documents
    new_docs = pl.DataFrame(
        {
            "id": range(5, 8),
            "text": [f"Document {i}" for i in range(5, 8)],
            "text.embedding": [None] * 3,
        }
    )
    df = pl.concat([df, new_docs])

    # Second pass - only new documents should be embedded
    df = await batch_embedder.embed_dataframe(df, text_column="text", re_embed=False)
    assert df["text.embedding"].null_count() == 0
    assert len(df) == 8


@pytest.mark.asyncio
async def test_embed_dataframe_large_batch(batch_embedder):
    """Test embedding a larger batch of texts."""
    n_docs = 50
    texts = [f"This is document number {i} with some content." for i in range(n_docs)]
    df = pl.DataFrame({"id": range(n_docs), "text": texts})

    result_df = await batch_embedder.embed_dataframe(df, text_column="text", verbose=False)

    assert len(result_df) == n_docs
    assert result_df["text.embedding"].null_count() == 0

    # Check token counts
    assert result_df["text.token_count"].sum() > 0
    assert result_df["text.token_count"].min() > 0


@pytest.mark.asyncio
async def test_embed_texts(batch_embedder, sample_texts):
    """Test embedding a list of texts."""
    embeddings, metadata = await batch_embedder.embed_texts(sample_texts, show_progress=False)

    assert len(embeddings) == len(sample_texts)
    assert all(len(emb) > 0 for emb in embeddings)

    # Check metadata
    assert metadata["num_texts"] == len(sample_texts)
    assert metadata["total_tokens"] > 0
    assert "estimated_cost" in metadata


@pytest.mark.asyncio
async def test_tokenize_dataframe(batch_embedder, sample_texts):
    """Test DataFrame tokenization."""
    df = pl.DataFrame({"text": sample_texts})

    tokenized_df = batch_embedder._tokenize_dataframe(df, text_column="text", verbose=False)

    assert "text.tokens" in tokenized_df.columns
    assert "text.token_count" in tokenized_df.columns
    assert tokenized_df["text.token_count"].sum() > 0


@pytest.mark.asyncio
async def test_create_batches(batch_embedder, sample_texts):
    """Test batch creation."""
    df = pl.DataFrame({"text": sample_texts})
    df = batch_embedder._tokenize_dataframe(df, text_column="text", verbose=False)

    batched_df = batch_embedder._create_batches(df, token_count_column="text.token_count")

    assert "batch_id" in batched_df.columns
    assert batched_df["batch_id"].n_unique() > 0


@pytest.mark.asyncio
async def test_weighted_averaging(batch_embedder):
    """Test weighted averaging for split texts."""
    import numpy as np

    # Create dummy embeddings
    embeddings = [
        np.array([1.0, 2.0, 3.0]),
        np.array([4.0, 5.0, 6.0]),
        np.array([7.0, 8.0, 9.0]),
    ]

    # First text split into 2 parts, second text not split
    split_lengths = [[2, 2], [1]]

    result = batch_embedder._weighted_average_embeddings(embeddings, split_lengths)

    assert len(result) == 2  # Two original texts
    assert len(result[0]) == 3  # Embedding dimension
    assert len(result[1]) == 3


@pytest.mark.asyncio
async def test_embed_dataframe_with_nulls(batch_embedder):
    """Test embedding DataFrame with null text values."""
    df = pl.DataFrame(
        {
            "id": range(5),
            "text": ["Doc 1", None, "Doc 3", "Doc 4", None],
        }
    )

    result_df = await batch_embedder.embed_dataframe(df, text_column="text", verbose=False)

    # Only non-null texts should be embedded
    non_null_count = df["text"].null_count()
    assert result_df["text.embedding"].null_count() >= non_null_count


@pytest.mark.asyncio
async def test_cost_estimation(batch_embedder, sample_texts):
    """Test cost estimation in batch embedder."""
    embeddings, metadata = await batch_embedder.embed_texts(sample_texts, show_progress=False)

    assert "estimated_cost" in metadata
    assert isinstance(metadata["estimated_cost"], float)
    assert metadata["estimated_cost"] >= 0
    assert "currency" in metadata
