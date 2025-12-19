# azure-llm-toolkit/tests/test_batch_api.py
import asyncio
import time

import numpy as np
import pytest

from azure_llm_toolkit import AzureConfig, PolarsBatchEmbedder
from azure_llm_toolkit.batch_api import MockBatchAPIClient


@pytest.mark.asyncio
async def test_mock_batch_api_client_basic():
    """
    Verify the MockBatchAPIClient lifecycle: create -> run -> status -> result.
    This validates the minimal async job contract expected by the embedder.
    """
    client = MockBatchAPIClient(embedding_dim=6, job_delay_seconds=0.5, seed=123)

    inputs = ["hello world", "quick brown fox", "sample text"]
    job_desc = await client.create(model="test-model", inputs=inputs)
    assert "id" in job_desc
    job_id = job_desc["id"]

    # Poll status until succeeded (with timeout)
    start = time.time()
    timeout = 5.0
    state = None
    while time.time() - start < timeout:
        status = await client.get_status(job_id)
        state = status.get("state")
        if state in ("succeeded", "finished", "completed"):
            break
        await asyncio.sleep(0.1)
    assert state in ("succeeded", "finished", "completed"), f"job did not complete, last state={state}"

    result = await client.get_result(job_id)
    assert isinstance(result, dict)
    assert "data" in result and isinstance(result["data"], list)
    assert len(result["data"]) == len(inputs)
    # Check each item has an 'embedding' list of the expected dim
    for item in result["data"]:
        assert "embedding" in item
        emb = item["embedding"]
        assert isinstance(emb, list)
        assert len(emb) == 6  # embedding_dim


@pytest.mark.asyncio
async def test_polars_batch_embedder_with_mock_batch_api():
    """
    End-to-end test: PolarsBatchEmbedder using the mock batch API client.
    Ensures the embedder uses the batch API path and returns embeddings in the expected shape.
    """
    config = AzureConfig()
    # Use small embedding dim to keep test lightweight
    mock_batch_client = MockBatchAPIClient(embedding_dim=5, job_delay_seconds=0.3, seed=7)

    # A tiny fake RateLimiter that records acquired tokens; the embedder will call .acquire(tokens=...)
    class FakeLimiter:
        def __init__(self):
            self.acquired = []
            self.calls = 0

        async def acquire(self, tokens: int) -> None:
            self.calls += 1
            self.acquired.append(int(tokens))
            # No actual throttling in the fake; simulate immediate acquire
            return None

    fake_limiter = FakeLimiter()

    embedder = PolarsBatchEmbedder(
        config=config,
        max_tokens_per_minute=10_000,
        max_lists_per_query=10,
        use_rate_limiting=True,
        rate_limiter=fake_limiter,
        use_batch_api=True,
        batch_api_client=mock_batch_client,
    )

    texts = ["alpha", "beta", "gamma", "delta"]
    embeddings, metadata = await embedder.embed_texts(texts, show_progress=False)

    # Basic metadata checks
    assert metadata["num_texts"] == len(texts)
    assert metadata["total_tokens"] >= 0
    assert isinstance(metadata["estimated_cost"], float)

    # Embeddings should be list-like with one embedding per text
    assert len(embeddings) == len(texts)
    for emb in embeddings:
        assert isinstance(emb, np.ndarray)
        assert emb.shape[0] == 5  # mock embedding_dim

    # Ensure the fake limiter was called at least once (batch-level acquire)
    assert fake_limiter.calls >= 1
    assert all(isinstance(t, int) and t >= 0 for t in fake_limiter.acquired)


# Optional: quick smoke test for embed_dataframe with the mock client as well
@pytest.mark.asyncio
async def test_polars_embedder_dataframe_batch_api():
    config = AzureConfig()
    mock_batch_client = MockBatchAPIClient(embedding_dim=4, job_delay_seconds=0.2, seed=11)
    embedder = PolarsBatchEmbedder(config=config, use_batch_api=True, batch_api_client=mock_batch_client)

    import polars as pl

    df = pl.DataFrame({"text": ["one", "two", "three"]})
    result_df = await embedder.embed_dataframe(df, text_column="text", verbose=False)

    # Check that embedding column exists and contains arrays of the correct length
    assert "text.embedding" in result_df.columns
    embeddings = result_df["text.embedding"].to_list()
    assert len(embeddings) == 3
    for emb in embeddings:
        # The Polars embedder writes arrays (as Python lists or numpy arrays); allow both
        assert len(list(emb)) == 4
