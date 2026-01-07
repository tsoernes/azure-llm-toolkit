"""
LIVE integration tests for AzureBatchAPIClient against real Azure OpenAI Batch endpoints.

IMPORTANT:
- These tests exercise the real Batch API and will incur real cost.
- They require valid Azure credentials in the project .env or environment:
    AZURE_OPENAI_API_KEY and AZURE_ENDPOINT (or AZURE_OPENAI_ENDPOINT).
- Ensure your Azure resource has a deployed embedding model and sufficient quota.
"""

import asyncio
import os

import pytest
import tiktoken
from dotenv import load_dotenv

from azure_llm_toolkit import AzureConfig
from azure_llm_toolkit.batch_api import (
    AzureBatchAPIClient,
    BatchJobStatus,
    BatchQuotaMonitor,
)

# Load .env from repo root
load_dotenv()


def _require_env_vars(names: list[str]) -> bool:
    """Return True if all given environment variables are present."""
    return all(os.getenv(n) for n in names)


@pytest.fixture(scope="session")
def live_config() -> AzureConfig:
    """Load AzureConfig for live tests."""
    return AzureConfig()


@pytest.fixture(scope="session")
async def batch_client(live_config: AzureConfig) -> AzureBatchAPIClient:
    """
    Create an AzureBatchAPIClient instance for live integration tests.
    """
    return AzureBatchAPIClient(live_config)


@pytest.mark.skipif(
    not _require_env_vars(["AZURE_OPENAI_API_KEY", "AZURE_ENDPOINT"]),
    reason="Azure credentials not configured; skipping live batch API tests.",
)
@pytest.mark.asyncio
async def test_live_batch_api_end_to_end(batch_client: AzureBatchAPIClient, live_config: AzureConfig) -> None:
    """
    End-to-end batch embedding workflow:
      - create job
      - wait for completion
      - retrieve embeddings
    """
    # Tokenize a sample text
    encoder = tiktoken.encoding_for_model(live_config.embedding_deployment)
    tokens = encoder.encode("Hello Azure Batch API")

    # Submit batch job
    job_resp = await batch_client.create(
        model=live_config.embedding_deployment,
        inputs=[tokens],
    )
    job_id = job_resp.get("id")
    assert job_id, "Batch create did not return a job ID"

    # Poll until job succeeds or fails
    status = await batch_client.wait_for_completion(job_id, poll_interval=5.0, timeout=600)
    assert status["state"] == BatchJobStatus.SUCCEEDED, f"Unexpected state: {status}"

    # Retrieve results
    result = await batch_client.get_result(job_id)
    assert isinstance(result, dict) and "data" in result, "Result missing 'data'"
    data = result["data"]
    assert isinstance(data, list) and data, "No embeddings returned"
    assert "embedding" in data[0], "Embedding key missing in first result"


@pytest.mark.skipif(
    not _require_env_vars(["AZURE_OPENAI_API_KEY", "AZURE_ENDPOINT"]),
    reason="Azure credentials not configured; skipping live batch cancel test.",
)
@pytest.mark.asyncio
async def test_live_batch_api_cancel(batch_client: AzureBatchAPIClient, live_config: AzureConfig) -> None:
    """
    Test cancelling a batch job immediately after creation.
    """
    encoder = tiktoken.encoding_for_model(live_config.embedding_deployment)
    tokens = encoder.encode("Cancel this job")

    job_resp = await batch_client.create(model=live_config.embedding_deployment, inputs=[tokens])
    job_id = job_resp.get("id")
    assert job_id

    cancel_resp = await batch_client.cancel(job_id)
    assert cancel_resp.get("status") in (BatchJobStatus.CANCELLED,), "Cancel did not mark job cancelled"


@pytest.mark.skipif(
    not _require_env_vars(["AZURE_OPENAI_API_KEY", "AZURE_ENDPOINT"]),
    reason="Azure credentials not configured; skipping live quota monitor test.",
)
def test_live_quota_monitor(live_config: AzureConfig) -> None:
    """
    Verify that BatchQuotaMonitor returns sensible limits and feasibility.
    """
    monitor = BatchQuotaMonitor(live_config, subscription_type="enterprise")
    quota_limit = monitor.get_quota_limit(live_config.embedding_deployment)
    assert isinstance(quota_limit, int) and quota_limit > 0

    # A tiny job should always be feasible
    can_submit, msg = monitor.check_job_feasibility(num_texts=1, avg_tokens_per_text=10, safety_margin=0.1)
    assert can_submit, f"Quota monitor incorrectly flagged small job: {msg}"
