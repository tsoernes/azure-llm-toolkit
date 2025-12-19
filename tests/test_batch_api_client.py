import asyncio
import json
import tempfile
from types import SimpleNamespace

import pytest

from azure_llm_toolkit.batch_api import (
    AzureBatchAPIClient,
    BatchJobStatus,
    BatchQuotaMonitor,
)
from azure_llm_toolkit.config import AzureConfig


class DummyFiles:
    """Dummy files client for testing AzureBatchAPIClient."""

    def __init__(self, file_id: str, content_text: str):
        self._file_id = file_id
        self._content_text = content_text

    async def create(self, file, purpose: str):
        # Purpose should be "batch"
        assert purpose == "batch"
        return SimpleNamespace(id=self._file_id)

    async def content(self, file_id: str):
        # Return an object with a .text attribute
        # Accept any file_id for testing
        return SimpleNamespace(text=self._content_text)


class DummyBatches:
    """Dummy batches client for testing AzureBatchAPIClient."""

    def __init__(self, job_id: str, status: str, request_counts: dict[str, int]):
        self._job_id = job_id
        self._status = status
        self._counts = request_counts

    async def create(self, input_file_id: str, endpoint: str, completion_window: str):
        # Validate basic parameters
        assert endpoint.endswith("/v1/embeddings")
        assert completion_window == "24h"
        return SimpleNamespace(
            id=self._job_id,
            status=self._status,
            created_at=12345,
        )

    async def retrieve(self, job_id: str):
        return SimpleNamespace(
            id=job_id,
            status=self._status,
            request_counts=SimpleNamespace(
                total=self._counts.get("total", 0),
                completed=self._counts.get("completed", 0),
                failed=self._counts.get("failed", 0),
            ),
            output_file_id="out-1" if self._status in ("completed", "succeeded") else None,
        )

    async def cancel(self, job_id: str):
        return SimpleNamespace(id=job_id, status=BatchJobStatus.CANCELLED, cancelled_at="now")

    async def list(self, limit: int = 20):
        data = []
        for i in range(limit):
            data.append(
                SimpleNamespace(
                    id=f"batch-{i}",
                    status=BatchJobStatus.COMPLETED,
                    created_at=1000 + i,
                    input_file_id="in-1",
                    output_file_id="out-1",
                )
            )
        return SimpleNamespace(data=data)


@pytest.mark.asyncio
async def test_create_raises_on_empty_inputs():
    """AzureBatchAPIClient.create should reject empty inputs."""
    config = AzureConfig()  # credentials not used for empty-input check
    dummy_client = SimpleNamespace(files=None, batches=None)
    client = AzureBatchAPIClient(config, client=dummy_client)
    with pytest.raises(ValueError):
        await client.create(model="m", inputs=[])


@pytest.mark.asyncio
async def test_full_create_status_result_cancel_flow(tmp_path, monkeypatch):
    """End-to-end flow: create -> status -> result -> cancel."""
    # Prepare a mock JSONL result for get_result
    example_response = {
        "custom_id": "request-0",
        "response": {
            "status_code": 200,
            "body": {"data": [{"embedding": [1.0, 2.0, 3.0]}]},
        },
    }
    content_text = json.dumps(example_response)

    files = DummyFiles(file_id="file-1", content_text=content_text)
    batches = DummyBatches(
        job_id="job-1",
        status="completed",
        request_counts={"total": 1, "completed": 1, "failed": 0},
    )
    client = AzureBatchAPIClient(AzureConfig(), client=SimpleNamespace(files=files, batches=batches))

    # Create
    job = await client.create(model="bert", inputs=[[0, 1, 2]])
    assert job["id"] == "job-1"

    # get_status maps to BatchJobStatus
    status = await client.get_status("job-1")
    assert status["state"] == BatchJobStatus.SUCCEEDED

    # get_result returns the parsed embedding
    result = await client.get_result("job-1")
    assert isinstance(result, dict) and "data" in result
    assert result["data"][0]["embedding"] == [1.0, 2.0, 3.0]

    # cancel returns cancelled status
    cancel = await client.cancel("job-1")
    assert cancel["status"] == BatchJobStatus.CANCELLED

    # list_jobs returns the expected count
    jobs = await client.list_jobs(limit=3)
    assert len(jobs) == 3
    assert all(isinstance(entry, dict) for entry in jobs)


@pytest.mark.asyncio
async def test_wait_for_completion_success_and_timeout(monkeypatch):
    """Test wait_for_completion handles success and timeout."""
    config = AzureConfig()
    dummy_files = SimpleNamespace()
    # Set retrieve to alternate running -> succeeded
    statuses = ["running", "running", "succeeded"]

    async def retrieve(job_id):
        state = statuses.pop(0)
        return SimpleNamespace(
            id=job_id,
            status=state,
            request_counts=SimpleNamespace(total=0, completed=0, failed=0),
            output_file_id="f" if state == "succeeded" else None,
        )

    dummy_client = SimpleNamespace(files=dummy_files, batches=SimpleNamespace(retrieve=retrieve))
    client = AzureBatchAPIClient(config, client=dummy_client)

    # Success path (fast polling)
    status = await client.wait_for_completion("job-x", poll_interval=0.01, timeout=1)
    assert status["state"] == BatchJobStatus.SUCCEEDED

    # Timeout path: retrieve always returns "running"
    statuses2 = ["running"] * 10

    async def always_running(job_id: str):
        return SimpleNamespace(
            id=job_id,
            status="running",
            request_counts=SimpleNamespace(total=0, completed=0, failed=0),
            output_file_id=None,
        )

    dummy_client2 = SimpleNamespace(
        files=dummy_files,
        batches=SimpleNamespace(retrieve=always_running),
    )
    client2 = AzureBatchAPIClient(config, client=dummy_client2)
    with pytest.raises(TimeoutError):
        await client2.wait_for_completion("job-x", poll_interval=0.01, timeout=0.05)


def test_quota_monitor_limits_and_feasibility():
    """BatchQuotaMonitor should return correct quota limits and feasibility."""
    config = AzureConfig()
    # Default subscription_type='enterprise'
    monitor = BatchQuotaMonitor(config)
    # enterprise default for gpt-4o
    limit_ent = monitor.get_quota_limit("gpt-4o")
    assert limit_ent == BatchQuotaMonitor.QUOTA_LIMITS["enterprise"]["gpt-4o"]

    # Default fallback for unknown models
    limit_unknown = monitor.get_quota_limit("unknown-model")
    assert isinstance(limit_unknown, int)

    # Simple estimation
    tokens = monitor.estimate_job_tokens(num_texts=100, avg_tokens_per_text=10)
    assert tokens == 1000

    # Feasible job
    can, msg = monitor.check_job_feasibility(num_texts=10, avg_tokens_per_text=10, safety_margin=0.1)
    assert can is True and "✅" in msg

    # Infeasible job (huge)
    can2, msg2 = monitor.check_job_feasibility(num_texts=1_000_000, avg_tokens_per_text=100)
    assert can2 is False
    assert "❌" in msg2 and "Split into" in msg2
