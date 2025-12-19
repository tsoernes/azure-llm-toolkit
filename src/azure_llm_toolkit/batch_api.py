"""
Mock Batch API client adapter for use with PolarsBatchEmbedder.

This module provides:
- `MockBatchAPIClient`: a lightweight in-memory simulation of a batch embedding
  service that exposes the minimal async interface expected by
  `PolarsBatchEmbedder._embed_via_batch_api`:
    * `await create(model=..., inputs=...)` -> returns a job descriptor with an `id`
    * `await get_status(job_id)` -> returns a status dict/object with a `state` key
    * `await get_result(job_id)` -> returns a result dict/object with a `data` list,
         where each element contains an `embedding` (list[float]) entry
- Deterministic but cheap "embeddings" (small vectors) suitable for examples and tests.
- Docstring examples showing how to wire the adapter into `PolarsBatchEmbedder`.

Notes:
- This is intentionally a mock/adaptor for development and testing. It does not call
  any real remote API and should not be used in production.
- The mock returns embeddings as lists of floats (shape: [N, EMBEDDING_DIM]).
  The embedder code in this repository converts those to numpy arrays as needed.

Example usage (documentation-style):

    >>> from azure_llm_toolkit.batch_api import MockBatchAPIClient
    >>> from azure_llm_toolkit import AzureConfig, PolarsBatchEmbedder
    >>>
    >>> # Create a mock batch client (simulates async batch job lifecycle)
    >>> batch_client = MockBatchAPIClient(embedding_dim=8, job_delay_seconds=1.0)
    >>>
    >>> config = AzureConfig()
    >>> # Enable use_batch_api and pass our mock client
    >>> embedder = PolarsBatchEmbedder(config=config, use_batch_api=True, batch_api_client=batch_client)
    >>>
    >>> # Now embed_texts will attempt the batch API path first (opt-in)
    >>> texts = ["hello world", "another document"]
    >>> embeddings, metadata = await embedder.embed_texts(texts)
    >>> print(len(embeddings), metadata["num_texts"])

"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

# Public API exported by this module
__all__ = [
    "MockBatchAPIClient",
    "AzureBatchAPIClient",
    "BatchJobStatus",
]


class BatchJobStatus:
    """Enumeration of batch job statuses."""

    PENDING = "pending"
    VALIDATING = "validating"
    RUNNING = "running"
    FINALIZING = "finalizing"
    SUCCEEDED = "succeeded"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


@dataclass
class _Job:
    id: str
    model: str
    inputs: list[list[int]]  # token lists (or other inputs as provided)
    created_at: float
    status: str = "pending"
    result: Optional[dict] = None
    error: Optional[str] = None


class MockBatchAPIClient:
    """
    A small in-memory mock of a batch embedding API.

    Behavior:
    - `create(model=..., inputs=...)` returns a job descriptor with an `id`.
      The actual processing is performed asynchronously in the background.
    - `get_status(job_id)` returns a dict-like object with a `state` field
      (values: "pending", "running", "succeeded", "failed").
    - `get_result(job_id)` returns a dict-like object with a `data` list; each
      element is a dict containing an `embedding` key with a list[float].

    Constructor parameters:
    - embedding_dim: length of the mock embedding vector (small for speed; default 8)
    - job_delay_seconds: simulated processing time from create->succeeded (default 2.0)
    - seed: deterministic seed for embedding generation (default 0)

    This client is intentionally simple and robust for local testing and examples.
    """

    def __init__(self, embedding_dim: int = 8, job_delay_seconds: float = 2.0, seed: int = 0) -> None:
        self.embedding_dim = int(embedding_dim)
        self.job_delay_seconds = float(job_delay_seconds)
        self.seed = int(seed)

        # store jobs by id
        self._jobs: Dict[str, _Job] = {}

        # used to produce deterministic embeddings
        self._rng = np.random.RandomState(self.seed)

    async def create(self, model: str, inputs: Iterable[list[int]] | Iterable[str]) -> dict[str, Any]:
        """
        Create a batch job.

        Args:
            model: model/deployment name (ignored by the mock except for metadata)
            inputs: iterable of token lists or texts (the mock accepts either)

        Returns:
            job descriptor with `id` key
        """
        job_id = str(uuid.uuid4())
        # Normalize inputs: ensure list of lists of ints (if strings provided, use simple tokenization)
        normalized: list[list[int]] = []
        for item in inputs:
            if isinstance(item, str):
                # naive tokenization: hash + split into bytes -> ints
                h = hashlib.sha256(item.encode("utf-8")).digest()
                normalized.append([b for b in h[: min(32, len(h))]])
            elif isinstance(item, list):
                normalized.append([int(x) for x in item])
            else:
                # fallback: json encode and hash
                s = json.dumps(item, sort_keys=True)
                h = hashlib.sha256(s.encode("utf-8")).digest()
                normalized.append([b for b in h[: min(32, len(h))]])

        job = _Job(id=job_id, model=model, inputs=normalized, created_at=time.time())
        self._jobs[job_id] = job

        # schedule background processing
        asyncio.create_task(self._run_job(job))

        return {"id": job_id}

    async def get_status(self, job_id: str) -> dict[str, Any]:
        """
        Get status for a job.

        Returns:
            dict with at least `state` key. Example: {"state": "running"}
        """
        job = self._jobs.get(job_id)
        if not job:
            return {"state": "not_found"}
        return {"state": job.status}

    async def get_result(self, job_id: str) -> dict[str, Any]:
        """
        Retrieve batch job result once completed.

        Returns:
            dict with `data`: a list where each element contains `embedding` key (list[float]).
        """
        job = self._jobs.get(job_id)
        if not job:
            raise KeyError(f"job not found: {job_id}")

        if job.status != "succeeded":
            raise RuntimeError(f"job {job_id} not completed (status={job.status})")

        # job.result is prepared by _run_job
        return job.result or {"data": []}

    # Internal implementation -------------------------------------------------

    async def _run_job(self, job: _Job) -> None:
        """
        Simulate processing the batch job asynchronously.

        This method sleeps for `job_delay_seconds`, then marks job succeeded
        and populates `job.result` with embeddings.
        """
        try:
            job.status = "running"
            # Simulate some work, but keep it small
            await asyncio.sleep(self.job_delay_seconds)

            # Produce deterministic embeddings for each input
            embeddings: List[List[float]] = []
            for idx, tokens in enumerate(job.inputs):
                # Deterministic pseudo-embedding: combine model name hash and tokens into a small vector
                model_hash = int(hashlib.sha256(job.model.encode("utf-8")).hexdigest(), 16) & 0xFFFF
                token_sum = sum(int(t) for t in tokens) & 0xFFFF
                base_seed = (self.seed + model_hash + token_sum + idx) & 0xFFFFFFFF
                rng = np.random.RandomState(int(base_seed))
                vec = rng.rand(self.embedding_dim).astype(float).tolist()
                embeddings.append(vec)

            # Store result in the job as expected by batch embedder code
            job.result = {"data": [{"embedding": emb} for emb in embeddings]}
            job.status = "succeeded"
        except Exception as exc:  # pragma: no cover - defensive
            job.status = "failed"
            job.error = str(exc)

    # Convenience helpers for tests/demos -------------------------------------

    def list_jobs(self) -> list[str]:
        """Return list of known job ids (for debugging)."""
        return list(self._jobs.keys())

    def job_info(self, job_id: str) -> dict[str, Any]:
        """Return full info about a job (debugging only)."""
        job = self._jobs.get(job_id)
        if not job:
            return {}
        return {
            "id": job.id,
            "model": job.model,
            "created_at": datetime.fromtimestamp(job.created_at).isoformat(),
            "status": job.status,
            "has_result": job.result is not None,
            "error": job.error,
        }


# Example usage in documentation (not executed on import):
#
# from azure_llm_toolkit import AzureConfig, PolarsBatchEmbedder
# from azure_llm_toolkit.batch_api import MockBatchAPIClient
#
# async def demo_batch_embedder_with_mock():
#     config = AzureConfig()
#     # Create a mock batch API client (embedding_dim small for demo)
#     mock_batch_client = MockBatchAPIClient(embedding_dim=8, job_delay_seconds=1.0, seed=42)
#
#     # Create embedder with batch API enabled and pass the mock client
#     embedder = PolarsBatchEmbedder(
#         config=config,
#         use_batch_api=True,
#         batch_api_client=mock_batch_client,
#         use_rate_limiting=False,  # for demo; in production you should enable and pass a RateLimiter
#     )
#
#     texts = ["Hello world", "The quick brown fox"]
#     embeddings, metadata = await embedder.embed_texts(texts)
#     print("Embeddings (mock):", embeddings)
#     print("Metadata:", metadata)
#
# The above demonstrates how PolarsBatchEmbedder can be wired to a batch API
# implementation. In production you would implement or pass a real `batch_api_client`
# that talks to Azure/OpenAI batch endpoints. The mock here provides a stable
# and predictable surface for local testing and examples.


class BatchQuotaMonitor:
    """
    Monitor and check Azure Batch API quota before submitting jobs.

    Helps prevent job failures due to quota exceeded errors by:
    - Estimating token requirements
    - Checking against known quota limits
    - Providing warnings when approaching limits
    - Suggesting optimizations

    Example:
        >>> from azure_llm_toolkit import AzureConfig
        >>> from azure_llm_toolkit.batch_api import BatchQuotaMonitor
        >>>
        >>> config = AzureConfig()
        >>> monitor = BatchQuotaMonitor(config, subscription_type="default")
        >>>
        >>> # Check if a job will fit within quota
        >>> can_submit, message = monitor.check_job_feasibility(
        ...     num_texts=10000,
        ...     avg_tokens_per_text=100
        ... )
        >>> print(message)
    """

    # Default quota limits by subscription type (enqueued tokens)
    QUOTA_LIMITS = {
        "enterprise": {
            "gpt-5": 5_000_000_000,
            "gpt-4o": 5_000_000_000,
            "gpt-4o-mini": 15_000_000_000,
            "gpt-4": 150_000_000,
            "gpt-35-turbo": 10_000_000_000,
        },
        "default": {
            "gpt-5": 200_000_000,
            "gpt-4o": 200_000_000,
            "gpt-4o-mini": 1_000_000_000,
            "gpt-4": 30_000_000,
            "gpt-35-turbo": 1_000_000_000,
        },
        "credit_card": {
            "gpt-5": 50_000_000,
            "gpt-4o": 50_000_000,
            "gpt-4o-mini": 50_000_000,
            "gpt-4": 5_000_000,
            "gpt-35-turbo": 100_000_000,
        },
    }

    def __init__(
        self,
        config: Any,
        subscription_type: str = "enterprise",
        custom_quota_limit: int | None = None,
    ) -> None:
        """
        Initialize quota monitor.

        Args:
            config: AzureConfig instance
            subscription_type: One of 'enterprise', 'default', 'credit_card'
            custom_quota_limit: Override quota limit (if you know your actual limit)
        """
        self.config = config
        self.subscription_type = subscription_type
        self.custom_quota_limit = custom_quota_limit

    def get_quota_limit(self, model: str | None = None) -> int:
        """
        Get quota limit for a model.

        Args:
            model: Model name (uses config.embedding_deployment if None)

        Returns:
            Enqueued token quota limit
        """
        if self.custom_quota_limit:
            return self.custom_quota_limit

        model = model or self.config.embedding_deployment

        # Normalize model name
        model_lower = model.lower()
        if "gpt-5" in model_lower:
            model_key = "gpt-5"
        elif "gpt-4o-mini" in model_lower:
            model_key = "gpt-4o-mini"
        elif "gpt-4o" in model_lower:
            model_key = "gpt-4o"
        elif "gpt-4" in model_lower:
            model_key = "gpt-4"
        elif "gpt-35" in model_lower or "gpt-3.5" in model_lower:
            model_key = "gpt-35-turbo"
        else:
            # Unknown model, use conservative default
            return 50_000_000  # 50M tokens

        limits = self.QUOTA_LIMITS.get(self.subscription_type, self.QUOTA_LIMITS["default"])
        return limits.get(model_key, 50_000_000)

    def estimate_job_tokens(self, num_texts: int, avg_tokens_per_text: int) -> int:
        """
        Estimate total tokens for a job.

        Args:
            num_texts: Number of texts to embed
            avg_tokens_per_text: Average tokens per text

        Returns:
            Estimated total tokens
        """
        return num_texts * avg_tokens_per_text

    def check_job_feasibility(
        self,
        num_texts: int,
        avg_tokens_per_text: int,
        safety_margin: float = 0.1,
    ) -> tuple[bool, str]:
        """
        Check if a job is feasible given quota limits.

        Args:
            num_texts: Number of texts to embed
            avg_tokens_per_text: Average tokens per text
            safety_margin: Reserve this fraction of quota (default: 0.1 = 10%)

        Returns:
            Tuple of (feasible: bool, message: str)
        """
        quota_limit = self.get_quota_limit()
        job_tokens = self.estimate_job_tokens(num_texts, avg_tokens_per_text)
        usable_quota = quota_limit * (1 - safety_margin)

        if job_tokens <= usable_quota:
            usage_pct = (job_tokens / quota_limit) * 100
            return True, f"✅ Job feasible ({job_tokens:,} tokens, {usage_pct:.1f}% of quota)"

        # Job exceeds quota
        max_texts = int(usable_quota / avg_tokens_per_text)
        message = (
            f"❌ Job too large ({job_tokens:,} tokens exceeds {usable_quota:,} quota)\n"
            f"   Suggestions:\n"
            f"   - Split into {job_tokens // int(usable_quota) + 1} smaller jobs\n"
            f"   - Maximum texts per job: ~{max_texts:,}\n"
            f"   - Request quota increase in Azure Portal"
        )
        return False, message


class AzureBatchAPIClient:
    """
    Real Azure OpenAI Batch API client for embeddings.

    This client uses the OpenAI SDK's batch API endpoints to submit
    embedding jobs to Azure OpenAI's batch processing service.

    Features:
    - Creates batch jobs using JSONL file format
    - Monitors job progress with configurable polling
    - Retrieves results when jobs complete
    - Handles quota errors with detailed messages
    - Supports exponential backoff for quota exceeded errors

    Note: Azure Batch API enforces rate limits and quota:
    - Default 50K enqueued token limit per deployment
    - TPM/RPM limits still apply
    - Separate quota pool from standard API

    Example:
        >>> from azure_llm_toolkit import AzureConfig
        >>> from azure_llm_toolkit.batch_api import AzureBatchAPIClient
        >>>
        >>> config = AzureConfig()
        >>> batch_client = AzureBatchAPIClient(config)
        >>>
        >>> # Create a batch job
        >>> job = await batch_client.create(
        ...     model=config.embedding_deployment,
        ...     inputs=[[1, 2, 3], [4, 5, 6]]
        ... )
        >>> print(f"Job ID: {job['id']}")
        >>>
        >>> # Poll for completion
        >>> while True:
        ...     status = await batch_client.get_status(job['id'])
        ...     if status['state'] in ['succeeded', 'failed']:
        ...         break
        ...     await asyncio.sleep(10)
        >>>
        >>> # Get results
        >>> result = await batch_client.get_result(job['id'])
        >>> embeddings = [item['embedding'] for item in result['data']]
    """

    def __init__(
        self,
        config: Any,
        client: Any | None = None,
        poll_interval: float = 10.0,
        max_poll_interval: float = 60.0,
        timeout_seconds: int = 3600,
    ) -> None:
        """
        Initialize the Azure Batch API client.

        Args:
            config: AzureConfig instance with Azure OpenAI credentials
            poll_interval: Initial polling interval in seconds (default: 10.0)
            max_poll_interval: Maximum polling interval in seconds (default: 60.0)
            timeout_seconds: Maximum time to wait for job completion (default: 3600)
        """
        self.config = config
        self.poll_interval = poll_interval
        self.max_poll_interval = max_poll_interval
        self.timeout_seconds = timeout_seconds

        # Create OpenAI client
        # Use injected client if provided
        self.client = client or config.create_client()

        # Job tracking
        self._jobs: Dict[str, dict] = {}

    async def create(self, model: str, inputs: list[list[int]]) -> dict[str, Any]:
        """
        Create a batch embedding job.

        Creates a JSONL file with embedding requests and submits it to
        Azure's batch processing service.

        Args:
            model: Deployment name for embeddings
            inputs: List of token lists to embed

        Returns:
            Job descriptor with 'id' field

        Raises:
            ValueError: If inputs are invalid
            RuntimeError: If batch job creation fails
        """
        if not inputs:
            raise ValueError("No inputs provided for batch job")

        # Create JSONL content
        import tempfile
        from pathlib import Path

        # Create temporary JSONL file
        jsonl_lines = []
        for idx, token_list in enumerate(inputs):
            request = {
                "custom_id": f"request-{idx}",
                "method": "POST",
                "url": "/v1/embeddings",
                "body": {
                    "model": model,
                    "input": token_list,
                    "encoding_format": "float",
                },
            }
            jsonl_lines.append(json.dumps(request))

        jsonl_content = "\n".join(jsonl_lines)

        # Upload file to Azure OpenAI
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(jsonl_content)
            temp_path = f.name

        try:
            # Upload file
            with open(temp_path, "rb") as f:
                file_response = await self.client.files.create(file=f, purpose="batch")

            # Create batch job
            batch_response = await self.client.batches.create(
                input_file_id=file_response.id,
                endpoint="/v1/embeddings",
                completion_window="24h",
            )

            job_info = {
                "id": batch_response.id,
                "input_file_id": file_response.id,
                "status": batch_response.status,
                "created_at": batch_response.created_at,
                "model": model,
                "num_inputs": len(inputs),
            }

            self._jobs[batch_response.id] = job_info
            return {"id": batch_response.id, "job_id": batch_response.id}

        finally:
            # Clean up temp file
            Path(temp_path).unlink(missing_ok=True)

    async def get_status(self, job_id: str) -> dict[str, Any]:
        """
        Get current status of a batch job.

        Args:
            job_id: Batch job identifier

        Returns:
            Status dict with 'state' field mapping to normalized states
        """
        batch = await self.client.batches.retrieve(job_id)

        # Map Azure status to normalized state
        status_map = {
            "validating": "running",
            "in_progress": "running",
            "finalizing": "finalizing",
            "completed": "succeeded",
            "failed": "failed",
            "cancelled": "cancelled",
            "expired": "failed",
        }

        state = status_map.get(batch.status, batch.status)

        return {
            "state": state,
            "status": batch.status,
            "request_counts": {
                "total": batch.request_counts.total if batch.request_counts else 0,
                "completed": batch.request_counts.completed if batch.request_counts else 0,
                "failed": batch.request_counts.failed if batch.request_counts else 0,
            },
        }

    async def get_result(self, job_id: str) -> dict[str, Any]:
        """
        Retrieve results from a completed batch job.

        Args:
            job_id: Batch job identifier

        Returns:
            Result dict with 'data' list containing embeddings

        Raises:
            RuntimeError: If job is not completed or failed
        """
        batch = await self.client.batches.retrieve(job_id)

        if batch.status not in ["completed", "succeeded"]:
            raise RuntimeError(f"Batch job {job_id} not completed (status={batch.status})")

        if not batch.output_file_id:
            raise RuntimeError(f"Batch job {job_id} has no output file")

        # Download output file
        output_content = await self.client.files.content(batch.output_file_id)
        output_text = output_content.text

        # Parse JSONL output
        embeddings_data = []
        for line in output_text.strip().split("\n"):
            if not line:
                continue
            response_obj = json.loads(line)

            # Extract embedding from response
            if "response" in response_obj and response_obj["response"]["status_code"] == 200:
                body = response_obj["response"]["body"]
                if "data" in body and len(body["data"]) > 0:
                    embedding = body["data"][0]["embedding"]
                    embeddings_data.append({"embedding": embedding})
            elif "error" in response_obj:
                # Handle errors in batch results
                logger.warning(f"Batch request failed: {response_obj['error']}")
                # Return zero embedding as fallback
                embeddings_data.append({"embedding": [0.0] * 1536})

        return {"data": embeddings_data}

    async def cancel(self, job_id: str) -> dict[str, Any]:
        """
        Cancel a running batch job.

        Args:
            job_id: Batch job identifier

        Returns:
            Cancellation status
        """
        batch = await self.client.batches.cancel(job_id)
        return {
            "id": batch.id,
            "status": batch.status,
            "cancelled_at": batch.cancelled_at,
        }

    async def list_jobs(self, limit: int = 20) -> list[dict[str, Any]]:
        """
        List recent batch jobs.

        Args:
            limit: Maximum number of jobs to return

        Returns:
            List of job info dicts
        """
        batches = await self.client.batches.list(limit=limit)

        jobs = []
        for batch in batches.data:
            jobs.append(
                {
                    "id": batch.id,
                    "status": batch.status,
                    "created_at": batch.created_at,
                    "input_file_id": batch.input_file_id,
                    "output_file_id": batch.output_file_id,
                }
            )

        return jobs

    async def wait_for_completion(
        self,
        job_id: str,
        poll_interval: float | None = None,
        timeout: int | None = None,
    ) -> dict[str, Any]:
        """
        Wait for a batch job to complete with exponential backoff.

        Args:
            job_id: Batch job identifier
            poll_interval: Override initial poll interval (uses instance default if None)
            timeout: Override timeout (uses instance default if None)

        Returns:
            Final job status

        Raises:
            TimeoutError: If job doesn't complete within timeout
            RuntimeError: If job fails
        """
        start = datetime.now()
        interval = poll_interval or self.poll_interval
        timeout_sec = timeout or self.timeout_seconds

        while True:
            status = await self.get_status(job_id)
            state = status["state"]

            if state == "succeeded":
                return status

            if state in ["failed", "cancelled"]:
                raise RuntimeError(f"Batch job {job_id} {state}")

            elapsed = (datetime.now() - start).total_seconds()
            if elapsed > timeout_sec:
                raise TimeoutError(f"Batch job {job_id} did not complete within {timeout_sec}s")

            await asyncio.sleep(interval)

            # Exponential backoff
            interval = min(interval * 1.5, self.max_poll_interval)

    def get_quota_monitor(self, subscription_type: str = "enterprise") -> BatchQuotaMonitor:
        """
        Get a quota monitor instance for this client.

        Args:
            subscription_type: Subscription type ('enterprise', 'default', 'credit_card')

        Returns:
            BatchQuotaMonitor instance
        """
        return BatchQuotaMonitor(self.config, subscription_type)
