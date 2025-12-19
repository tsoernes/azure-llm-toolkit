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
]


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
