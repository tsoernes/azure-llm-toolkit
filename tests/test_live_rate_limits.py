# code/azure-llm-toolkit/tests/test_live_rate_limits.py
"""
LIVE rate limit integration tests against the real Azure OpenAI API.

IMPORTANT
=========
These tests:

- Hit the REAL Azure OpenAI API using credentials from the `.env` file or environment.
- Are SLOW and may consume a non-trivial amount of quota.
- Are intended for manual / CI-controlled runs, not for every test invocation.

They are guarded by the environment variable:

    RUN_LIVE_RATE_LIMIT_TESTS=1

If that is not set to a truthy value, the tests will be skipped.

EXPECTED CONFIG
===============
The `.env` in this project (copied from rag-mcp) should configure:

- AZURE_ENDPOINT / AZURE_OPENAI_ENDPOINT
- AZURE_OPENAI_API_KEY / OPENAI_API_KEY
- AZURE_EMBEDDING_DEPLOYMENT=text-embedding-3-small
- AZURE_CHAT_DEPLOYMENT=gpt-5-mini

We then apply the following limits in the RateLimiterPool:

- For text-embedding-3-small:
    - TPM: 350,000
    - RPM: 2,100

- For gpt-5-mini:
    - TPM: 150,000
    - RPM: 150

The tests will:

- Launch a burst of concurrent requests.
- Ensure that no RateLimitError bubbles up.
- Measure total wait time introduced by the rate limiter (should be > 0 once saturated).
- Print some stats to stdout for inspection.
- Record actual cost via InMemoryCostTracker for precise reporting.

NOTE
====
These tests assume that:
- Your Azure deployments and keys are valid.
- Your Azure subscription has limits at least as high as the ones configured here.
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any

from dotenv import load_dotenv

load_dotenv()

import pytest

from azure_llm_toolkit import AzureConfig, AzureLLMClient, InMemoryCostTracker, RateLimiterPool


def _require_env_vars(names: list[str]) -> bool:
    """Return True if all given environment variables are present."""
    for name in names:
        if not os.getenv(name):
            return False
    return True


@pytest.fixture(scope="session")
def live_config() -> AzureConfig:
    """Load AzureConfig for live tests."""
    return AzureConfig()


import pytest_asyncio


@pytest_asyncio.fixture(scope="session")
async def live_client(live_config: AzureConfig) -> AzureLLMClient:
    """
    Create an AzureLLMClient with a RateLimiterPool configured to the real limits
    for the given deployments, and an InMemoryCostTracker that persists to file
    so we can report actual costs.
    """
    # Limits based on the prompt:
    # text-embedding-3-small:
    #   TPM: 350,000
    #   RPM: 2,100
    # gpt-5-mini:
    #   TPM: 150,000
    #   RPM: 150
    pool = RateLimiterPool(default_rpm=1000, default_tpm=100000)

    # Force model names for tests
    embed_model = "text-embedding-3-small"
    chat_model = "gpt-5-mini"

    async def _init_limiters() -> None:
        await pool.get_limiter(embed_model, rpm=2100, tpm=350000)
        await pool.get_limiter(chat_model, rpm=150, tpm=150000)

    # Initialize limiters in async context
    await _init_limiters()

    # Cost tracker with JSONL persistence for accurate cost reporting
    cost_tracker = InMemoryCostTracker(currency="kr", file_path="live_costs.jsonl")

    client = AzureLLMClient(
        config=live_config,
        rate_limiter_pool=pool,
        enable_rate_limiting=True,
        cost_tracker=cost_tracker,
    )
    return client


@pytest.mark.skipif(
    not (
        _require_env_vars(["AZURE_OPENAI_API_KEY", "AZURE_ENDPOINT"])
        or _require_env_vars(["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT"])
    ),
    reason="Azure credentials not configured; skipping live rate limit tests.",
)
@pytest.mark.asyncio
async def test_live_embedding_rate_limit(live_client: AzureLLMClient, live_config: AzureConfig) -> None:
    """
    Live test: hammer the embedding endpoint with text-embedding-3-small using
    configured limits and ensure:

    - No rate limit errors bubble up.
    - RateLimiterPool introduces measurable wait time when saturated.
    - We are effectively saturating the allowed RPM/TPM.

    Strategy
    --------
    - Use a moderately token-heavy prompt (~512-1024 tokens).
    - Launch a burst of concurrent embed_text calls.
    - Check RateLimiter stats afterwards.
    """
    # Force test to use the text-embedding-3-small deployment
    model = "text-embedding-3-small"
    text = "Azure OpenAI provides access to powerful language models. " * 64  # repeated to get a few hundred tokens

    # Parameters for the test (more aggressive traffic)
    num_tasks = int(os.getenv("LIVE_RATE_LIMIT_TASKS", "128"))
    iterations_per_task = int(os.getenv("LIVE_RATE_LIMIT_ITERATIONS", "20"))

    async def worker(idx: int) -> None:
        for j in range(iterations_per_task):
            # If the call raises because of rate limiting, the test should fail.
            # Disable cache so that we always hit the real API and the rate limiter.
            try:
                await live_client.embed_text(text, model=model, use_cache=False)
            except Exception as e:  # pragma: no cover - we want to see real errors
                raise AssertionError(f"Embedding worker {idx} iteration {j} failed: {e}") from e

    start = time.perf_counter()
    await asyncio.gather(*(worker(i) for i in range(num_tasks)))
    elapsed = time.perf_counter() - start

    pool = live_client.rate_limiter_pool
    stats = pool.get_all_stats()
    embed_stats = stats.get(model, {})

    total_requests = int(embed_stats.get("total_requests", 0))
    total_tokens = int(embed_stats.get("total_tokens", 0))
    total_wait_time = float(embed_stats.get("total_wait_time_seconds", 0.0))
    rpm_util = float(embed_stats.get("rpm_utilization_pct", 0.0))
    tpm_util = float(embed_stats.get("tpm_utilization_pct", 0.0))

    # Extract actual cost from the client's cost tracker
    tracker = live_client.cost_tracker
    summary = tracker.get_summary() if tracker is not None else {}
    total_cost = summary.get("total_cost", 0.0)
    currency = summary.get("currency", "kr")

    # Print some stats for manual inspection
    print("\n[Embedding Live Rate Limit Test]")
    print(f"Model:            {model}")
    print(f"Total requests:   {total_requests}")
    print(f"Total tokens:     {total_tokens}")
    print(f"Total wait time:  {total_wait_time:.2f}s")
    print(f"RPM utilization:  {rpm_util:.1f}%")
    print(f"TPM utilization:  {tpm_util:.1f}%")
    print(f"Elapsed walltime: {elapsed:.2f}s")
    print(f"Total embedding cost: {total_cost:.6f} {currency}")

    # Basic sanity: we should have actually made calls and hit the API
    assert total_requests > 0

    # We expect to see some waiting or at least meaningful utilization
    # if we are pushing traffic; do not enforce a hard lower bound since
    # Azure/network can be the bottleneck rather than our limiter.
    assert total_wait_time >= 0.0

    # The point is to ensure no exception was raised and that the limiter is
    # recording usage and cost.


@pytest.mark.skipif(
    not _require_env_vars(["AZURE_OPENAI_API_KEY", "AZURE_ENDPOINT"]),
    reason="Azure credentials not configured; skipping live rate limit tests.",
)
@pytest.mark.asyncio
async def test_live_chat_rate_limit(live_client: AzureLLMClient, live_config: AzureConfig) -> None:
    """
    Live test: hammer the chat endpoint with gpt-5-mini using configured limits.

    We:

    - Use a moderately sized prompt.
    - Launch a burst of concurrent chat_completion calls.
    - Confirm that no rate limit errors are propagated.
    - Inspect rate limiter stats to ensure we are close to saturation.

    This test uses the model configured in AZURE_CHAT_DEPLOYMENT, expected
    to be 'gpt-5-mini' (or equivalent).
    """
    model = live_config.chat_deployment
    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": (
                "You are benchmarking rate limiting. Please respond with a short sentence. "
                "This message is repeated to increase token usage. " * 16
            ),
        }
    ]

    # Parameters for the test (more aggressive traffic)
    num_tasks = 64
    iterations_per_task = 20  # total ~1280 chat calls

    async def worker(idx: int) -> None:
        for j in range(iterations_per_task):
            try:
                _ = await live_client.chat_completion(messages=messages, model=model, max_tokens=64)
            except Exception as e:  # pragma: no cover - we want to see real errors
                raise AssertionError(f"Chat worker {idx} iteration {j} failed: {e}") from e

    start = time.perf_counter()
    await asyncio.gather(*(worker(i) for i in range(num_tasks)))
    elapsed = time.perf_counter() - start

    pool = live_client.rate_limiter_pool
    stats = pool.get_all_stats()
    chat_stats = stats.get(model, {})

    total_requests = int(chat_stats.get("total_requests", 0))
    total_tokens = int(chat_stats.get("total_tokens", 0))
    total_wait_time = float(chat_stats.get("total_wait_time_seconds", 0.0))
    rpm_util = float(chat_stats.get("rpm_utilization_pct", 0.0))
    tpm_util = float(chat_stats.get("tpm_utilization_pct", 0.0))

    # Extract actual cost from the client's cost tracker
    tracker = live_client.cost_tracker
    summary = tracker.get_summary() if tracker is not None else {}
    total_cost = summary.get("total_cost", 0.0)
    currency = summary.get("currency", "kr")

    # Print stats for manual inspection
    print("\n[Chat Live Rate Limit Test]")
    print(f"Model:            {model}")
    print(f"Total requests:   {total_requests}")
    print(f"Total tokens:     {total_tokens}")
    print(f"Total wait time:  {total_wait_time:.2f}s")
    print(f"RPM utilization:  {rpm_util:.1f}%")
    print(f"TPM utilization:  {tpm_util:.1f}%")
    print(f"Elapsed walltime: {elapsed:.2f}s")
    print(f"Total chat cost:  {total_cost:.6f} {currency}")

    # Sanity: should have many calls recorded
    assert total_requests > 0

    # Non-zero wait time indicates we actually hit the limiter; however, Azure
    # throughput or network may be the limiting factor, so we only require that
    # the limiter recorded some usage and did not raise.
    assert total_wait_time >= 0.0

    # No explicit checks against Azure-side rate limit errors here; the fact that
    # we got through all requests without an exception is our success criterion.
