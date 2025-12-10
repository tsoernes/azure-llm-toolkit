#!/usr/bin/env python
"""
Performance benchmarking script for azure-llm-toolkit components.

This script provides a simple, self-contained way to benchmark the
following building blocks:

- AzureLLMClient: chat and embeddings
- ChatBatchRunner / EmbeddingBatchRunner: logical batch processing
- LogprobReranker: logprob-based reranking

The benchmarks are designed to be run either with a real Azure environment
(configured via environment variables) or in "dummy" mode using lightweight
mock clients, so you can get indicative numbers without incurring costs.

Usage
-----

From the repository root:

    # Run all benchmarks with real Azure (requires env vars configured)
    python benchmarks/benchmark_runners.py --mode real

    # Run all benchmarks with dummy clients (no network calls)
    python benchmarks/benchmark_runners.py --mode dummy

    # Only run chat benchmarks
    python benchmarks/benchmark_runners.py --mode real --bench chat

    # Run with specific sizes
    python benchmarks/benchmark_runners.py --mode real --bench chat --num-requests 100

Environment variables for real mode
-----------------------------------

When using --mode real, you must configure Azure like any other usage
of azure-llm-toolkit:

    AZURE_OPENAI_API_KEY
    AZURE_ENDPOINT
    AZURE_CHAT_DEPLOYMENT
    AZURE_EMBEDDING_DEPLOYMENT

The script will automatically construct AzureConfig() from the environment.

Notes
-----

- These benchmarks are not meant to be exhaustive micro-benchmarks, but
  rather indicative throughput/latency measurements for typical workloads.
- For accurate results, run multiple times and consider warm-up runs.
- Be mindful of your Azure rate limits and quotas.
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import time
from dataclasses import dataclass
from typing import Any, Callable, Coroutine, Iterable, List, Optional, Tuple

try:
    from azure_llm_toolkit import (
        AzureConfig,
        AzureLLMClient,
        ChatBatchItem,
        ChatBatchRunner,
        EmbeddingBatchItem,
        EmbeddingBatchRunner,
        LogprobReranker,
        RerankerConfig,
    )
except ImportError as e:  # pragma: no cover - import error is a usage issue
    raise SystemExit(
        "Failed to import azure_llm_toolkit. Make sure you have the package installed in this environment."
    ) from e


# =============================================================================
# Benchmark result dataclasses
# =============================================================================


@dataclass
class BenchmarkResult:
    name: str
    num_ops: int
    total_time: float
    avg_latency: float
    p95_latency: float
    p99_latency: float
    ops_per_second: float

    def pretty_print(self) -> None:
        print(f"\n=== {self.name} ===")
        print(f"Operations        : {self.num_ops}")
        print(f"Total time        : {self.total_time:.3f} s")
        print(f"Throughput        : {self.ops_per_second:.2f} ops/s")
        print(f"Average latency   : {self.avg_latency * 1000:.1f} ms")
        print(f"p95 latency       : {self.p95_latency * 1000:.1f} ms")
        print(f"p99 latency       : {self.p99_latency * 1000:.1f} ms")


# =============================================================================
# Dummy client for "dummy" mode (no network calls)
# =============================================================================


class DummyUsage:
    """Simple usage object mimicking API usage for dummy mode."""

    def __init__(self, prompt_tokens: int = 0, completion_tokens: int = 0) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = prompt_tokens + completion_tokens


class DummyChatResponse:
    """Minimal raw_response-like object for dummy mode."""

    def __init__(self, content: str, model: str = "dummy-model") -> None:
        self.choices = []
        self.model = model
        self.id = "dummy"
        self.object = "chat.completion"
        self.created = 0


# =============================================================================
# Helper functions
# =============================================================================


def compute_stats(latencies: List[float]) -> Tuple[float, float, float]:
    """Compute avg, p95, p99 from a list of latencies (seconds)."""
    if not latencies:
        return 0.0, 0.0, 0.0
    latencies_sorted = sorted(latencies)
    avg = sum(latencies_sorted) / len(latencies_sorted)
    p95 = latencies_sorted[int(len(latencies_sorted) * 0.95) - 1]
    p99 = latencies_sorted[int(len(latencies_sorted) * 0.99) - 1]
    return avg, p95, p99


async def run_concurrent(
    coro_factory: Callable[[], Coroutine[Any, Any, Any]],
    num: int,
    max_concurrent: int,
) -> Tuple[float, List[float]]:
    """
    Run `num` coroutines produced by `coro_factory` with concurrency limit.

    Returns:
        total_time (seconds), list of latencies (seconds) for each op.
    """
    sem = asyncio.Semaphore(max_concurrent)
    latencies: List[float] = []

    async def worker() -> None:
        async with sem:
            start = time.perf_counter()
            await coro_factory()
            end = time.perf_counter()
            latencies.append(end - start)

    start_total = time.perf_counter()
    await asyncio.gather(*(worker() for _ in range(num)))
    total_time = time.perf_counter() - start_total
    return total_time, latencies


# =============================================================================
# Benchmarks
# =============================================================================


async def benchmark_chat(
    client: AzureLLMClient,
    num_requests: int,
    max_concurrent: int,
    prompt: str,
) -> BenchmarkResult:
    """Benchmark chat_completion with the given client."""

    async def coro_factory():
        await client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            system_prompt="You are a helpful assistant.",
        )

    total_time, latencies = await run_concurrent(coro_factory, num_requests, max_concurrent)
    avg, p95, p99 = compute_stats(latencies)
    ops_per_sec = num_requests / total_time if total_time > 0 else 0.0
    return BenchmarkResult(
        name="chat_completion",
        num_ops=num_requests,
        total_time=total_time,
        avg_latency=avg,
        p95_latency=p95,
        p99_latency=p99,
        ops_per_second=ops_per_sec,
    )


async def benchmark_embeddings(
    client: AzureLLMClient,
    num_requests: int,
    max_concurrent: int,
    text: str,
) -> BenchmarkResult:
    """Benchmark embed_text with the given client."""

    async def coro_factory():
        await client.embed_text(text)

    total_time, latencies = await run_concurrent(coro_factory, num_requests, max_concurrent)
    avg, p95, p99 = compute_stats(latencies)
    ops_per_sec = num_requests / total_time if total_time > 0 else 0.0
    return BenchmarkResult(
        name="embed_text",
        num_ops=num_requests,
        total_time=total_time,
        avg_latency=avg,
        p95_latency=p95,
        p99_latency=p99,
        ops_per_second=ops_per_sec,
    )


async def benchmark_chat_batch_runner(
    client: AzureLLMClient,
    num_items: int,
    max_concurrent: int,
    prompt_template: str,
) -> BenchmarkResult:
    """Benchmark ChatBatchRunner with `num_items` items."""
    items: List[ChatBatchItem] = [
        ChatBatchItem(
            id=f"item-{i}",
            messages=[{"role": "user", "content": prompt_template.format(i=i)}],
        )
        for i in range(num_items)
    ]
    runner = ChatBatchRunner(client, max_concurrent=max_concurrent)

    start = time.perf_counter()
    results = await runner.run(items)
    total_time = time.perf_counter() - start

    # Compute per-item latency roughly as total_time / N because latency per item is not individually tracked.
    latencies = [total_time / num_items] * num_items if num_items > 0 else []
    avg, p95, p99 = compute_stats(latencies)
    ops_per_sec = num_items / total_time if total_time > 0 else 0.0

    # Sanity check to ensure tasks ran successfully
    succeeded = sum(1 for r in results if r.status.name == "SUCCESS")
    print(f"\nChatBatchRunner: {succeeded}/{num_items} items succeeded")

    return BenchmarkResult(
        name="chat_batch_runner",
        num_ops=num_items,
        total_time=total_time,
        avg_latency=avg,
        p95_latency=p95,
        p99_latency=p99,
        ops_per_second=ops_per_sec,
    )


async def benchmark_embedding_batch_runner(
    client: AzureLLMClient,
    num_items: int,
    batch_size: int,
    max_concurrent: int,
    text_template: str,
) -> BenchmarkResult:
    """Benchmark EmbeddingBatchRunner with `num_items` items."""
    items: List[EmbeddingBatchItem] = [
        EmbeddingBatchItem(id=f"doc-{i}", text=text_template.format(i=i)) for i in range(num_items)
    ]
    runner = EmbeddingBatchRunner(client, batch_size=batch_size, max_concurrent=max_concurrent)

    start = time.perf_counter()
    results = await runner.run(items)
    total_time = time.perf_counter() - start

    latencies = [total_time / num_items] * num_items if num_items > 0 else []
    avg, p95, p99 = compute_stats(latencies)
    ops_per_sec = num_items / total_time if total_time > 0 else 0.0

    succeeded = sum(1 for r in results if r.status.name == "SUCCESS")
    print(f"\nEmbeddingBatchRunner: {succeeded}/{num_items} items succeeded")

    return BenchmarkResult(
        name="embedding_batch_runner",
        num_ops=num_items,
        total_time=total_time,
        avg_latency=avg,
        p95_latency=p95,
        p99_latency=p99,
        ops_per_second=ops_per_sec,
    )


async def benchmark_reranker(
    client: AzureLLMClient,
    num_queries: int,
    docs_per_query: int,
    max_concurrent: int,
) -> BenchmarkResult:
    """
    Benchmark LogprobReranker scoring/reranking.
    NOTE: In 'real' mode this will exercise the model with logprobs=True
    which may incur non-trivial costs.
    """
    from azure_llm_toolkit import LogprobReranker, RerankerConfig  # lazy import to avoid circular issues

    reranker = LogprobReranker(client=client.client if hasattr(client, "client") else client, config=RerankerConfig())
    query = "What is machine learning?"
    documents = [f"Document {i}: Some content about machine learning and AI." for i in range(docs_per_query)]

    async def coro_factory():
        await reranker.rerank(query, documents, top_k=5)

    total_time, latencies = await run_concurrent(coro_factory, num_queries, max_concurrent)
    avg, p95, p99 = compute_stats(latencies)
    ops_per_sec = num_queries / total_time if total_time > 0 else 0.0

    return BenchmarkResult(
        name="logprob_reranker",
        num_ops=num_queries,
        total_time=total_time,
        avg_latency=avg,
        p95_latency=p95,
        p99_latency=p99,
        ops_per_second=ops_per_sec,
    )


# =============================================================================
# Orchestration
# =============================================================================


async def run_benchmarks(
    mode: str,
    benches: List[str],
    num_requests: int,
    docs_per_query: int,
    batch_size: int,
    max_concurrent: int,
) -> None:
    if mode not in {"real", "dummy"}:
        raise ValueError("mode must be 'real' or 'dummy'")

    config = AzureConfig()
    client = AzureLLMClient(config=config)
    if mode == "real":
        print("Running benchmarks in REAL mode (Azure environment).")
    else:
        print("Running benchmarks in DUMMY mode (no network calls).")

        async def fake_chat_completion(
            messages: list[dict[str, str]],
            system_prompt: str | None = None,
            model: str | None = None,
            temperature: float | None = None,
            max_tokens: int | None = None,
            reasoning_effort: str | None = None,
            response_format: Any | None = None,
            track_cost: bool = True,
            use_cache: bool = True,
            tools: list[dict[str, Any]] | None = None,
            tool_choice: Any | None = None,
        ):
            user_contents = [m.get("content", "") for m in messages if m.get("role") == "user"]
            last_user = user_contents[-1] if user_contents else ""
            content = f"[dummy] {last_user[:80]}"
            usage = DummyUsage(prompt_tokens=10, completion_tokens=5)
            return type(
                "DummyChatResult",
                (),
                {
                    "content": content,
                    "usage": usage,
                    "model": model or "dummy-model",
                    "finish_reason": "stop",
                    "raw_response": DummyChatResponse(content=content, model=model or "dummy-model"),
                },
            )()

        async def fake_embed_text(
            text: str,
            model: str | None = None,
            track_cost: bool = True,
            use_cache: bool = True,
        ) -> list[float]:
            return [float(len(text)), 0.0, 0.0]

        async def fake_embed_texts(
            texts: list[str],
            model: str | None = None,
            batch_size: int = 100,
            track_cost: bool = True,
        ):
            embeddings: list[list[float]] = []
            for idx, text in enumerate(texts):
                embeddings.append([float(len(text)), float(idx), 0.0])
            usage = DummyUsage(prompt_tokens=len(texts), completion_tokens=0)
            return type(
                "DummyEmbeddingResult",
                (),
                {
                    "embeddings": embeddings,
                    "usage": usage,
                    "model": model or "dummy-embed-model",
                },
            )()

        client.chat_completion = fake_chat_completion  # type: ignore[assignment]
        client.embed_text = fake_embed_text  # type: ignore[assignment]
        client.embed_texts = fake_embed_texts  # type: ignore[assignment]

    # Chat benchmark
    if "chat" in benches:
        prompt = "Explain the concept of large language models in two sentences."
        result = await benchmark_chat(client, num_requests=num_requests, max_concurrent=max_concurrent, prompt=prompt)
        result.pretty_print()

    # Embedding benchmark
    if "embed" in benches:
        text = "This is a sample document for embedding. " * 5
        result = await benchmark_embeddings(client, num_requests=num_requests, max_concurrent=max_concurrent, text=text)
        result.pretty_print()

    # Chat batch runner benchmark
    if "chat-batch" in benches:
        prompt_template = "User question #{i}: How does Azure OpenAI rate limiting work?"
        result = await benchmark_chat_batch_runner(
            client, num_items=num_requests, max_concurrent=max_concurrent, prompt_template=prompt_template
        )
        result.pretty_print()

    # Embedding batch runner benchmark
    if "embed-batch" in benches:
        text_template = "Document #{i}: This is text to be embedded for benchmarking."
        result = await benchmark_embedding_batch_runner(
            client,
            num_items=num_requests,
            batch_size=batch_size,
            max_concurrent=max_concurrent,
            text_template=text_template,
        )
        result.pretty_print()

    # Reranker benchmark (only sensible in real mode; in dummy mode it's still useful to measure overhead)
    if "reranker" in benches:
        result = await benchmark_reranker(
            client, num_queries=num_requests, docs_per_query=docs_per_query, max_concurrent=max_concurrent
        )
        result.pretty_print()


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Performance benchmarks for azure-llm-toolkit.")
    parser.add_argument(
        "--mode",
        choices=["real", "dummy"],
        default="dummy",
        help="Benchmark mode: 'real' uses Azure, 'dummy' uses local mock clients (no network).",
    )
    parser.add_argument(
        "--bench",
        choices=["chat", "embed", "chat-batch", "embed-batch", "reranker", "all"],
        default="all",
        nargs="*",
        help="Which benchmark(s) to run.",
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=50,
        help="Number of operations to perform for single-op benchmarks (chat/embed) and items for batch benchmarks.",
    )
    parser.add_argument(
        "--docs-per-query",
        type=int,
        default=20,
        help="Number of documents per query for reranker benchmark.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for embedding batch runner.",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=10,
        help="Maximum concurrency for relevant benchmarks.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    benches = args.bench
    if "all" in benches:
        benches = ["chat", "embed", "chat-batch", "embed-batch", "reranker"]

    asyncio.run(
        run_benchmarks(
            mode=args.mode,
            benches=benches,
            num_requests=args.num_requests,
            docs_per_query=args.docs_per_query,
            batch_size=args.batch_size,
            max_concurrent=args.max_concurrent,
        )
    )


if __name__ == "__main__":
    main()
