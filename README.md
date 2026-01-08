azure-llm-toolkit/README.md#L1-220
# Azure LLM Toolkit (v0.1.5)

A Python toolkit that wraps Azure OpenAI interactions with production-friendly features:
- Rate limiting (RPM / TPM)
- Cost estimation & pluggable cost tracking
- Retry logic and circuit-breaker patterns
- Disk-based caching for embeddings & chat completions
- Batch embedding (Polars-based high-performance embedder)
- Utilities: token counting, streaming, reranking helpers

This repository is packaged as `azure-llm-toolkit` (see `pyproject.toml`, version 0.1.5).

---
## Key components (API surface)

Top-level imports you will typically use:

- `AzureConfig` — configuration loader for environment / constructor-based config
- `AzureLLMClient` — async client with:
  - `embed_text(...)` — embed a single text (async)
  - `chat_completion(...)` — chat completion (async)
  - `chat_completion_stream(...)` — streaming chat completions (async generator)
  - token counting helpers: `count_tokens(...)`, `count_message_tokens(...)`
  - cost estimation helpers: `estimate_embedding_cost(...)`, `estimate_chat_cost(...)`
- `AzureLLMClientSync` — synchronous wrapper that runs the async client in an event loop
- `PolarsBatchEmbedder` — high-performance batch embedder for large datasets (async)
- `CostEstimator`, `CostTracker`, `InMemoryCostTracker` — cost estimation and tracking
- `RateLimiter`, `RateLimiterPool` — rate limiting primitives
- `CacheManager`, `EmbeddingCache`, `ChatCache` — disk-based caches for embeddings / chat responses
- `LogprobReranker`, `create_reranker` — logprob-based reranker utilities
- `detect_embedding_dimension(config)` — probe or read cached embedding dimensionality

(See the package `azure_llm_toolkit.__init__` for the full exported list.)

---
## Installation

Install from PyPI:

```/dev/null/example.md#L1-4
pip install azure-llm-toolkit
```

Or install editable from source:

```/dev/null/example.md#L1-4
git clone https://github.com/torsteinsornes/azure-llm-toolkit.git
cd azure-llm-toolkit
pip install -e .
```

Development extras:

```/dev/null/example.md#L1-4
pip install -e ".[dev]"
```

---
## Configuration

The library loads configuration from environment variables by default. Common variables:

- `AZURE_OPENAI_API_KEY` (or `OPENAI_API_KEY`) — REQUIRED
- `AZURE_ENDPOINT` (or `AZURE_OPENAI_ENDPOINT`) — REQUIRED (e.g. `https://your-resource.openai.azure.com`)
- `AZURE_API_VERSION` — default: `2024-12-01-preview`
- `AZURE_CHAT_DEPLOYMENT` — default: `gpt-5-mini`
- `AZURE_RERANKER_DEPLOYMENT` — default: `gpt-4o-east-US`
- `AZURE_EMBEDDING_DEPLOYMENT` — default: `text-embedding-3-large`
- `AZURE_TIMEOUT_SECONDS` — request timeout in seconds (default: `None` = infinite, recommended for reasoning models)
- `AZURE_MAX_RETRIES` — default: `5`
- `TOKENIZER_MODEL` — model used by tiktoken for token counting (defaults to chat deployment)
- `FORCE_EMBED_DIM` — optional integer to force embedding dim (useful in tests/offline)

You can also pass these values directly when constructing `AzureConfig(...)`.

---
## Quick start — async (basic)

Below are succinct examples showing common workflows.

Embed a single text (async):

```/dev/null/example.md#L1-40
import asyncio
from azure_llm_toolkit import AzureConfig, AzureLLMClient

async def main():
    config = AzureConfig()  # loads from env by default
    client = AzureLLMClient(config=config)

    emb = await client.embed_text("Hello, world!")
    print(f"Embedding length: {len(emb)}")
    print(f"First 8 dims: {emb[:8]}")

asyncio.run(main())
```

Chat completion (async):

```/dev/null/example.md#L1-80
import asyncio
from azure_llm_toolkit import AzureConfig, AzureLLMClient

async def main():
    config = AzureConfig()
    client = AzureLLMClient(config=config)

    messages = [{"role": "user", "content": "Explain supervised learning in simple terms."}]
    result = await client.chat_completion(messages=messages, system_prompt="You are a helpful assistant.")
    print("Response:")
    print(result.content)
    print("Usage (tokens):", result.usage.total_tokens)

asyncio.run(main())
```

Streaming chat completion:

```/dev/null/example.md#L1-80
import asyncio
from azure_llm_toolkit import AzureConfig, AzureLLMClient

async def stream_example():
    client = AzureLLMClient(AzureConfig())
    async for chunk in client.chat_completion_stream(
        messages=[{"role":"user","content":"Tell me a short story about a robot."}],
        system_prompt="You are a creative storyteller."
    ):
        print(chunk, end="", flush=True)

asyncio.run(stream_example())
```

---
## Quick start — batch embeddings (Polars)

When embedding large corpora, use `PolarsBatchEmbedder` which tokenizes in parallel, batches intelligently, and supports weighted averaging for splits.

The batch embedder uses a **dual rate-limiting approach**:
- Built-in batching with sleep delays between batches (always active)
- Optional integration with `RateLimiter` for coordinated throttling (set `use_rate_limiting=True`)

Example (async):

```/dev/null/example.md#L1-200
import asyncio
import polars as pl
from azure_llm_toolkit import AzureConfig, PolarsBatchEmbedder

async def main():
    config = AzureConfig()
    embedder = PolarsBatchEmbedder(config=config, max_tokens_per_minute=450_000, max_lists_per_query=1024)

    df = pl.DataFrame({"id": list(range(1000)), "text": [f"Document {i}" for i in range(1000)]})
    result_df = await embedder.embed_dataframe(df, text_column="text", verbose=True)

    # result_df includes columns: text, text.token_count, text.embedding
    print("Embedded rows:", len(result_df))

asyncio.run(main())
```

For more examples including rate limiter integration, cost tracking, and handling large datasets, see `examples/polars_batch_embedder_comprehensive.py`.

---
## Caching

If enabled, the client caches embeddings and chat completions on disk (content-based keys). Example usage:

```/dev/null/example.md#L1-80
from azure_llm_toolkit import AzureConfig, AzureLLMClient

config = AzureConfig()
client = AzureLLMClient(config=config, enable_cache=True)

# First call — hits API
emb1 = await client.embed_text("Cache demo text", use_cache=True)

# Second call — should be a cache hit
emb2 = await client.embed_text("Cache demo text", use_cache=True)
```

You can access cache statistics via `client.cache_manager.get_stats()` when `CacheManager` is used.

---
## Rate limiting

By default, `AzureLLMClient` creates a `RateLimiterPool` to throttle requests. You can provide a custom pool:

```/dev/null/example.md#L1-40
from azure_llm_toolkit import AzureConfig, AzureLLMClient, RateLimiterPool

pool = RateLimiterPool(default_rpm=3000, default_tpm=300_000)
client = AzureLLMClient(config=AzureConfig(), rate_limiter_pool=pool, enable_rate_limiting=True)
```

The Polars embedder also respects token/list limits configured at construction.

---
## Cost estimation & tracking

Use `CostEstimator` to estimate costs before making calls; use `InMemoryCostTracker` (or implement `CostTracker`) to record costs after calls.

Estimate cost for a chat:

```/dev/null/example.md#L1-40
from azure_llm_toolkit import AzureConfig, AzureLLMClient, CostEstimator

config = AzureConfig()
client = AzureLLMClient(config=config)
est = client.estimate_chat_cost(messages=[{"role":"user","content":"Hello"}], estimated_output_tokens=200)
print("Estimated cost:", est)
```

Record costs automatically by passing a `CostTracker` to the client (example in docs and tests). `InMemoryCostTracker` can be used for quick local tracking.

---
## Reranker (logprob-based)

The toolkit includes a logprob-based reranker that uses token log probabilities to produce calibrated relevance scores. Typical flow:

- Retrieve candidate docs via vector DB
- Use `LogprobReranker` / `create_reranker` to score documents
- Optionally rerank and return top-K

Example (async):

```/dev/null/example.md#L1-120
from azure_llm_toolkit import AzureConfig, AzureLLMClient
from azure_llm_toolkit.reranker import create_reranker

config = AzureConfig()
client = AzureLLMClient(config=config)

reranker = create_reranker(client=client, model="gpt-4o")
results = await reranker.rerank("What is machine learning?", ["Doc A text", "Doc B text"], top_k=3)

for r in results:
    print(r.score, r.document)
```

Note: the reranker requires a model that supports logprobs.

---
## Synchronous usage (legacy code)

The `AzureLLMClientSync` provides blocking wrappers:

```/dev/null/example.md#L1-80
from azure_llm_toolkit import AzureConfig, AzureLLMClientSync

client = AzureLLMClientSync(config=AzureConfig())
embedding = client.embed_text("Hello sync world")
response = client.chat_completion(messages=[{"role":"user","content":"Hi"}])
print(response.content)
```

(Under the hood this runs the async client in an event loop or a background thread if already inside an event loop.)

---
## Utilities

- `detect_embedding_dimension(config)` — probe the configured embedding deployment to detect vector dimensionality (with caching).
- `AzureConfig.count_tokens(...)` and client helpers for token counting.
- Streaming sinks, tools for function-calling integrations, health checks, metrics collector interfaces (Prometheus / OpenTelemetry helpers), and more — see `src/azure_llm_toolkit/` for modules and docstrings.

---
## Development & testing

Install dev dependencies:

```/dev/null/example.md#L1-4
pip install -e ".[dev]"
```

Run tests:

```/dev/null/example.md#L1-4
pytest -q
```

Type checking:

```/dev/null/example.md#L1-4
basedpyright src/
mypy src/
```

Formatting & linting:

```/dev/null/example.md#L1-4
ruff format .
ruff check .
```

---
## Contributing

1. Fork the repo
2. Create a branch (`git checkout -b feature/awesome`)
3. Add tests for new functionality
4. Ensure tests and static checks pass
5. Open a PR with a clear description

See `CONTRIBUTING.md` for more details.

---
## License

MIT — see the `LICENSE` file.

---
## Where to look next (code entry points)

- `src/azure_llm_toolkit/client.py` — async client implementation and chat/embedding primitives
- `src/azure_llm_toolkit/config.py` — configuration and tokenization helpers
- `src/azure_llm_toolkit/batch_embedder.py` — `PolarsBatchEmbedder` implementation
- `src/azure_llm_toolkit/sync_client.py` — synchronous wrapper
- `src/azure_llm_toolkit/reranker.py` — reranking utilities
- `src/azure_llm_toolkit/cache.py` — caching primitives

If you need curated examples, the `examples/` directory contains runnable demos for caching, batching, reranking, and Prometheus / dashboard integrations.

---

If you want, I can:
- Open/produce a one-file example matching your exact environment (async or sync),
- Or update the examples/ directory to include a minimal runnable script demonstrating embed + chat + caching + cost tracking with your preferred settings.