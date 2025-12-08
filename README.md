# Azure LLM Toolkit

A comprehensive Python library for working with Azure OpenAI APIs, featuring rate limiting, cost tracking, retry logic, and more.

## Features

- **Automatic Rate Limiting**: Built-in TPM (Tokens Per Minute) and RPM (Requests Per Minute) rate limiting using token bucket algorithm
- **Cost Tracking & Estimation**: Track and estimate costs for all API calls with configurable pricing
- **Retry Logic**: Exponential backoff retry logic for handling transient failures
- **Disk-Based Caching**: Cache embeddings and chat completions to disk to avoid redundant API calls and save costs
- **Batch Processing**: Efficient batch embedding with automatic splitting
- **High-Performance Batch Embedder**: Advanced Polars-based batch embedder for processing large datasets with intelligent batching and weighted averaging
- **Chat Completions**: Support for chat completions with reasoning models (GPT-4o, o1, etc.)
- **Query Rewriting**: LLM-powered query rewriting for better retrieval
- **Metadata Extraction**: Extract structured metadata from filenames and content
- **Token Counting**: Accurate token counting using tiktoken
- **Type-Safe**: Full type hints and Pydantic models for configuration

## Installation

```bash
pip install azure-llm-toolkit
```

Or install from source:

```bash
git clone https://github.com/torsteinsornes/azure-llm-toolkit.git
cd azure-llm-toolkit
pip install -e .
```

## Quick Start

### Basic Configuration

Set up your Azure OpenAI credentials via environment variables:

```bash
export AZURE_OPENAI_API_KEY="your-api-key"
export AZURE_ENDPOINT="https://your-resource.openai.azure.com"
export AZURE_CHAT_DEPLOYMENT="gpt-4o"
export AZURE_EMBEDDING_DEPLOYMENT="text-embedding-3-large"
```

Or use a `.env` file:

```env
AZURE_OPENAI_API_KEY=your-api-key
AZURE_ENDPOINT=https://your-resource.openai.azure.com
AZURE_CHAT_DEPLOYMENT=gpt-4o
AZURE_EMBEDDING_DEPLOYMENT=text-embedding-3-large
```

### Simple Usage

```python
import asyncio
from azure_llm_toolkit import AzureConfig, AzureLLMClient

async def main():
    # Create configuration (loads from environment variables)
    config = AzureConfig()
    
    # Create client
    client = AzureLLMClient(config=config)
    
    # Generate embeddings
    result = await client.embed_texts([
        "Hello, world!",
        "Azure OpenAI is powerful",
    ])
    print(f"Generated {len(result.embeddings)} embeddings")
    print(f"Usage: {result.usage.total_tokens} tokens")
    
    # Chat completion
    response = await client.chat_completion(
        messages=[
            {"role": "user", "content": "What is machine learning?"}
        ],
        system_prompt="You are a helpful AI assistant."
    )
    print(f"Response: {response.content}")
    print(f"Tokens: {response.usage.total_tokens}")

asyncio.run(main())
```

## Advanced Usage

### Rate Limiting

Rate limiting is enabled by default and prevents hitting Azure OpenAI quota limits:

```python
from azure_llm_toolkit import AzureLLMClient, RateLimiterPool

# Configure custom rate limits
rate_limiter_pool = RateLimiterPool(
    default_rpm=3000,  # Requests per minute
    default_tpm=300000  # Tokens per minute
)

client = AzureLLMClient(
    enable_rate_limiting=True,
    rate_limiter_pool=rate_limiter_pool
)

# The client will automatically throttle requests to stay within limits
for i in range(1000):
    result = await client.embed_text(f"Document {i}")
    print(f"Embedded document {i}")
```

### Cost Tracking

Track costs for all API operations:

```python
from azure_llm_toolkit import (
    AzureLLMClient,
    InMemoryCostTracker,
    CostEstimator
)

# Create cost tracker
cost_tracker = InMemoryCostTracker(currency="kr")

# Create cost estimator with custom pricing
cost_estimator = CostEstimator(currency="kr")
cost_estimator.set_model_pricing(
    model="gpt-4o",
    input_price=41.25,  # per 1M tokens
    output_price=165.00,
    cached_input_price=20.63
)

# Create client with cost tracking
client = AzureLLMClient(
    cost_tracker=cost_tracker,
    cost_estimator=cost_estimator
)

# Perform operations
await client.chat_completion(
    messages=[{"role": "user", "content": "Hello!"}],
    track_cost=True  # Enable cost tracking for this call
)

# Get cost summary
summary = cost_tracker.get_summary()
print(f"Total cost: {summary['total_cost']:.2f} {summary['currency']}")
print(f"By category: {summary['by_category']}")
print(f"By model: {summary['by_model']}")
```

### Custom Cost Tracker

Implement your own cost tracker (e.g., database-backed):

```python
from azure_llm_toolkit import CostTracker
from typing import Any

class DatabaseCostTracker(CostTracker):
    def __init__(self, db_connection):
        self.db = db_connection
    
    def record_cost(
        self,
        category: str,
        model: str,
        tokens_input: int,
        tokens_output: int,
        tokens_cached_input: int,
        currency: str,
        amount: float,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.db.execute(
            "INSERT INTO costs (category, model, tokens_input, tokens_output, amount) "
            "VALUES (?, ?, ?, ?, ?)",
            (category, model, tokens_input, tokens_output, amount)
        )
    
    def get_total_cost(self, category: str | None = None) -> float:
        if category:
            return self.db.query("SELECT SUM(amount) FROM costs WHERE category = ?", (category,))
        return self.db.query("SELECT SUM(amount) FROM costs")

# Use custom tracker
tracker = DatabaseCostTracker(my_db)
client = AzureLLMClient(cost_tracker=tracker)
```

### Batch Embeddings

Efficiently embed large numbers of texts:

```python
# Embed many documents with automatic batching
documents = [f"Document {i}" for i in range(10000)]

result = await client.embed_texts(
    texts=documents,
    batch_size=100,  # Process 100 at a time
    track_cost=True
)

print(f"Embedded {len(result.embeddings)} documents")
print(f"Total tokens: {result.usage.total_tokens}")
```

### High-Performance Batch Embedding with Polars

For large-scale embedding tasks, use the Polars-based batch embedder:

```python
import polars as pl
from azure_llm_toolkit import AzureConfig, PolarsBatchEmbedder

# Create DataFrame with texts
df = pl.DataFrame({
    "id": range(10000),
    "text": [f"Document {i} content..." for i in range(10000)]
})

# Configure embedder
config = AzureConfig()
embedder = PolarsBatchEmbedder(
    config=config,
    max_tokens_per_minute=450_000,  # Adjust based on your quota
    max_lists_per_query=1000,  # Texts per API call
)

# Embed entire DataFrame
result_df = await embedder.embed_dataframe(df, text_column="text")

# Result includes:
# - Original columns
# - text.tokens: Token IDs
# - text.token_count: Token counts
# - text.embedding: Embedding vectors

print(f"Embedded {len(result_df)} documents")
print(f"Total tokens: {result_df['text.token_count'].sum():,}")

# Save to Parquet for later use
result_df.write_parquet("embeddings.parquet")
```

Features of the Polars batch embedder:
- **Intelligent batching**: Automatically creates batches based on token and list limits
- **Weighted averaging**: Handles texts exceeding token limits by splitting and averaging
- **Incremental processing**: Only embed new documents (skip existing embeddings)
- **Progress tracking**: Built-in tqdm progress bars
- **High performance**: Uses multiprocessing for tokenization and Polars for data operations
- **Disk caching**: Optional saving of intermediate results

### Disk-Based Caching

Save costs and improve performance by caching LLM responses:

```python
from azure_llm_toolkit import AzureConfig, AzureLLMClient, CacheManager

# Create client with caching enabled (default)
config = AzureConfig()
client = AzureLLMClient(config=config, enable_cache=True)

texts = ["Hello world", "Azure OpenAI", "Machine learning"]

# First call - hits the API
result1 = await client.embed_texts(texts, use_cache=True)
print(f"Generated {len(result1.embeddings)} embeddings")

# Second call - retrieves from cache (no API call, no cost!)
result2 = await client.embed_texts(texts, use_cache=True)
print(f"Retrieved {len(result2.embeddings)} embeddings from cache")

# Works with chat completions too
messages = [{"role": "user", "content": "What is AI?"}]
response1 = await client.chat_completion(messages, use_cache=True)  # API call
response2 = await client.chat_completion(messages, use_cache=True)  # From cache

# Get cache statistics
cache_manager = client.cache_manager
stats = cache_manager.get_stats()
print(f"Cache size: {stats['total_size_mb']:.2f} MB")
print(f"Total files: {stats['total_files']}")

# Clear cache when needed
cache_manager.clear_all()
```

Features of the caching system:
- **Automatic caching**: Embeddings and chat completions are automatically cached
- **Content-based**: Cache keys based on content, model, and parameters
- **Partial hits**: Smart handling of partial cache hits in batch operations
- **Cost savings**: Avoid redundant API calls and reduce costs
- **Custom directories**: Configure cache location
- **Easy management**: Get stats and clear cache as needed

### Query Rewriting

Improve retrieval by rewriting queries:

```python
# Rewrite a query for better search results
original_query = "how to train ml model"

rewrite_result = await client.rewrite_query(original_query)

print(f"Original: {rewrite_result.original}")
print(f"Rewritten: {rewrite_result.rewritten}")
# Output:
# Original: how to train ml model
# Rewritten: What are the best practices and step-by-step procedures 
#            for training a machine learning model?
```

### Metadata Extraction

Extract structured metadata from documents:

```python
# Extract metadata from filename
metadata = await client.extract_metadata_from_filename(
    "2024-Q4-Financial-Report-Final.pdf"
)
print(metadata)
# Output: {'title': 'Financial Report', 'date': '2024-Q4', 
#          'document_type': 'report', 'status': 'final'}

# Extract metadata from content
content = """
Title: Machine Learning Best Practices
Author: John Doe
Date: 2024-12-01

This document covers best practices for ML...
"""

metadata = await client.extract_metadata_from_content(
    content=content,
    filename="ml-best-practices.md"
)
print(metadata)
# Output: {'title': 'Machine Learning Best Practices', 
#          'author': 'John Doe', 'date': '2024-12-01', ...}
```

### RAG-Style Question Answering

Generate answers with context:

```python
context = """
Azure OpenAI Service provides REST API access to OpenAI's powerful 
language models including GPT-4, GPT-3.5-Turbo, and Embeddings models.
"""

question = "What models does Azure OpenAI provide?"

result = await client.generate_answer(
    question=question,
    context=context,
    system_prompt="Answer based on the context provided."
)

print(result.content)
# Output: Azure OpenAI Service provides access to GPT-4, GPT-3.5-Turbo, 
#         and Embeddings models.
```

### Reasoning Models (o1, GPT-5)

Use reasoning models with appropriate settings:

```python
# Use reasoning effort parameter for o1/GPT-5 models
response = await client.chat_completion(
    messages=[
        {"role": "user", "content": "Solve this complex problem: ..."}
    ],
    model="o1-preview",
    reasoning_effort="high",  # or "low", "medium"
)

print(f"Answer: {response.content}")
print(f"Finish reason: {response.finish_reason}")
```

### Token Counting

Estimate tokens before making API calls:

```python
# Count tokens in text
text = "This is a sample text for token counting."
token_count = client.count_tokens(text)
print(f"Text has {token_count} tokens")

# Count tokens in messages
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is AI?"}
]
token_count = client.count_message_tokens(messages)
print(f"Messages have {token_count} tokens")

# Estimate cost before calling
cost = client.estimate_chat_cost(
    messages=messages,
    estimated_output_tokens=500
)
print(f"Estimated cost: {cost:.4f} kr")
```

### Custom Configuration

Override default configuration:

```python
from pathlib import Path

config = AzureConfig(
    api_key="your-key",
    endpoint="https://your-resource.openai.azure.com",
    api_version="2024-12-01-preview",
    chat_deployment="gpt-4o",
    embedding_deployment="text-embedding-3-large",
    timeout_seconds=120,
    max_retries=10,
    tokenizer_model="gpt-4o",
    cache_dir=Path.home() / ".cache" / "azure-llm-toolkit"
)

client = AzureLLMClient(config=config)
```

## Configuration Reference

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `AZURE_OPENAI_API_KEY` | Azure OpenAI API key | Required |
| `AZURE_ENDPOINT` | Azure OpenAI endpoint URL | Required |
| `AZURE_API_VERSION` | API version | `2024-12-01-preview` |
| `AZURE_CHAT_DEPLOYMENT` | Chat model deployment name | `gpt-4o` |
| `AZURE_EMBEDDING_DEPLOYMENT` | Embedding model deployment name | `text-embedding-3-large` |
| `AZURE_TIMEOUT_SECONDS` | Request timeout in seconds | `60` |
| `AZURE_MAX_RETRIES` | Maximum retry attempts | `5` |
| `TOKENIZER_MODEL` | Tokenizer model name | `gpt-4o` |
| `FORCE_EMBED_DIM` | Force embedding dimension (for testing) | None |

### Default Pricing (NOK per 1M tokens)

| Model | Input | Output | Cached Input |
|-------|-------|--------|--------------|
| gpt-4o | 41.25 | 165.00 | 20.63 |
| gpt-4o-mini | 1.24 | 4.95 | 0.62 |
| gpt-4-turbo | 82.50 | 247.50 | 41.25 |
| o1-preview | 123.75 | 495.00 | 61.88 |
| o1-mini | 24.75 | 99.00 | 12.38 |
| text-embedding-3-large | 1.03 | - | - |
| text-embedding-3-small | 0.17 | - | - |

## Architecture

### Rate Limiting

The library implements a token bucket algorithm for rate limiting:

- **TPM (Tokens Per Minute)**: Limits total tokens processed per minute
- **RPM (Requests Per Minute)**: Limits number of requests per minute
- **Automatic throttling**: Requests are queued and delayed as needed
- **Per-model limits**: Different rate limits for different models

### Retry Logic

Automatic retry with exponential backoff for:

- `APIConnectionError`: Network connectivity issues
- `RateLimitError`: API rate limit errors
- `APITimeoutError`: Request timeout errors
- `APIStatusError`: Server-side errors

Retry configuration:
- Initial delay: 1 second
- Maximum delay: 10 seconds
- Maximum attempts: 5

### Cost Tracking

Cost tracking supports:

- **Category-based tracking**: Separate costs by category (embedding, chat, etc.)
- **Model-based tracking**: Track costs per model
- **Token breakdown**: Input, output, and cached tokens
- **Custom implementations**: Implement your own `CostTracker` protocol

## Development

### Setup

```bash
# Clone repository
git clone https://github.com/torsteinsornes/azure-llm-toolkit.git
cd azure-llm-toolkit

# Install with development dependencies
pip install -e ".[dev]"
```

### Testing

```bash
# Run tests
pytest

# Run with coverage
pytest --cov=azure_llm_toolkit --cov-report=html

# Type checking
basedpyright src/
mypy src/
```

### Code Quality

```bash
# Format code
ruff format .

# Lint code
ruff check .

# Fix linting issues
ruff check --fix .
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass (`pytest`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built on top of the official [OpenAI Python SDK](https://github.com/openai/openai-python)
- Uses [tiktoken](https://github.com/openai/tiktoken) for accurate token counting
- Inspired by the need for robust Azure OpenAI client tooling

## Support

For issues, questions, or contributions, please:

- Open an issue on [GitHub Issues](https://github.com/torsteinsornes/azure-llm-toolkit/issues)
- Check existing issues for solutions
- Provide detailed information about your environment and use case

## Changelog

### 0.1.0 (2024-12-08)

- Initial release
- Rate limiting with TPM/RPM support
- Cost tracking and estimation
- Batch embedding support
- Chat completions with reasoning models
- Query rewriting
- Metadata extraction
- Token counting utilities