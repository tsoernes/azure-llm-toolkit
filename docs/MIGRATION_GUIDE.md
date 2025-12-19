# Migration Guide: From rag-mcp to azure-llm-toolkit

This guide helps you migrate from the integrated Azure client code in `rag-mcp` to the standalone `azure-llm-toolkit` library.

## Overview

The Azure OpenAI functionality has been extracted from `rag-mcp` into a reusable library called `azure-llm-toolkit`. This provides:

- **Reusability**: Use the same Azure client code across multiple projects
- **Maintainability**: Centralized updates and bug fixes
- **Testing**: Dedicated test suite for Azure functionality
- **Documentation**: Comprehensive docs and examples

## Installation

### Option 1: Install from PyPI (when published)

```bash
pip install azure-llm-toolkit
```

### Option 2: Install from Git

```bash
pip install git+https://github.com/tsoernes/azure-llm-toolkit.git
```

### Option 3: Install for development

```bash
git clone https://github.com/tsoernes/azure-llm-toolkit.git
cd azure-llm-toolkit
pip install -e .
```

## Migration Steps

### 1. Update Dependencies

**Before (pyproject.toml):**
```toml
dependencies = [
  "openai",
  "tiktoken",
  "tenacity",
  # ... other deps
]
```

**After (pyproject.toml):**
```toml
dependencies = [
  "azure-llm-toolkit>=0.1.0",
  # ... other deps (openai, tiktoken, tenacity now come via azure-llm-toolkit)
]
```

### 2. Update Imports

**Before:**
```python
from rag_mcp.azure_client import AzureClients
from rag_mcp.config import Settings, load_config
from rag_mcp.rate_limiter import RateLimiter, RateLimiterPool
```

**After:**
```python
from azure_llm_toolkit import (
    AzureLLMClient,
    AzureConfig,
    RateLimiter,
    RateLimiterPool,
    CostEstimator,
    InMemoryCostTracker,
)
```

### 3. Update Configuration

**Before:**
```python
from rag_mcp.config import Settings, load_config

settings = load_config()
```

**After:**
```python
from azure_llm_toolkit import AzureConfig

config = AzureConfig(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    endpoint=os.getenv("AZURE_ENDPOINT"),
    chat_deployment=os.getenv("AZURE_CHAT_DEPLOYMENT", "gpt-4o"),
    embedding_deployment=os.getenv("AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-3-large"),
)
```

### 4. Update Client Initialization

**Before:**
```python
from rag_mcp.azure_client import AzureClients

clients = AzureClients(settings)
```

**After:**
```python
from azure_llm_toolkit import AzureLLMClient, CostEstimator, InMemoryCostTracker

# Basic client
client = AzureLLMClient(config=config)

# Or with cost tracking
cost_tracker = InMemoryCostTracker(currency="kr")
cost_estimator = CostEstimator(currency="kr")

client = AzureLLMClient(
    config=config,
    cost_tracker=cost_tracker,
    cost_estimator=cost_estimator,
    enable_rate_limiting=True,
)
```

### 5. Update Embedding Calls

**Before:**
```python
embeddings = await clients.embed_texts(texts)
```

**After:**
```python
result = await client.embed_texts(texts)
embeddings = result.embeddings
usage = result.usage  # UsageInfo object with token counts
```

### 6. Update Chat Completion Calls

**Before:**
```python
response, usage = await clients.chat_completion_with_usage(
    messages=messages,
    system_prompt=system_prompt,
)
```

**After:**
```python
result = await client.chat_completion(
    messages=messages,
    system_prompt=system_prompt,
)
content = result.content
usage = result.usage  # UsageInfo object
```

### 7. Update Cost Estimation

**Before:**
```python
cost = clients.estimate_cost_kr(
    model="gpt-4o",
    tokens_input=1000,
    tokens_output=500,
    tokens_cached_input=100,
)
```

**After:**
```python
cost = client.cost_estimator.estimate_cost(
    model="gpt-4o",
    tokens_input=1000,
    tokens_output=500,
    tokens_cached_input=100,
)
```

### 8. Update Token Counting

**Before:**
```python
token_count = settings.token_len(text)
```

**After:**
```python
token_count = client.count_tokens(text)
# Or directly from config
token_count = config.count_tokens(text)
```

<!-- Query rewriting has been removed from the public API in this release.
If you previously relied on `rewrite_query`, please migrate your flow to
use your own rewriting logic or a separate service. -->

### 10. Update Metadata Extraction

**Before:**
```python
metadata, usage = await clients.extract_filename_metadata_with_usage(filename)
```

**After:**
```python
metadata = await client.extract_metadata_from_filename(
    filename=filename,
    track_cost=True,
)
```

## API Mapping Reference

| Old API (rag-mcp) | New API (azure-llm-toolkit) |
|-------------------|----------------------------|
| `AzureClients` | `AzureLLMClient` |
| `Settings` | `AzureConfig` |
| `embed_texts()` | `embed_texts()` (returns `EmbeddingResult`) |
| `chat_completion_with_usage()` | `chat_completion()` (returns `ChatCompletionResult`) |
| `estimate_cost_kr()` | `cost_estimator.estimate_cost()` |
| `estimate_embedding_cost_kr()` | `estimate_embedding_cost()` |
| `count_message_tokens()` | `count_message_tokens()` |
| `_token_len()` | `count_tokens()` |
| `extract_filename_metadata_with_usage()` | `extract_metadata_from_filename()` |
| `extract_content_metadata_with_usage()` | `extract_metadata_from_content()` |

## Breaking Changes

### 1. Return Types

Most methods now return dataclass objects instead of tuples:

**Before:**
```python
response, usage = await clients.chat_completion_with_usage(messages)
```

**After:**
```python
result = await client.chat_completion(messages)
response = result.content
usage = result.usage
```

### 2. Configuration Structure

Configuration is now handled by `AzureConfig` instead of `Settings`:

- Removed RAG-specific settings (persistence_dir, default_context_tokens, etc.)
- Simplified to Azure-specific configuration only
- Uses Pydantic for validation

### 3. Cost Tracking

Cost tracking is now opt-in via the `CostTracker` protocol:

**Before:**
```python
# Cost tracking was integrated into storage
```

**After:**
```python
# Explicit cost tracker
cost_tracker = InMemoryCostTracker()
client = AzureLLMClient(cost_tracker=cost_tracker)

# Get summary
summary = cost_tracker.get_summary()
```

### 4. Rate Limiting

Rate limiting configuration is now separate:

**Before:**
```python
# Rate limiting was implicit
```

**After:**
```python
# Explicit control
rate_limiter_pool = RateLimiterPool(default_rpm=3000, default_tpm=300000)
client = AzureLLMClient(
    rate_limiter_pool=rate_limiter_pool,
    enable_rate_limiting=True,
)
```

## Complete Example: Before and After

### Before (rag-mcp)

```python
from rag_mcp.config import load_config
from rag_mcp.azure_client import AzureClients

# Configuration
settings = load_config()

# Client
clients = AzureClients(settings)

# Embeddings
embeddings = await clients.embed_texts(["text1", "text2"])

# Chat
response, usage = await clients.chat_completion_with_usage(
    messages=[{"role": "user", "content": "Hello"}],
)

# Cost estimation
cost = clients.estimate_cost_kr(
    model="gpt-4o",
    tokens_input=usage.get("prompt_tokens", 0),
    tokens_output=usage.get("completion_tokens", 0),
)
```

### After (azure-llm-toolkit)

```python
from azure_llm_toolkit import (
    AzureConfig,
    AzureLLMClient,
    CostEstimator,
    InMemoryCostTracker,
)

# Configuration
config = AzureConfig()  # Loads from environment

# Client with cost tracking
cost_tracker = InMemoryCostTracker(currency="kr")
client = AzureLLMClient(
    config=config,
    cost_tracker=cost_tracker,
)

# Embeddings
result = await client.embed_texts(["text1", "text2"])
embeddings = result.embeddings

# Chat
result = await client.chat_completion(
    messages=[{"role": "user", "content": "Hello"}],
    track_cost=True,
)
response = result.content
usage = result.usage

# Cost summary
summary = cost_tracker.get_summary()
total_cost = summary["total_cost"]
```

## Gradual Migration Strategy

If you need to migrate gradually:

1. **Keep both dependencies temporarily**:
   ```toml
   dependencies = [
     "azure-llm-toolkit>=0.1.0",
     # Keep existing deps during migration
   ]
   ```

2. **Migrate module by module**: Start with isolated modules and work your way to core functionality

3. **Run tests frequently**: Ensure each migrated component works before moving to the next

4. **Update documentation**: Keep your project docs in sync with the migration

## Need Help?

- Check the [azure-llm-toolkit README](https://github.com/tsoernes/azure-llm-toolkit/blob/master/README.md)
- Review [examples](https://github.com/tsoernes/azure-llm-toolkit/tree/master/examples)
- Open an issue: https://github.com/tsoernes/azure-llm-toolkit/issues

## Benefits After Migration

1. **Cleaner separation of concerns**: RAG logic separate from Azure client logic
2. **Easier testing**: Test Azure functionality independently
3. **Reusability**: Use the same Azure client in multiple projects
4. **Better documentation**: Dedicated docs for Azure functionality
5. **Community contributions**: Easier for others to contribute Azure-specific improvements