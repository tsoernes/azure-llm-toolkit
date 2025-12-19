# Batch API vs Standard API: Decision Guide

## Overview

This guide helps you decide whether to use Azure OpenAI's Batch API or the standard real-time API for your embedding workloads.

## Quick Decision Tree

```
START
  |
  ├─> Need results in < 1 minute? ──> YES ──> Use STANDARD API
  |                                  |
  |                                  NO
  |                                  |
  ├─> Processing > 100K texts? ──────> YES ──> Consider BATCH API
  |                                  |
  |                                  NO
  |                                  |
  ├─> Cost savings critical? ────────> YES ──> Consider BATCH API
  |                                  |
  |                                  NO
  |                                  |
  └─> Use STANDARD API (simpler, faster feedback)
```

## Detailed Comparison

| Feature | Standard API | Batch API |
|---------|-------------|-----------|
| **Response Time** | < 1 second | Up to 24 hours (typically 1-8 hours) |
| **Cost** | Full price | 50% discount |
| **Rate Limits** | Standard TPM/RPM | Separate enqueued token quota |
| **Maximum Requests** | Unlimited (subject to rate limits) | 100,000 per file |
| **File Size Limit** | N/A | 200 MB per file |
| **Queue Limit** | N/A | 50K tokens (default, can increase) |
| **Complexity** | Simple (direct API calls) | Complex (file upload, polling, parsing) |
| **Retry Logic** | Built-in with toolkit | Manual implementation needed |
| **Progress Tracking** | Immediate | Polling required |
| **Use Case** | Interactive, real-time | Bulk, offline processing |

## When to Use Standard API

✅ **Use standard API when:**

1. **You need immediate results**
   - Interactive applications
   - Real-time search
   - User-facing features
   - Prototyping and development

2. **You have smaller datasets**
   - < 10,000 texts
   - Processing time < 10 minutes
   - Memory constraints are not an issue

3. **You want simplicity**
   - Simple async/await pattern
   - Built-in retry logic
   - Immediate error feedback
   - Easy debugging

4. **Your workload is unpredictable**
   - Ad-hoc queries
   - Variable batch sizes
   - Frequent schema changes

**Example workloads:**
- Search indexing for small document sets
- Real-time semantic search
- Interactive RAG applications
- Development and testing

## When to Use Batch API

✅ **Use Batch API when:**

1. **You can wait for results**
   - Overnight processing acceptable
   - Non-time-sensitive workloads
   - Scheduled batch jobs

2. **You have large datasets**
   - > 100,000 texts
   - Multi-million document corpora
   - Full database re-indexing
   - Data warehouse processing

3. **Cost is a major concern**
   - 50% discount is significant
   - Budget-constrained projects
   - Regular bulk processing

4. **You need to bypass standard rate limits**
   - Hitting TPM/RPM limits frequently
   - Separate quota pool needed
   - Large parallel workloads

**Example workloads:**
- Nightly embedding refreshes
- Historical data processing
- Archive indexing
- Data migration projects
- Research datasets

## Important: Batch API Limitations

⚠️ **Critical things to know:**

### 1. Rate Limits Still Apply
- Batch API has **separate quota** but is **NOT unlimited**
- Default: **50K enqueued token limit** across all active jobs
- TPM/RPM limits still enforced
- Jobs fail if quota exceeded

### 2. Quota Management Required
```python
# Check quota BEFORE submitting
from azure_llm_toolkit.batch_api import BatchQuotaMonitor

monitor = BatchQuotaMonitor(config, subscription_type="default")
can_submit, msg = monitor.check_job_feasibility(
    num_texts=100_000,
    avg_tokens_per_text=100
)

if not can_submit:
    print(msg)  # Get recommendations
```

### 3. No Real-Time Feedback
- No immediate error messages
- Must poll for status
- Errors only visible after job completes
- Debugging is slower

### 4. File Management Overhead
- Create JSONL files
- Upload to Azure
- Track file IDs
- Clean up after completion
- 500 file limit (10,000 with expiration)

## Cost Comparison Calculator

### Standard API Pricing
- **text-embedding-3-small**: $0.00002 per 1K tokens
- **text-embedding-3-large**: $0.00013 per 1K tokens

### Batch API Pricing (50% discount)
- **text-embedding-3-small**: $0.00001 per 1K tokens
- **text-embedding-3-large**: $0.000065 per 1K tokens

### Example: 1 Million Documents

Assumptions:
- 1,000,000 documents
- 100 tokens per document
- Total: 100,000,000 tokens (100M)

**Using text-embedding-3-small:**

| API Type | Cost per 1M tokens | Total Cost | Savings |
|----------|-------------------|------------|---------|
| Standard | $20 | $2,000 | - |
| Batch | $10 | $1,000 | $1,000 (50%) |

**Using text-embedding-3-large:**

| API Type | Cost per 1M tokens | Total Cost | Savings |
|----------|-------------------|------------|---------|
| Standard | $130 | $13,000 | - |
| Batch | $65 | $6,500 | $6,500 (50%) |

### Break-even Analysis

The batch API is worth it when:
- **Savings > Implementation Cost**
- Implementation cost includes: development time, testing, monitoring, error handling

**Rule of thumb:**
- < 10K texts: Use standard (simpler)
- 10K - 100K texts: Evaluate based on cost/time tradeoff
- > 100K texts: Strong candidate for batch (significant savings)

## Implementation Patterns

### Pattern 1: Pure Standard API (Simple)

```python
from azure_llm_toolkit import AzureConfig, PolarsBatchEmbedder

config = AzureConfig()
embedder = PolarsBatchEmbedder(config)

# Uses standard API with built-in batching and rate limiting
embeddings, metadata = await embedder.embed_texts(texts)
```

**Pros:**
- Simple, one-liner
- Built-in retry logic
- Immediate results
- Easy debugging

**Cons:**
- Full cost
- Subject to standard rate limits

### Pattern 2: Standard API with Rate Limiter Integration

```python
from azure_llm_toolkit import AzureConfig, PolarsBatchEmbedder, RateLimiter

config = AzureConfig()
rate_limiter = RateLimiter(
    deployment_name=config.embedding_deployment,
    requests_per_minute=3000,
    tokens_per_minute=1_000_000,
)

embedder = PolarsBatchEmbedder(
    config=config,
    use_rate_limiting=True,
    rate_limiter=rate_limiter,
)

embeddings, metadata = await embedder.embed_texts(texts)
```

**Pros:**
- Coordinated rate limiting
- Prevents quota exhaustion
- Suitable for mixed workloads

**Cons:**
- More complex setup
- Requires rate limiter configuration

### Pattern 3: Batch API (Cost-Optimized)

```python
from azure_llm_toolkit import AzureConfig, PolarsBatchEmbedder
from azure_llm_toolkit.batch_api import AzureBatchAPIClient, BatchQuotaMonitor

config = AzureConfig()

# Check quota first
monitor = BatchQuotaMonitor(config, subscription_type="default")
can_submit, msg = monitor.check_job_feasibility(
    num_texts=len(texts),
    avg_tokens_per_text=100
)

if not can_submit:
    print(msg)
    # Split job or request quota increase
    exit(1)

# Create batch client
batch_client = AzureBatchAPIClient(config)

# Create embedder with batch API enabled
embedder = PolarsBatchEmbedder(
    config=config,
    use_batch_api=True,
    batch_api_client=batch_client,
)

embeddings, metadata = await embedder.embed_texts(texts)
```

**Pros:**
- 50% cost savings
- Separate quota pool
- Suitable for very large datasets

**Cons:**
- More complex
- Longer wait time
- Requires quota monitoring
- More error handling needed

## Real-World Scenarios

### Scenario 1: Startup Prototype
- **Dataset**: 5,000 documents
- **Frequency**: Ad-hoc
- **Budget**: Limited developer time
- **Recommendation**: **Standard API**
- **Reason**: Simplicity > cost savings at this scale

### Scenario 2: Production RAG System
- **Dataset**: 50,000 documents (growing)
- **Frequency**: Weekly refresh
- **Budget**: Moderate
- **Recommendation**: **Standard API** (initially), **Batch API** (when > 100K)
- **Reason**: Standard API easier to maintain until scale justifies complexity

### Scenario 3: Enterprise Data Migration
- **Dataset**: 5,000,000 documents
- **Frequency**: One-time + monthly updates
- **Budget**: Cost-conscious
- **Recommendation**: **Batch API**
- **Reason**: $50K+ savings justifies implementation effort

### Scenario 4: Research Dataset Processing
- **Dataset**: 1,000,000 papers
- **Frequency**: One-time
- **Timeline**: Flexible (2-3 days acceptable)
- **Recommendation**: **Batch API**
- **Reason**: Significant cost savings, time not critical

## Migration Path

### From Standard to Batch

If you start with standard API and want to migrate to batch:

1. **Measure current usage**
   ```python
   # Track costs with InMemoryCostTracker
   from azure_llm_toolkit import InMemoryCostTracker
   
   tracker = InMemoryCostTracker()
   # ... use tracker with embedder
   summary = tracker.get_summary()
   ```

2. **Calculate potential savings**
   - Current cost × 0.5 = Batch API cost
   - Savings = Current cost - Batch API cost

3. **Estimate implementation effort**
   - Development: 4-8 hours
   - Testing: 2-4 hours
   - Monitoring setup: 2-4 hours

4. **Make decision**
   - If savings > (implementation cost / number of runs), migrate
   - Otherwise, stay with standard

## Best Practices

### Standard API
1. ✅ Use `PolarsBatchEmbedder` for efficient batching
2. ✅ Enable caching for repeated texts
3. ✅ Integrate with `RateLimiter` for coordination
4. ✅ Monitor costs with `InMemoryCostTracker`
5. ✅ Use appropriate `max_tokens_per_minute` settings

### Batch API
1. ✅ **Always** check quota before submission
2. ✅ Implement exponential backoff retry logic
3. ✅ Monitor job progress regularly
4. ✅ Set file expiration to avoid hitting 500 file limit
5. ✅ Split very large jobs into manageable chunks
6. ✅ Keep job metadata for tracking and debugging
7. ✅ Handle partial results gracefully

## Quota Management

### Checking Your Quota

```python
# Use the interactive checker
python examples/check_batch_quota.py
```

### Requesting Quota Increase

1. Go to [Azure Portal](https://portal.azure.com)
2. Navigate to your Azure OpenAI resource
3. Select "Quotas" from left menu
4. Click "Request quota increase"
5. Select "Tokens per minute per deployment"
6. In description, specify: "Need batch API queue capacity increase to [X] tokens"
7. Emphasize production use case for faster approval

**Typical approval times:**
- Enterprise accounts: 24-48 hours
- Standard accounts: 3-5 days
- Credit card accounts: 5-7 days

### Default Quotas by Subscription

| Subscription | GPT-4o | GPT-4o-mini | GPT-4 |
|-------------|---------|-------------|-------|
| Enterprise | 5B tokens | 15B tokens | 150M tokens |
| Default | 200M tokens | 1B tokens | 30M tokens |
| Credit Card | 50M tokens | 50M tokens | 5M tokens |

## Error Handling

### Common Batch API Errors

#### 1. `token_limit_exceeded`

```
Error: The number of enqueued tokens has surpassed the configured limit
```

**Solutions:**
- Split job into smaller batches
- Wait for current jobs to complete
- Request quota increase

#### 2. `invalid_json_line`

```
Error: A line in your input file wasn't valid JSON
```

**Solutions:**
- Validate JSONL format
- Check for proper escaping
- Use JSON library for generation

#### 3. `model_not_found`

```
Error: The deployment name wasn't found
```

**Solutions:**
- Verify deployment name matches exactly
- Check deployment exists in your resource
- Ensure same deployment name on all lines

## Monitoring and Observability

### Standard API Monitoring

```python
from azure_llm_toolkit import AzureLLMClient, InMemoryCostTracker

tracker = InMemoryCostTracker()
client = AzureLLMClient(config, cost_tracker=tracker)

# ... make calls

summary = tracker.get_summary()
print(f"Total cost: {summary['total_cost']}")
```

### Batch API Monitoring

```python
from azure_llm_toolkit.batch_api import AzureBatchAPIClient

batch_client = AzureBatchAPIClient(config)

# Submit job
job = await batch_client.create(model=config.embedding_deployment, inputs=token_lists)

# Wait with progress monitoring
status = await batch_client.wait_for_completion(
    job['id'],
    poll_interval=10.0,
    timeout=3600
)

# Get results
result = await batch_client.get_result(job['id'])
```

## Performance Tips

### Optimizing Standard API

1. **Batch size**: Use `max_lists_per_query=2048` for maximum throughput
2. **Token limit**: Set `max_tokens_per_minute` based on your quota
3. **Parallelism**: The toolkit handles this automatically
4. **Caching**: Enable for repeated texts

### Optimizing Batch API

1. **File size**: Larger files (100K+ requests) process more efficiently
2. **Expiration**: Set file expiration to avoid hitting 500 file limit
3. **Chunking**: Split jobs to fit within quota safely (leave 10-20% margin)
4. **Polling**: Start with 60s intervals, increase exponentially

## Code Examples

### Example 1: Standard API for Small Dataset

```python
import asyncio
import polars as pl
from azure_llm_toolkit import AzureConfig, PolarsBatchEmbedder

async def embed_small_dataset():
    config = AzureConfig()
    embedder = PolarsBatchEmbedder(config)
    
    # Small dataset
    df = pl.DataFrame({
        "id": range(1000),
        "text": [f"Document {i}" for i in range(1000)]
    })
    
    result = await embedder.embed_dataframe(df, text_column="text")
    print(f"Embedded {len(result)} documents")

asyncio.run(embed_small_dataset())
```

### Example 2: Batch API for Large Dataset

```python
import asyncio
import polars as pl
from azure_llm_toolkit import AzureConfig, PolarsBatchEmbedder
from azure_llm_toolkit.batch_api import AzureBatchAPIClient, BatchQuotaMonitor

async def embed_large_dataset():
    config = AzureConfig()
    
    # Check quota first
    monitor = BatchQuotaMonitor(config, subscription_type="default")
    
    df = pl.DataFrame({
        "id": range(1_000_000),
        "text": [f"Document {i} with content" for i in range(1_000_000)]
    })
    
    # Estimate tokens
    avg_tokens = 50  # Conservative estimate
    can_submit, msg = monitor.check_job_feasibility(
        num_texts=len(df),
        avg_tokens_per_text=avg_tokens
    )
    
    if not can_submit:
        print(msg)
        # Option 1: Split into chunks
        chunk_size = 100_000
        for i in range(0, len(df), chunk_size):
            chunk_df = df[i:i + chunk_size]
            # Process chunk...
            pass
        return
    
    # Create batch client
    batch_client = AzureBatchAPIClient(config)
    
    # Create embedder with batch API
    embedder = PolarsBatchEmbedder(
        config=config,
        use_batch_api=True,
        batch_api_client=batch_client,
    )
    
    # Submit job
    result = await embedder.embed_dataframe(df, text_column="text")
    print(f"Embedded {len(result)} documents")

asyncio.run(embed_large_dataset())
```

### Example 3: Hybrid Approach

```python
import asyncio
import polars as pl
from azure_llm_toolkit import AzureConfig, PolarsBatchEmbedder
from azure_llm_toolkit.batch_api import AzureBatchAPIClient

async def hybrid_embedding(df: pl.DataFrame, threshold: int = 50_000):
    """
    Use standard API for small batches, batch API for large ones.
    """
    config = AzureConfig()
    
    if len(df) < threshold:
        # Small batch: use standard API
        embedder = PolarsBatchEmbedder(config)
        return await embedder.embed_dataframe(df, text_column="text")
    else:
        # Large batch: use batch API
        batch_client = AzureBatchAPIClient(config)
        embedder = PolarsBatchEmbedder(
            config=config,
            use_batch_api=True,
            batch_api_client=batch_client,
        )
        return await embedder.embed_dataframe(df, text_column="text")
```

## Summary Recommendations

| Your Situation | Recommendation | Why |
|----------------|----------------|-----|
| Just starting | Standard API | Learn the basics first |
| < 10K documents | Standard API | Cost difference minimal |
| 10K - 100K documents | Standard API | Unless cost is critical |
| > 100K documents | Batch API | Significant savings |
| Real-time needs | Standard API | Batch too slow |
| Overnight processing | Batch API | Time flexibility enables savings |
| Tight budget | Batch API | 50% discount is substantial |
| Fast iteration | Standard API | Immediate feedback |

## Additional Resources

- [Azure Batch API Documentation](https://learn.microsoft.com/en-us/azure/ai-foundry/openai/how-to/batch)
- [Batch API Notes](./BATCH_API_NOTES.md)
- [Comprehensive Examples](../examples/polars_batch_embedder_comprehensive.py)
- [Quota Checker Tool](../examples/check_batch_quota.py)

## Quick Reference Commands

```bash
# Check your quota
python examples/check_batch_quota.py

# Run comprehensive examples
python examples/polars_batch_embedder_comprehensive.py

# Test standard API performance
pytest tests/test_batch_embedder.py -v

# Check batch API implementation
pytest tests/test_batch_api.py -v
```

---

**Last Updated:** 2024-12-19
**Version:** 0.1.5