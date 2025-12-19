# Batch API and Rate Limiting Notes

## Summary

This document summarizes findings about the Polars embedder's rate limiting implementation and Azure Batch API rate limit behavior.

## Polars Embedder Rate Limiting

### Current Implementation

The `PolarsBatchEmbedder` uses a **dual rate-limiting approach**:

#### 1. Built-in Batching Logic (Always Active)

- **Token-based batching**: Splits texts into batches based on `max_tokens_per_minute` and `max_lists_per_query`
- **Sleep-based backoff**: Implements delays between batches (`sleep_sec`, `sleep_inc`)
- **Automatic retry**: Handles `RateLimitError` exceptions with exponential backoff
- This is independent from the toolkit's main `RateLimiterPool`

**Constructor parameters:**
```python
max_tokens_per_minute: int = 1_000_000  # Rate limit
max_tokens_per_row: int = 8190          # Max tokens per text
max_lists_per_query: int = 2048         # Max texts per API call
sleep_sec: int = 60                     # Initial sleep on rate limit
sleep_inc: int = 5                      # Sleep increment on consecutive errors
```

#### 2. Optional RateLimiter Integration (Opt-in)

Added in recent updates to coordinate with the toolkit's main rate limiting infrastructure:

```python
use_rate_limiting: bool = True
rate_limiter: RateLimiter | None = None
```

When enabled, the embedder calls `rate_limiter.acquire(tokens=estimated_tokens_batch)` before each batch request, allowing coordination with the `RateLimiterPool`.

### Key Differences from Main Client

**Yes, the Polars embedder has different rate limiting logic:**

1. **Batching-first approach**: Groups texts into batches before making API calls
2. **Sleep-based delays**: Uses `asyncio.sleep()` between batches rather than token bucket algorithms
3. **Local token counting**: Calculates batch sizes locally using tiktoken
4. **Optional integration**: Can work standalone or coordinate with `RateLimiter`

The main `AzureLLMClient` uses `RateLimiterPool` for all requests, which implements:
- Token bucket algorithm
- Per-deployment rate limiting
- Dynamic quota support
- No inter-batch delays

## Azure Batch API Rate Limits

### Key Finding: Rate Limits DO Apply ❌

**The Azure Batch API does NOT bypass rate limiting.** Despite being asynchronous with a 24-hour processing window, Azure enforces strict quota limits on batch jobs.

### Evidence from Research

#### 1. Token Queue Limits

From Microsoft Q&A (October 2025):
- **Default limit**: 50K tokens enqueued across all active batch jobs per deployment
- Error message: `"The number of enqueued tokens has surpassed the configured limit of 50K"`
- This is **total tokens queued**, not TPM (tokens per minute)
- Users must request quota increases through Azure Portal

**Example failure scenario:**
```
Batch 1-4: submitted successfully (total < 50K tokens)
Batch 5: failed with token_limit_exceeded at ~50K enqueued
```

#### 2. TPM/RPM Limits Still Apply

From Azure documentation:
- Batch endpoints are subject to the same **TPM (tokens per minute)** and **RPM (requests per minute)** quotas as standard endpoints
- Batch API has a **separate quota pool** but not unlimited capacity
- Default quotas vary by subscription type and model

**Global Batch default quotas** (enqueued tokens):
- Enterprise: 5B tokens (gpt-4o), 15B (gpt-4o-mini)
- Default: 200M (gpt-4o), 1B (gpt-4o-mini)
- Credit card subscriptions: 50M (gpt-4o)

#### 3. Batch-Specific Limits

Per-batch constraints:
- **Maximum requests per file**: 100,000
- **Maximum input file size**: 200 MB
- **Maximum files per resource**: 500 (or 10,000 with expiration set)
- **Completion window**: 24h (configurable, but jobs taking longer continue until canceled)

### Why Rate Limiting Matters for Batch API

1. **Queue saturation**: Large batch jobs can quickly fill the enqueued token quota
2. **Job failures**: Exceeding quota causes immediate job failure
3. **Resource blocking**: Other batch jobs cannot start until quota is freed
4. **No automatic retry**: Failed jobs don't automatically re-queue

### Best Practices for Batch API

1. **Monitor quota usage**: Track enqueued tokens before submitting jobs
2. **Request quota increases**: For production workloads, increase the default 50K limit
3. **Implement retry logic**: Use exponential backoff for quota-exceeded errors
4. **Chunk large jobs**: Split very large jobs into smaller batches
5. **Use file expiration**: Set `expires_after` to increase file limit from 500 to 10,000

### Recommended Implementation Approach

For the `PolarsBatchEmbedder` Batch API support:

```python
# DO implement:
- Quota checking before job submission
- Exponential backoff retry logic
- Progress monitoring with status polling
- Error handling for quota-exceeded scenarios

# DO NOT implement:
- Bypassing rate limits
- Ignoring quota errors
- Submitting without quota checks
```

## Current Toolkit Status

### Implemented ✅
- Mock Batch API client for testing
- Optional batch API path in `PolarsBatchEmbedder`
- Dual rate limiting (local + optional RateLimiter integration)

### Not Implemented ❌
- Real Azure Batch API client
- Automatic quota monitoring
- Batch job queue management
- Production-ready error handling for batch-specific errors

## Recommendations

### For Users

1. **Use standard embedding endpoint** for most use cases (simpler, more reliable)
2. **Use Batch API** when:
   - Processing very large datasets (millions of texts)
   - Cost savings are critical (50% discount)
   - 24-hour turnaround is acceptable
   - You have sufficient enqueued token quota

### For Future Development

1. **Implement real Batch API client** with proper Azure SDK integration
2. **Add quota monitoring** to prevent job failures
3. **Document quota requirements** clearly for users
4. **Provide examples** showing batch vs. standard endpoint tradeoffs
5. **Consider automatic fallback** from batch to standard when quota is low

## References

- [Azure OpenAI Batch API Documentation](https://learn.microsoft.com/en-us/azure/ai-foundry/openai/how-to/batch)
- [Azure OpenAI Quotas and Limits](https://learn.microsoft.com/en-us/azure/ai-foundry/openai/quotas-limits)
- [Microsoft Q&A: Batch Token Limits](https://learn.microsoft.com/en-us/answers/questions/5572494/clarification-on-azure-openai-batch-limits-token-l)
- [OpenAI Batch API Guide](https://platform.openai.com/docs/guides/batch)

## Last Updated

2024-12-18