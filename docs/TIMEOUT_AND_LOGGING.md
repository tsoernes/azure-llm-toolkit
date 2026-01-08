# Timeout Configuration and Logging Guide

## Overview

The Azure LLM Toolkit provides flexible timeout configuration and comprehensive logging to help you debug issues, monitor performance, and optimize API usage. This guide covers timeout behavior, logging improvements, and best practices.

## Table of Contents

1. [Timeout Configuration](#timeout-configuration)
2. [Logging Improvements](#logging-improvements)
3. [Understanding Log Messages](#understanding-log-messages)
4. [Best Practices](#best-practices)
5. [Troubleshooting](#troubleshooting)

---

## Timeout Configuration

### Default Behavior

**As of version 0.2.0**, the default timeout is **infinite** (`None`), meaning API requests will wait indefinitely for a response. This is recommended for:

- **Reasoning models** (o1, GPT-5) that can take 30+ seconds for complex tasks
- **Large batch operations** that may take longer to process
- **Production environments** where you want to avoid false failures

### Setting a Custom Timeout

You can set a custom timeout in three ways:

#### 1. Environment Variable (Recommended)

```bash
# In your .env file
AZURE_TIMEOUT_SECONDS=300  # 5 minutes
```

#### 2. Configuration Object

```python
from azure_llm_toolkit import AzureConfig, AzureLLMClient

config = AzureConfig(timeout_seconds=120.0)  # 2 minutes
client = AzureLLMClient(config=config)
```

#### 3. Runtime Override

```python
import os
os.environ['AZURE_TIMEOUT_SECONDS'] = '180'

# Config will pick up the new value
config = AzureConfig()
```

### Timeout Values

- **`None`** (default): Infinite timeout - wait as long as needed
- **`float` value**: Timeout in seconds (e.g., `60.0` = 1 minute)
- **Minimum recommended**: `120` seconds for reasoning models
- **Conservative value**: `300` seconds (5 minutes) for all models

### Breaking Change Notice

⚠️ **Breaking Change in v0.2.0**: Default timeout changed from 60 seconds to infinite.

**Migration**: If you need the old behavior:
```bash
AZURE_TIMEOUT_SECONDS=60
```

---

## Logging Improvements

### Log Levels

The toolkit uses Python's standard logging module with the following levels:

- **DEBUG**: Detailed diagnostic information (initialization, timings, rate limiter)
- **INFO**: Informational messages (slow requests, performance notes)
- **WARNING**: Retry attempts, cache failures, configuration warnings
- **ERROR**: Final failures with actionable suggestions

### Enabling Logging

```python
import logging

# Enable DEBUG logging to see all details
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)

from azure_llm_toolkit import AzureLLMClient

client = AzureLLMClient()
```

### What Gets Logged

#### 1. Client Initialization (DEBUG)

Shows your configuration when the client is created:

```
DEBUG - Initialized AzureLLMClient (api_timeout=infinite, max_retries=5, rate_limiting=True, caching=True)
```

#### 2. Request Timing (DEBUG)

Every API request logs its elapsed time:

```
DEBUG - Chat completion completed in 2.34s (model=gpt-4o)
DEBUG - embed_text received response (model=text-embedding-3-small, elapsed_seconds=0.85)
```

#### 3. Slow Requests (INFO)

Informational warnings for requests that take longer than expected:

```
INFO - Chat completion took 45.67s (model=gpt-5-mini, threshold=30.0s). This is normal for reasoning models.
INFO - Embedding request took 12.43s (model=text-embedding-3-large)
```

**Thresholds**:
- Reasoning models (o1, GPT-5): 30 seconds
- Other models: 10 seconds
- Embeddings: 10 seconds

#### 4. Retry Attempts (WARNING)

When an API call fails and is retried, showing both retry backoff delay and current timeout config:

```
WARNING - Retry attempt 1 after APITimeoutError: Request timed out. (payload_hash=68abcf0e, retry_backoff_delay=1.0s, api_timeout=infinite)
WARNING - Retry attempt 2 after RateLimitError: Rate limit exceeded. (payload_hash=68abcf0e, retry_backoff_delay=2.0s, api_timeout=120.0s)
```

#### 5. Timeout Errors (ERROR)

Specific error messages for timeout failures with actionable advice:

```
ERROR - API timeout on chat completion after 3 attempts, configured timeout=60s, model=gpt-5-mini. Consider increasing AZURE_TIMEOUT_SECONDS if needed.
ERROR - API timeout on embedding request, configured timeout=60s, model=text-embedding-3-large. Consider increasing AZURE_TIMEOUT_SECONDS if needed. Error: Request timed out.
```

---

## Understanding Log Messages

### Common Confusion: Retry Backoff vs API Timeout

❌ **WRONG INTERPRETATION**:
```
WARNING - Retry attempt 1 after APITimeoutError (retry_backoff_delay=1.0s, ...)
```
"The API timed out after 1 second"

✅ **CORRECT INTERPRETATION**:
- The API request timed out after the configured timeout (e.g., 60s or infinite)
- The system will wait **1.0 seconds** before retrying (exponential backoff)
- The `retry_backoff_delay` is how long to wait **between retries**, not the timeout duration

### Retry Backoff Schedule

The retry mechanism uses exponential backoff:

| Attempt | Wait Before Retry |
|---------|------------------|
| 1st     | 1 second         |
| 2nd     | 2 seconds        |
| 3rd     | 4 seconds        |
| 4th     | 8 seconds        |
| 5th     | 10 seconds (max) |

**Total retries**: 5 attempts (original + 4 retries)

**Retry conditions**:
- `APIConnectionError` - Network/connection issues
- `RateLimitError` - API rate limit exceeded
- `APITimeoutError` - Request timeout exceeded
- `APIStatusError` - HTTP 5xx server errors

---

## Best Practices

### 1. Use Infinite Timeout for Reasoning Models

```python
# Recommended for o1, GPT-5 models
from azure_llm_toolkit import AzureLLMClient

client = AzureLLMClient()  # Uses infinite timeout by default
```

### 2. Set Reasonable Timeouts for Production

If you need bounded timeouts for SLA purposes:

```python
config = AzureConfig(timeout_seconds=300.0)  # 5 minutes
client = AzureLLMClient(config=config)
```

### 3. Enable DEBUG Logging During Development

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# You'll see initialization, timing, and detailed diagnostics
```

### 4. Monitor Slow Requests

Watch for INFO-level logs about slow requests:
- Frequent slow requests may indicate model or deployment issues
- Reasoning models normally take 30+ seconds - this is expected
- Standard models taking >10s may need investigation

### 5. Handle Timeout Errors Gracefully

```python
from openai import APITimeoutError

try:
    result = await client.chat_completion(
        messages=[{"role": "user", "content": "Your prompt"}],
        model="gpt-5-mini"
    )
except APITimeoutError as e:
    logger.error(f"Request timed out: {e}")
    # Retry with longer timeout or different strategy
```

### 6. Use Different Timeouts for Different Operations

```python
# Embeddings: shorter timeout (usually fast)
embed_config = AzureConfig(timeout_seconds=30.0)
embed_client = AzureLLMClient(config=embed_config)

# Chat completions: infinite timeout (reasoning models)
chat_config = AzureConfig(timeout_seconds=None)
chat_client = AzureLLMClient(config=chat_config)
```

---

## Troubleshooting

### Issue: "Retry attempt 1 after APITimeoutError"

**Symptoms**: Seeing timeout errors in logs

**Check**:
1. What is your configured timeout?
   ```python
   print(client.config.timeout_seconds)  # None or float
   ```

2. Enable DEBUG logging to see initialization:
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   # Look for: "Initialized AzureLLMClient (api_timeout=...)"
   ```

3. Check if it's a reasoning model that needs more time

**Solutions**:
- Set `AZURE_TIMEOUT_SECONDS` to a higher value or remove it for infinite
- Check Azure service health
- Monitor request timing logs to see actual durations

### Issue: Requests Taking Too Long

**Symptoms**: INFO logs showing slow requests

**Investigation**:
1. Check model type:
   - Reasoning models (o1, GPT-5): 30+ seconds is normal
   - Standard models: >10s may indicate issues

2. Check prompt complexity:
   - Very long prompts take longer to process
   - Complex reasoning tasks take longer

3. Check Azure metrics:
   - High utilization on your deployment
   - Throttling or queuing

**Solutions**:
- For reasoning models: This is expected behavior
- For standard models: Check prompt size, deployment capacity
- Consider different Azure region or deployment tier

### Issue: Too Many Retries

**Symptoms**: Multiple retry attempts before success

**Check**:
1. Are retries due to timeouts or other errors?
   - Timeouts: Increase timeout or check service health
   - Rate limits: Increase deployment capacity or add rate limiting
   - Connection errors: Check network connectivity

2. Look at the retry pattern in logs:
   ```
   WARNING - Retry attempt 1 after RateLimitError (retry_backoff_delay=1.0s, ...)
   WARNING - Retry attempt 2 after RateLimitError (retry_backoff_delay=2.0s, ...)
   ```

**Solutions**:
- **Rate limit errors**: Enable rate limiting or increase Azure quotas
- **Timeout errors**: Increase timeout configuration
- **Connection errors**: Check network, Azure service status

### Issue: Want to Disable Retries

**Not recommended**, but if needed:

```python
from openai import AsyncAzureOpenAI

# Create client without retry wrapper
raw_client = AsyncAzureOpenAI(
    azure_endpoint=config.endpoint,
    api_key=config.api_key,
    api_version=config.api_version,
    timeout=None,
    max_retries=0  # Disable OpenAI SDK retries
)

# Use raw_client (but lose toolkit benefits)
```

Note: The toolkit's retry mechanism adds value. Consider keeping it enabled.

---

## Example Logging Output

### Successful Request (DEBUG level)

```
2026-01-08 14:45:00 DEBUG - azure_llm_toolkit.client - Initialized AzureLLMClient (api_timeout=infinite, max_retries=5, rate_limiting=True, caching=True)
2026-01-08 14:45:00 DEBUG - azure_llm_toolkit.rate_limiter - Rate limiter acquiring 150 tokens for model gpt-5-mini
2026-01-08 14:45:32 DEBUG - azure_llm_toolkit.client - Chat completion completed in 32.45s (model=gpt-5-mini)
2026-01-08 14:45:32 INFO - azure_llm_toolkit.client - Chat completion took 32.45s (model=gpt-5-mini, threshold=30.0s). This is normal for reasoning models.
```

### Request with Timeout and Retry (WARNING level)

```
2026-01-08 14:50:00 DEBUG - azure_llm_toolkit.client - Initialized AzureLLMClient (api_timeout=60.0s, max_retries=5, rate_limiting=True, caching=True)
2026-01-08 14:51:00 WARNING - azure_llm_toolkit.client - Retry attempt 1 after APITimeoutError: Request timed out. (payload_hash=a1b2c3d4, retry_backoff_delay=1.0s, api_timeout=60.0s)
2026-01-08 14:52:03 DEBUG - azure_llm_toolkit.client - Chat completion completed in 62.12s (model=gpt-4o)
```

### Final Timeout Failure (ERROR level)

```
2026-01-08 15:00:00 WARNING - azure_llm_toolkit.client - API timeout on chat completion (attempt 1/3), configured timeout=30s, model=gpt-5-mini. Error: Request timed out.
2026-01-08 15:00:33 WARNING - azure_llm_toolkit.client - API timeout on chat completion (attempt 2/3), configured timeout=30s, model=gpt-5-mini. Error: Request timed out.
2026-01-08 15:01:07 ERROR - azure_llm_toolkit.client - API timeout on chat completion after 3 attempts, configured timeout=30s, model=gpt-5-mini. Consider increasing AZURE_TIMEOUT_SECONDS if needed.
Traceback (most recent call last):
  ...
openai.APITimeoutError: Request timed out.
```

---

## Configuration Reference

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `AZURE_TIMEOUT_SECONDS` | `None` (infinite) | API request timeout in seconds |
| `AZURE_MAX_RETRIES` | `5` | Maximum retry attempts |

### Configuration Object

```python
from azure_llm_toolkit import AzureConfig

config = AzureConfig(
    timeout_seconds=None,        # float or None (infinite)
    max_retries=5,               # int
    # ... other config options
)
```

### Retry Configuration (Built-in)

The retry mechanism is configured in the client and **cannot be disabled** through configuration (by design for reliability):

- **Max attempts**: 5 (1 original + 4 retries)
- **Wait strategy**: Exponential backoff (1s, 2s, 4s, 8s, 10s max)
- **Retry conditions**: Connection errors, rate limits, timeouts, server errors
- **No retry on**: Bad request (4xx), authentication errors

---

## Changelog

### Version 0.2.0 (2026-01-08)

**Breaking Changes**:
- Default timeout changed from 60 seconds to infinite (`None`)
- Recommended for reasoning models and production use

**New Features**:
- Client initialization logging with timeout config
- Request timing logs for all API calls
- Enhanced retry logging with distinction between backoff delay and API timeout
- Specific APITimeoutError handling with actionable error messages
- Performance warnings for slow requests (model-aware thresholds)

**Improvements**:
- Clearer log messages: `retry_backoff_delay` vs `api_timeout`
- Better debugging information for timeout issues
- Context-aware error messages with configuration details

---

## Additional Resources

- [Azure OpenAI Service Documentation](https://learn.microsoft.com/en-us/azure/ai-services/openai/)
- [OpenAI API Timeout Documentation](https://github.com/openai/openai-python#timeouts)
- [Python Logging Documentation](https://docs.python.org/3/library/logging.html)

---

## Support

For issues, questions, or feature requests:
- Check existing GitHub issues
- Review this documentation
- Enable DEBUG logging for detailed diagnostics
- Include log output when reporting issues