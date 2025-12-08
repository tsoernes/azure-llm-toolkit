"""
Azure LLM Toolkit - Azure OpenAI client wrapper with advanced features.

This library provides:
- Rate limiting (TPM/RPM) with token bucket algorithm
- Cost tracking and estimation for Azure OpenAI API calls
- Automatic retry logic with exponential backoff
- Batch embedding support
- Chat completions with reasoning support
- Query rewriting utilities
- Token counting and cost estimation

Example usage:

    from azure_llm_toolkit import AzureConfig, AzureLLMClient, CostEstimator

    # Configure Azure OpenAI
    config = AzureConfig(
        api_key="your-api-key",
        endpoint="https://your-resource.openai.azure.com",
        chat_deployment="gpt-4o",
        embedding_deployment="text-embedding-3-large",
    )

    # Create client
    client = AzureLLMClient(config=config, enable_rate_limiting=True)

    # Embed texts
    result = await client.embed_texts(["Hello world", "Another text"])
    print(f"Generated {len(result.embeddings)} embeddings")

    # Chat completion
    response = await client.chat_completion(
        messages=[{"role": "user", "content": "What is AI?"}],
        system_prompt="You are a helpful assistant.",
    )
    print(response.content)

    # Cost estimation
    estimator = CostEstimator(currency="kr")
    cost = estimator.estimate_cost(
        model="gpt-4o",
        tokens_input=1000,
        tokens_output=500,
    )
    print(f"Estimated cost: {cost:.2f} kr")
"""

from __future__ import annotations

from .batch_embedder import PolarsBatchEmbedder
from .cache import CacheManager, ChatCache, EmbeddingCache, LLMCache
from .client import AzureLLMClient
from .config import AzureConfig, detect_embedding_dimension
from .cost_tracker import CostEstimator, CostTracker, InMemoryCostTracker
from .rate_limiter import RateLimiter, RateLimiterPool, get_rate_limiter_pool
from .types import (
    ChatCompletionResult,
    CostInfo,
    EmbeddingResult,
    QueryRewriteResult,
    UsageInfo,
)

__version__ = "0.1.0"

__all__ = [
    # Main client
    "AzureLLMClient",
    # Configuration
    "AzureConfig",
    "detect_embedding_dimension",
    # Cost tracking
    "CostEstimator",
    "CostTracker",
    "InMemoryCostTracker",
    # Rate limiting
    "RateLimiter",
    "RateLimiterPool",
    "get_rate_limiter_pool",
    # Caching
    "CacheManager",
    "LLMCache",
    "EmbeddingCache",
    "ChatCache",
    # Types
    "UsageInfo",
    "CostInfo",
    "ChatCompletionResult",
    "EmbeddingResult",
    "QueryRewriteResult",
    # Batch embedder
    "PolarsBatchEmbedder",
    # Version
    "__version__",
]
