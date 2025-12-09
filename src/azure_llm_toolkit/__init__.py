"""
Azure LLM Toolkit - Azure OpenAI client wrapper with advanced features.

This library provides:
- Rate limiting (TPM/RPM) with token bucket algorithm
- Cost tracking and estimation for Azure OpenAI API calls
- Automatic retry logic with exponential backoff
- Batch embedding support
- Chat completions with reasoning support
- Token counting and cost estimation
- Disk-based caching for embeddings and chat completions
- Circuit breaker pattern for resilience
- Metrics collection and cost analytics
- Streaming helpers for writing responses to files/queues

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

    # Embed a single text
    embedding = await client.embed_text("Hello world")
    print(f"Embedding dimension: {len(embedding)}")

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
from .circuit_breaker import (
    CircuitBreaker,
    MultiCircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerMetrics,
    CircuitState,
    CircuitBreakerError,
)
from .client import AzureLLMClient
from .config import AzureConfig, detect_embedding_dimension
from .cost_tracker import CostEstimator, CostTracker, InMemoryCostTracker
from .dashboard import (
    RateLimiterSnapshot,
    snapshot_rate_limiter,
    render_rate_limiter_snapshot,
    render_rate_limiter_pool,
    render_circuit_breaker,
    render_multi_circuit_breaker,
    render_operation_metrics,
    render_full_dashboard,
)
from .metrics import (
    MetricsCollector,
    MetricsTracker,
    OperationMetrics,
    AggregatedMetrics,
    PrometheusMetrics,
    create_collector_with_prometheus,
    PROMETHEUS_AVAILABLE,
    OPENTELEMETRY_AVAILABLE,
)
from .rate_limiter import RateLimiter, RateLimiterPool, get_rate_limiter_pool
from .streaming import (
    StreamSink,
    StreamChunk,
    FileSink,
    JSONLSink,
    QueueSink,
    CallbackSink,
    MultiSink,
    BufferedSink,
    StreamProcessor,
    stream_to_file,
    stream_to_queue,
    stream_with_callback,
)
from .analytics import (
    CostAnalytics,
    CostReport,
    CostBreakdown,
    UsageStats,
    CostTrend,
    Anomaly,
)
from .types import (
    ChatCompletionResult,
    CostInfo,
    EmbeddingResult,
    QueryRewriteResult,
    UsageInfo,
)

__version__ = "0.1.1"

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
    # Circuit breaker
    "CircuitBreaker",
    "MultiCircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitBreakerMetrics",
    "CircuitState",
    "CircuitBreakerError",
    # Metrics & telemetry
    "MetricsCollector",
    "MetricsTracker",
    "OperationMetrics",
    "AggregatedMetrics",
    "PrometheusMetrics",
    "create_collector_with_prometheus",
    "PROMETHEUS_AVAILABLE",
    "OPENTELEMETRY_AVAILABLE",
    # Dashboard helpers
    "RateLimiterSnapshot",
    "snapshot_rate_limiter",
    "render_rate_limiter_snapshot",
    "render_rate_limiter_pool",
    "render_circuit_breaker",
    "render_multi_circuit_breaker",
    "render_operation_metrics",
    "render_full_dashboard",
    # Streaming
    "StreamSink",
    "StreamChunk",
    "FileSink",
    "JSONLSink",
    "QueueSink",
    "CallbackSink",
    "MultiSink",
    "BufferedSink",
    "StreamProcessor",
    "stream_to_file",
    "stream_to_queue",
    "stream_with_callback",
    # Cost analytics
    "CostAnalytics",
    "CostReport",
    "CostBreakdown",
    "UsageStats",
    "CostTrend",
    "Anomaly",
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
