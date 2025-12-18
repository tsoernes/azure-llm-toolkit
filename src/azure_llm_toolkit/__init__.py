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

from .analytics import (
    Anomaly,
    CostAnalytics,
    CostBreakdown,
    CostReport,
    CostTrend,
    UsageStats,
)
from .batch import (
    BaseBatchItem,
    BaseBatchResult,
    BatchError,
    BatchStatus,
    ChatBatchItem,
    ChatBatchResult,
    ChatBatchRunner,
    EmbeddingBatchItem,
    EmbeddingBatchResult,
    EmbeddingBatchRunner,
)
from .batch_embedder import PolarsBatchEmbedder
from .cache import CacheManager, ChatCache, EmbeddingCache, LLMCache
from .circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerError,
    CircuitBreakerMetrics,
    CircuitState,
    MultiCircuitBreaker,
)
from .client import AzureLLMClient
from .config import AzureConfig, detect_embedding_dimension
from .conversation import (
    ConversationConfig,
    ConversationManager,
    ConversationMessage,
)
from .cost_tracker import CostEstimator, CostTracker, InMemoryCostTracker
from .dashboard import (
    RateLimiterSnapshot,
    render_circuit_breaker,
    render_full_dashboard,
    render_multi_circuit_breaker,
    render_operation_metrics,
    render_rate_limiter_pool,
    render_rate_limiter_snapshot,
    snapshot_rate_limiter,
)
from .health import (
    ComponentHealth,
    HealthChecker,
    HealthCheckResult,
    HealthStatus,
)
from .metrics import (
    OPENTELEMETRY_AVAILABLE,
    PROMETHEUS_AVAILABLE,
    AggregatedMetrics,
    MetricsCollector,
    MetricsTracker,
    OperationMetrics,
    PrometheusMetrics,
    create_collector_with_prometheus,
)
from .rate_limiter import (
    InFlightRateLimiter,
    RateLimiter,
    RateLimiterPool,
    get_rate_limiter_pool,
)
from .reranker import (
    LogprobReranker,
    RerankerConfig,
    RerankResult,
    create_reranker,
)
from .streaming import (
    BufferedSink,
    CallbackSink,
    FileSink,
    JSONLSink,
    MultiSink,
    QueueSink,
    StreamChunk,
    StreamProcessor,
    StreamSink,
    stream_to_file,
    stream_to_queue,
    stream_with_callback,
)
from .sync_client import AzureLLMClientSync
from .tools import (
    FunctionDefinition,
    ParameterProperty,
    ToolCall,
    ToolCallResult,
    ToolChoiceType,
    ToolRegistry,
    get_default_registry,
    reset_default_registry,
    tool,
)
from .types import (
    ChatCompletionResult,
    CostInfo,
    EmbeddingResult,
    UsageInfo,
)
from .validation import (
    StructuredOutputError,
    StructuredOutputManager,
    ValidationRetryExhaustedError,
    chat_completion_structured,
    create_extraction_prompt,
    extract_structured_data,
    generate_json_schema,
    parse_json_response,
)

__version__ = "0.1.6"

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
    "InFlightRateLimiter",
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
    # Reranker
    "LogprobReranker",
    "RerankerConfig",
    "RerankResult",
    "create_reranker",
    # Tools / function calling
    "ToolChoiceType",
    "ParameterProperty",
    "FunctionDefinition",
    "ToolCall",
    "ToolCallResult",
    "ToolRegistry",
    "tool",
    "get_default_registry",
    "reset_default_registry",
    # Structured output / validation
    "StructuredOutputError",
    "ValidationRetryExhaustedError",
    "generate_json_schema",
    "create_extraction_prompt",
    "parse_json_response",
    "chat_completion_structured",
    "extract_structured_data",
    "StructuredOutputManager",
    # Sync client
    "AzureLLMClientSync",
    # Health checks
    "HealthStatus",
    "ComponentHealth",
    "HealthCheckResult",
    "HealthChecker",
    # Conversation manager
    "ConversationMessage",
    "ConversationConfig",
    "ConversationManager",
    # Types
    "UsageInfo",
    "CostInfo",
    "ChatCompletionResult",
    "EmbeddingResult",
    # Batch embedder
    "PolarsBatchEmbedder",
    # Batch runners
    "BatchStatus",
    "BatchError",
    "BaseBatchItem",
    "BaseBatchResult",
    "ChatBatchItem",
    "ChatBatchResult",
    "ChatBatchRunner",
    "EmbeddingBatchItem",
    "EmbeddingBatchResult",
    "EmbeddingBatchRunner",
    # Version
    "__version__",
]
