# Azure LLM Toolkit - Suggested Features & Improvements

This document outlines potential features, improvements, and enhancements for the azure-llm-toolkit project, organized by priority and category.

---

## 🚀 High Priority Features

### 1. **Streaming Support for Chat Completions**
**Status**: Not implemented  
**Priority**: High  
**Effort**: Medium

Add support for streaming chat completions with proper rate limiting and cost tracking.

**Implementation**:
```python
async def chat_completion_stream(
    self,
    messages: list[dict],
    callback: Callable[[str], None],
    **kwargs
) -> ChatCompletionResult:
    """Stream chat completion with incremental callbacks."""
    pass
```

**Benefits**:
- Better UX for interactive applications
- Reduced perceived latency
- Real-time token counting and cost tracking

**Related**: `streaming.py` already has some infrastructure

---

### 2. **Function Calling / Tools Support**
**Status**: Not implemented  
**Priority**: High  
**Effort**: Medium

Add first-class support for Azure OpenAI function calling.

**Implementation**:
```python
@dataclass
class FunctionDefinition:
    name: str
    description: str
    parameters: dict[str, Any]

async def chat_completion_with_tools(
    self,
    messages: list[dict],
    tools: list[FunctionDefinition],
    tool_choice: str = "auto",
) -> ChatCompletionResult:
    """Chat completion with function calling support."""
    pass
```

**Benefits**:
- Enable agent architectures
- Structured output generation
- Better integration with external systems

---

### 3. **Async Context Manager Support**
**Status**: Partial  
**Priority**: High  
**Effort**: Low

Add proper async context manager protocol for resource cleanup.

**Implementation**:
```python
class AzureLLMClient:
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
    
    async def close(self):
        """Clean up resources (close connections, flush caches, etc.)."""
        pass
```

**Usage**:
```python
async with AzureLLMClient(config=config) as client:
    result = await client.chat_completion(...)
# Automatic cleanup
```

---

### 4. **Batch API Support**
**Status**: Not implemented  
**Priority**: High  
**Effort**: High

Support for Azure OpenAI Batch API (50% cost savings for non-urgent tasks).

**Implementation**:
```python
class BatchRequest:
    requests: list[dict]
    metadata: dict[str, str]

async def create_batch(
    self,
    requests: BatchRequest,
    completion_window: str = "24h",
) -> str:
    """Submit batch job and return batch_id."""
    pass

async def get_batch_status(self, batch_id: str) -> dict:
    """Check batch job status."""
    pass

async def retrieve_batch_results(self, batch_id: str) -> list[dict]:
    """Retrieve completed batch results."""
    pass
```

**Benefits**:
- 50% cost savings for batch workloads
- Process large datasets efficiently
- Background job processing

---

## 🎯 Medium Priority Features

### 5. **Vision Model Support**
**Status**: Not implemented  
**Priority**: Medium  
**Effort**: Medium

Support for GPT-4 Vision (image understanding).

**Implementation**:
```python
async def chat_completion_with_images(
    self,
    messages: list[dict],  # Can include image URLs or base64
    max_tokens: int = 1000,
    detail: str = "auto",  # "low" | "high" | "auto"
) -> ChatCompletionResult:
    """Chat completion with image inputs."""
    pass
```

**Example**:
```python
messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "What's in this image?"},
            {"type": "image_url", "image_url": {"url": "https://..."}}
        ]
    }
]
```

---

### 6. **Advanced Caching Strategies**
**Status**: Basic disk caching implemented  
**Priority**: Medium  
**Effort**: Medium

Enhance caching with TTL, LRU eviction, and size limits.

**Implementation**:
```python
@dataclass
class CacheConfig:
    max_size_mb: int = 1000
    ttl_seconds: int = 86400  # 24 hours
    eviction_policy: str = "lru"  # "lru" | "fifo" | "lfu"
    compression: bool = True

class AdvancedCache:
    def __init__(self, config: CacheConfig):
        pass
    
    async def cleanup_expired(self):
        """Remove expired entries."""
        pass
    
    async def enforce_size_limit(self):
        """Evict entries if cache exceeds size limit."""
        pass
```

**Benefits**:
- Automatic cache management
- Better memory efficiency
- Configurable retention policies

---

### 7. **Sync Client Wrapper**
**Status**: Not implemented  
**Priority**: Medium  
**Effort**: Low

Synchronous wrapper for non-async codebases.

**Implementation**:
```python
class AzureLLMClientSync:
    """Synchronous wrapper around async client."""
    
    def __init__(self, config: AzureConfig):
        self._async_client = AzureLLMClient(config)
        self._loop = asyncio.new_event_loop()
    
    def chat_completion(self, **kwargs) -> ChatCompletionResult:
        return self._loop.run_until_complete(
            self._async_client.chat_completion(**kwargs)
        )
```

**Benefits**:
- Support legacy codebases
- Easier integration with sync frameworks
- No async/await required

---

### 8. **Response Validation & Parsing**
**Status**: Not implemented  
**Priority**: Medium  
**Effort**: Medium

Structured output parsing with validation.

**Implementation**:
```python
from pydantic import BaseModel

class ExtractedInfo(BaseModel):
    name: str
    age: int
    email: str

async def chat_completion_structured(
    self,
    messages: list[dict],
    response_model: Type[BaseModel],
    max_retries: int = 3,
) -> BaseModel:
    """Get validated structured output from LLM."""
    pass
```

**Benefits**:
- Type-safe LLM outputs
- Automatic validation
- Retry on parse failures

---

### 9. **Multi-Model Routing**
**Status**: Not implemented  
**Priority**: Medium  
**Effort**: Medium

Intelligent routing between models based on task complexity.

**Implementation**:
```python
class ModelRouter:
    def __init__(
        self,
        small_model: str = "gpt-4o-mini",
        large_model: str = "gpt-4o",
        classifier: Callable | None = None,
    ):
        pass
    
    async def route_and_complete(
        self,
        messages: list[dict],
        complexity_threshold: float = 0.5,
    ) -> ChatCompletionResult:
        """Route to appropriate model based on complexity."""
        pass
```

**Benefits**:
- Cost optimization
- Automatic model selection
- Performance tuning

---

## 🔧 Quality of Life Improvements

### 10. **Better Logging & Observability**
**Status**: Basic logging  
**Priority**: Medium  
**Effort**: Low

Enhanced logging with structured formats and filtering.

**Implementation**:
```python
import structlog

class ObservabilityConfig:
    log_format: str = "json"  # "json" | "text"
    log_level: str = "INFO"
    log_api_requests: bool = True
    log_api_responses: bool = False
    redact_sensitive: bool = True

# Usage
logger = structlog.get_logger()
logger.info(
    "chat_completion",
    model=model,
    tokens=usage.total_tokens,
    cost=cost.amount,
    latency_ms=elapsed * 1000,
)
```

**Add**:
- Request/response logging
- Performance metrics
- Error context

---

### 11. **Configuration Presets**
**Status**: Manual configuration  
**Priority**: Low  
**Effort**: Low

Pre-configured settings for common scenarios.

**Implementation**:
```python
class AzureConfig:
    @classmethod
    def for_production(cls, **overrides) -> AzureConfig:
        """Production-optimized settings."""
        return cls(
            enable_rate_limiting=True,
            enable_cache=True,
            retry_attempts=5,
            **overrides
        )
    
    @classmethod
    def for_development(cls, **overrides) -> AzureConfig:
        """Development-friendly settings."""
        return cls(
            enable_rate_limiting=False,
            enable_cache=False,
            log_level="DEBUG",
            **overrides
        )
    
    @classmethod
    def for_cost_optimization(cls, **overrides) -> AzureConfig:
        """Cost-optimized settings."""
        return cls(
            default_model="gpt-4o-mini",
            enable_cache=True,
            cache_ttl=86400,
            **overrides
        )
```

---

### 12. **CLI Tool**
**Status**: Not implemented  
**Priority**: Low  
**Effort**: Medium

Command-line interface for common operations.

**Implementation**:
```bash
# Test configuration
azure-llm-toolkit test-config

# Estimate costs
azure-llm-toolkit estimate --model gpt-4o --tokens 1000

# Batch embed files
azure-llm-toolkit embed --input docs/ --output embeddings.parquet

# Interactive chat
azure-llm-toolkit chat --model gpt-4o

# Analyze costs
azure-llm-toolkit costs --since 2024-01-01 --breakdown model

# Clear cache
azure-llm-toolkit cache clear
```

---

### 13. **Retry Strategy Customization**
**Status**: Fixed exponential backoff  
**Priority**: Low  
**Effort**: Low

Configurable retry strategies.

**Implementation**:
```python
from enum import Enum

class RetryStrategy(Enum):
    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    FIBONACCI = "fibonacci"
    CONSTANT = "constant"

@dataclass
class RetryConfig:
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
    max_attempts: int = 5
    initial_delay: float = 1.0
    max_delay: float = 60.0
    jitter: bool = True
```

---

## 📊 Analytics & Monitoring

### 14. **Cost Analytics Dashboard**
**Status**: Basic cost tracking  
**Priority**: Medium  
**Effort**: Medium

Web-based dashboard for cost visualization.

**Features**:
- Cost trends over time
- Model breakdown
- Category analysis
- Budget alerts
- Usage patterns

**Tech Stack**:
- FastAPI backend
- React/Vue frontend
- Plotly/Chart.js for graphs

---

### 15. **Prometheus Metrics Exporter**
**Status**: Partial (metrics.py has some support)  
**Priority**: Medium  
**Effort**: Low

Export metrics in Prometheus format.

**Metrics**:
```python
# Counters
azure_llm_requests_total{model, status}
azure_llm_tokens_total{model, type}

# Histograms
azure_llm_request_duration_seconds{model}
azure_llm_token_count{model}

# Gauges
azure_llm_cost_total{model, currency}
azure_llm_rate_limiter_available{resource_type}
```

**Integration**:
```python
from prometheus_client import start_http_server

# Start metrics server
start_http_server(8000)
```

---

### 16. **OpenTelemetry Integration**
**Status**: Basic support  
**Priority**: Medium  
**Effort**: Medium

Full OpenTelemetry tracing and metrics.

**Features**:
- Distributed tracing
- Span context propagation
- Automatic instrumentation
- Custom attributes

---

## 🔒 Security & Compliance

### 17. **Secrets Management Integration**
**Status**: Environment variables only  
**Priority**: Medium  
**Effort**: Low

Support for secret managers.

**Implementation**:
```python
from enum import Enum

class SecretProvider(Enum):
    ENV = "env"
    AZURE_KEYVAULT = "azure_keyvault"
    AWS_SECRETS = "aws_secrets"
    HASHICORP_VAULT = "hashicorp_vault"

class AzureConfig:
    @classmethod
    async def from_secret_provider(
        cls,
        provider: SecretProvider,
        secret_name: str,
    ) -> AzureConfig:
        """Load config from secret manager."""
        pass
```

---

### 18. **PII Detection & Redaction**
**Status**: Not implemented  
**Priority**: Medium  
**Effort**: Medium

Automatic detection and redaction of sensitive data.

**Implementation**:
```python
class PIIDetector:
    def __init__(self, patterns: list[str] | None = None):
        self.patterns = patterns or self._default_patterns()
    
    def detect(self, text: str) -> list[tuple[str, str]]:
        """Detect PII in text. Returns [(type, value)]."""
        pass
    
    def redact(self, text: str, replacement: str = "[REDACTED]") -> str:
        """Redact PII from text."""
        pass

# Auto-redact before API calls
client = AzureLLMClient(config=config, enable_pii_redaction=True)
```

---

### 19. **Audit Logging**
**Status**: Not implemented  
**Priority**: Medium  
**Effort**: Low

Comprehensive audit trail for compliance.

**Features**:
- Log all API calls
- User/session tracking
- Input/output recording
- Timestamp and metadata
- Tamper-proof logs

---

## 🚀 Performance Optimizations

### 20. **Connection Pooling**
**Status**: Uses default httpx pooling  
**Priority**: Medium  
**Effort**: Medium

Optimize connection management.

**Implementation**:
```python
import httpx

class AzureLLMClient:
    def __init__(self, config: AzureConfig):
        limits = httpx.Limits(
            max_keepalive_connections=20,
            max_connections=100,
            keepalive_expiry=30.0,
        )
        self._http_client = httpx.AsyncClient(limits=limits)
```

---

### 21. **Request Batching**
**Status**: Manual batching only  
**Priority**: Low  
**Effort**: Medium

Automatic request batching for efficiency.

**Implementation**:
```python
class BatchingClient:
    def __init__(
        self,
        client: AzureLLMClient,
        batch_size: int = 10,
        batch_timeout: float = 0.1,
    ):
        pass
    
    async def embed_text(self, text: str) -> list[float]:
        """Automatically batched embedding."""
        pass
```

**Benefits**:
- Reduced API calls
- Better throughput
- Lower costs

---

### 22. **Parallel Embedding Optimization**
**Status**: Basic parallel support  
**Priority**: Low  
**Effort**: Low

Optimize parallel embedding with semaphores.

**Implementation**:
```python
async def embed_texts_parallel(
    self,
    texts: list[str],
    max_concurrent: int = 10,
) -> EmbeddingResult:
    """Embed texts with concurrency control."""
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def embed_with_semaphore(text: str):
        async with semaphore:
            return await self.embed_text(text)
    
    embeddings = await asyncio.gather(*[
        embed_with_semaphore(text) for text in texts
    ])
```

---

## 🧪 Testing & Quality

### 23. **Mock Server for Testing**
**Status**: Not implemented  
**Priority**: Medium  
**Effort**: Medium

Mock Azure OpenAI server for testing.

**Implementation**:
```python
class MockAzureServer:
    def __init__(self, responses: dict[str, Any]):
        self.responses = responses
    
    async def start(self, port: int = 8000):
        """Start mock server."""
        pass

# Usage in tests
async def test_client():
    mock = MockAzureServer(responses={
        "/chat/completions": {"choices": [...]},
    })
    await mock.start()
    
    client = AzureLLMClient(endpoint="http://localhost:8000")
    result = await client.chat_completion(...)
```

---

### 24. **Integration Test Suite**
**Status**: Basic unit tests  
**Priority**: Medium  
**Effort**: High

Comprehensive integration tests.

**Coverage**:
- End-to-end workflows
- Rate limiting under load
- Cost tracking accuracy
- Cache behavior
- Error recovery
- Multi-model scenarios

---

### 25. **Performance Benchmarks**
**Status**: Not implemented  
**Priority**: Low  
**Effort**: Medium

Automated performance benchmarking.

**Metrics**:
- Throughput (requests/sec)
- Latency (p50, p95, p99)
- Token processing speed
- Cache hit rates
- Memory usage

---

## 🌐 Multi-Provider Support

### 26. **OpenAI API Support**
**Status**: Azure-only  
**Priority**: Low  
**Effort**: Medium

Support direct OpenAI API (non-Azure).

**Benefits**:
- Use same interface for both
- Easy migration between providers
- Testing flexibility

---

### 27. **Azure AI Inference Support**
**Status**: Not implemented  
**Priority**: Low  
**Effort**: High

Support for Azure AI services endpoints (services.ai).

**Features**:
- Unified interface
- OSS model support
- Multi-region routing

---

## 📚 Documentation & Examples

### 28. **Interactive Tutorials**
**Status**: Basic examples  
**Priority**: Low  
**Effort**: Medium

Jupyter notebooks with interactive examples.

**Topics**:
- Getting started
- Rate limiting strategies
- Cost optimization
- RAG implementation
- Agent patterns
- Production deployment

---

### 29. **API Reference Site**
**Status**: Docstrings only  
**Priority**: Low  
**Effort**: Low

Auto-generated API documentation.

**Tech**:
- Sphinx or MkDocs
- Auto-generated from docstrings
- Search functionality
- Examples embedded

---

### 30. **Cookbook Repository**
**Status**: Not implemented  
**Priority**: Low  
**Effort**: Medium

Collection of recipes and patterns.

**Examples**:
- RAG pipeline
- Agent with tools
- Batch processing
- Streaming chatbot
- Cost optimization
- Error handling patterns

---

## 🔄 Migration & Compatibility

### 31. **LangChain Integration**
**Status**: Not implemented  
**Priority**: Medium  
**Effort**: Medium

Adapter for LangChain compatibility.

**Implementation**:
```python
from langchain.llms.base import LLM

class AzureLLMToolkitLLM(LLM):
    client: AzureLLMClient
    
    def _call(self, prompt: str, **kwargs) -> str:
        result = asyncio.run(
            self.client.chat_completion(
                messages=[{"role": "user", "content": prompt}]
            )
        )
        return result.content
```

---

### 32. **LlamaIndex Integration**
**Status**: Not implemented  
**Priority**: Medium  
**Effort**: Low

Integration with LlamaIndex framework.

---

## 🎨 Developer Experience

### 33. **Type Stubs & Editor Support**
**Status**: Full type hints  
**Priority**: Low  
**Effort**: Low

Enhanced IDE support.

**Features**:
- Better autocomplete
- Inline documentation
- Type checking improvements

---

### 34. **Debug Mode**
**Status**: Basic logging  
**Priority**: Low  
**Effort**: Low

Enhanced debugging capabilities.

**Features**:
```python
client = AzureLLMClient(config=config, debug=True)

# Automatically enables:
# - Verbose logging
# - Request/response dumping
# - Performance profiling
# - Rate limiter visualization
```

---

## 📦 Deployment & Operations

### 35. **Docker Images**
**Status**: Not implemented  
**Priority**: Low  
**Effort**: Low

Official Docker images.

**Variants**:
- Slim (minimal dependencies)
- Full (with all features)
- Development (with dev tools)

---

### 36. **Kubernetes Operators**
**Status**: Not implemented  
**Priority**: Low  
**Effort**: High

Kubernetes operator for managing deployments.

**Features**:
- Auto-scaling based on load
- Rate limit coordination
- Cost budgets
- Multi-region failover

---

### 37. **Health Checks & Readiness Probes**
**Status**: Not implemented  
**Priority**: Medium  
**Effort**: Low

Endpoints for monitoring.

**Implementation**:
```python
async def health_check(self) -> dict[str, Any]:
    """Check client health."""
    return {
        "status": "healthy",
        "rate_limiter": self.rate_limiter.get_stats(),
        "cache": {"size": self.cache.size()},
        "api_connectivity": await self._check_api(),
    }
```

---

## 🎯 Specialized Use Cases

### 38. **Reranker Improvements**
**Status**: Basic logprob reranker  
**Priority**: Medium  
**Effort**: Medium

Enhanced reranking capabilities.

**Features**:
- Multiple reranking strategies (cross-encoder, colbert, etc.)
- Ensemble reranking
- Custom scoring functions
- Caching of rerank scores
- A/B testing framework

---

### 39. **RAG-Specific Utilities**
**Status**: Not implemented  
**Priority**: Medium  
**Effort**: Medium

Helper utilities for RAG pipelines.

**Features**:
```python
class RAGUtilities:
    async def chunk_text(
        self,
        text: str,
        chunk_size: int = 512,
        overlap: int = 50,
    ) -> list[str]:
        pass
    
    async def generate_hypothetical_questions(
        self,
        document: str,
        num_questions: int = 3,
    ) -> list[str]:
        """HyDE: Hypothetical Document Embeddings."""
        pass
    
    async def evaluate_context_relevance(
        self,
        query: str,
        context: str,
    ) -> float:
        """Score context relevance."""
        pass
```

---

### 40. **Multi-Turn Conversation Manager**
**Status**: Not implemented  
**Priority**: Low  
**Effort**: Medium

Conversation state management.

**Features**:
```python
class ConversationManager:
    def __init__(self, max_history: int = 10):
        self.history: list[dict] = []
    
    async def send_message(
        self,
        user_message: str,
        client: AzureLLMClient,
    ) -> str:
        """Send message and update history."""
        pass
    
    def summarize_history(self) -> str:
        """Compress old messages."""
        pass
```

---

## Implementation Priority Matrix

| Feature | Priority | Effort | Impact | Quick Win |
|---------|----------|--------|--------|-----------|
| Streaming Support | High | Medium | High | No |
| Function Calling | High | Medium | High | No |
| Async Context Manager | High | Low | Medium | ✅ Yes |
| Batch API | High | High | High | No |
| Vision Support | Medium | Medium | Medium | No |
| Advanced Caching | Medium | Medium | Medium | No |
| Sync Client | Medium | Low | Medium | ✅ Yes |
| Better Logging | Medium | Low | High | ✅ Yes |
| Config Presets | Low | Low | Low | ✅ Yes |
| CLI Tool | Low | Medium | Medium | No |

---

## Recommended Implementation Order

### Phase 1 (Quick Wins - 1-2 weeks)
1. Async context manager support
2. Better logging & observability
3. Configuration presets
4. Sync client wrapper
5. Retry strategy customization

### Phase 2 (Core Features - 1 month)
6. Streaming support
7. Function calling / tools
8. Vision model support
9. Response validation & parsing
10. Multi-model routing

### Phase 3 (Advanced Features - 2 months)
11. Batch API support
12. Advanced caching strategies
13. Prometheus metrics exporter
14. Integration tests
15. Mock server for testing

### Phase 4 (Enterprise Features - 3 months)
16. Secrets management
17. PII detection & redaction
18. Audit logging
19. Health checks
20. Cost analytics dashboard

---

## Community Input Welcome

This is a living document. Contributions, suggestions, and feedback are welcome!

**How to contribute**:
1. Open an issue to discuss new features
2. Comment on priority/effort estimates
3. Share your use cases
4. Submit PRs for implementation

**Contact**:
- GitHub Issues: https://github.com/tsoernes/azure-llm-toolkit/issues
- Discussions: https://github.com/tsoernes/azure-llm-toolkit/discussions
- Email: t.soernes@gmail.com

---

**Last Updated**: 2025-01-13  
**Version**: 1.0  
**Status**: Living Document