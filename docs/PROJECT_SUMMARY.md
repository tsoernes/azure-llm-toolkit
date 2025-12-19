# Azure LLM Toolkit - Project Summary

## Overview

**Azure LLM Toolkit** is a standalone Python library extracted from the `rag-mcp` project. It provides a comprehensive, production-ready wrapper around Azure OpenAI APIs with advanced features for rate limiting, cost tracking, retry logic, and more.

## Project Details

- **Repository**: https://github.com/tsoernes/azure-llm-toolkit
- **Version**: 0.1.0
- **License**: MIT
- **Python Support**: 3.10+
- **Status**: Active Development

## What Was Factored Out

The following functionality was extracted from `rag-mcp` into this standalone library:

### 1. **Core Azure Client** (`client.py`)
- Async Azure OpenAI client wrapper
- Batch embedding support with automatic splitting
- Chat completions with reasoning model support (GPT-4o, o1, etc.)
- Metadata extraction from filenames and content
- Token counting and cost estimation
- RAG-style question answering

### 2. **Rate Limiting** (`rate_limiter.py`)
- Token bucket algorithm implementation
- TPM (Tokens Per Minute) rate limiting
- RPM (Requests Per Minute) rate limiting
- Per-model rate limiter pools
- Automatic request throttling
- Statistics tracking (wait times, token usage, etc.)

### 3. **Cost Tracking** (`cost_tracker.py`)
- Cost estimation for all API operations
- Configurable pricing per model
- Category-based cost tracking (embedding, chat, indexing, etc.)
- Model-based cost breakdown
- Token usage tracking (input, output, cached)
- Pluggable cost tracker interface (in-memory, database, etc.)
- Default Norwegian Krone (NOK) pricing for common models

### 4. **Configuration** (`config.py`)
- Pydantic-based configuration model
- Environment variable loading
- Endpoint normalization
- Token encoder management
- Embedding dimension detection and caching
- Configuration validation

### 5. **Type Definitions** (`types.py`)
- `UsageInfo`: Token usage tracking
- `CostInfo`: Cost information
- `ChatCompletionResult`: Chat completion response with metadata
- `EmbeddingResult`: Embedding response with metadata
### 6. **Retry Logic**
- Exponential backoff for transient failures
- Automatic retry for rate limits, timeouts, and connection errors
- Configurable retry attempts and delays
- Detailed retry logging with payload hashing

## Key Features

### ✅ Production-Ready
- Comprehensive error handling
- Automatic retries with exponential backoff
- Rate limiting to prevent quota exhaustion
- Cost tracking for budget management

### ✅ Type-Safe
- Full type hints throughout
- Pydantic models for configuration
- Modern Python type annotations (3.10+)

### ✅ Flexible
- Pluggable cost tracker interface
- Optional rate limiting
- Configurable per-model settings
- Support for custom pricing

### ✅ Well-Documented
- Comprehensive README with examples
- API documentation in docstrings
- Migration guide for existing users
- Contributing guidelines

### ✅ Tested
- Unit test structure in place
- Async test support (pytest-asyncio)
- Type checking with basedpyright and mypy

## Architecture Highlights

### Rate Limiting Architecture
```
Client Request → Rate Limiter (TPM/RPM check) → Azure API
                      ↓
                Wait if needed
                      ↓
                Update usage stats
```

### Cost Tracking Architecture
```
API Call → Usage Info → Cost Estimator → Cost Tracker
                             ↓
                       Record cost entry
                             ↓
                       Category/Model aggregation
```

### Retry Logic Architecture
```
API Call → Transient Error?
              ↓
        Exponential Backoff
              ↓
        Retry (max 5 attempts)
              ↓
        Success or Final Error
```

## Default Pricing (NOK per 1M tokens)

| Model | Input | Output | Cached Input |
|-------|-------|--------|--------------|
| gpt-4o | 41.25 | 165.00 | 20.63 |
| gpt-4o-mini | 1.24 | 4.95 | 0.62 |
| gpt-4-turbo | 82.50 | 247.50 | 41.25 |
| o1-preview | 123.75 | 495.00 | 61.88 |
| o1-mini | 24.75 | 99.00 | 12.38 |
| text-embedding-3-large | 1.03 | - | - |
| text-embedding-3-small | 0.17 | - | - |

## Dependencies

### Core Dependencies
- `openai` - Official OpenAI Python SDK
- `tiktoken` - Token counting
- `tenacity` - Retry logic
- `pydantic` - Configuration validation
- `python-dotenv` - Environment variable loading
- `numpy` - Numerical operations for embeddings

### Development Dependencies
- `pytest` - Testing framework
- `pytest-asyncio` - Async test support
- `ruff` - Linting and formatting
- `mypy` - Type checking
- `basedpyright` - Type checking (alternative)

## Installation Options

### From PyPI (when published)
```bash
pip install azure-llm-toolkit
```

### From Git
```bash
pip install git+https://github.com/tsoernes/azure-llm-toolkit.git
```

### For Development
```bash
git clone https://github.com/tsoernes/azure-llm-toolkit.git
cd azure-llm-toolkit
pip install -e ".[dev]"
```

## Usage Example

```python
import asyncio
from azure_llm_toolkit import AzureConfig, AzureLLMClient

async def main():
    # Configure (loads from environment)
    config = AzureConfig()
    
    # Create client with cost tracking
    client = AzureLLMClient(config=config, enable_rate_limiting=True)
    
    # Embed texts
    result = await client.embed_texts(["Hello", "World"])
    print(f"Generated {len(result.embeddings)} embeddings")
    
    # Chat completion
    response = await client.chat_completion(
        messages=[{"role": "user", "content": "What is AI?"}],
        system_prompt="You are a helpful assistant.",
    )
    print(f"Response: {response.content}")

asyncio.run(main())
```

## Repository Structure

```
azure-llm-toolkit/
├── src/
│   └── azure_llm_toolkit/
│       ├── __init__.py          # Public API exports
│       ├── client.py            # Main Azure LLM client
│       ├── config.py            # Configuration management
│       ├── cost_tracker.py      # Cost tracking utilities
│       ├── rate_limiter.py      # Rate limiting implementation
│       └── types.py             # Type definitions
├── examples/
│   └── basic_usage.py           # Comprehensive examples
├── tests/                       # Test suite (to be expanded)
├── pyproject.toml               # Project metadata and dependencies
├── README.md                    # Main documentation
├── CONTRIBUTING.md              # Contributing guidelines
├── MIGRATION_GUIDE.md           # Migration guide for rag-mcp users
├── LICENSE                      # MIT License
└── .env.example                 # Environment variable template
```

## What's NOT Included

This library focuses on Azure OpenAI client functionality. The following remain in `rag-mcp`:

- Vector storage (FAISS, DuckDB)
- Full-text search (FTS5)
- Document indexing and chunking
- File watching and auto-indexing
- MCP server implementation
- RAG-specific storage and retrieval logic

## Benefits of Extraction

### For azure-llm-toolkit
1. **Focused scope**: Single responsibility (Azure OpenAI client)
2. **Reusability**: Can be used in any Python project
3. **Independent versioning**: Updates don't affect rag-mcp
4. **Easier testing**: Isolated test suite
5. **Better documentation**: Dedicated docs

### For rag-mcp
1. **Cleaner architecture**: Separation of concerns
2. **Smaller codebase**: Easier to maintain
3. **Reduced complexity**: Focus on RAG-specific logic
4. **Dependency management**: Clear separation

### For Users
1. **Flexibility**: Use Azure client without RAG overhead
2. **Modularity**: Pick and choose what you need
3. **Community**: More focused contributions
4. **Examples**: Clear, isolated examples

## Next Steps

### Immediate (v0.1.x)
- [x] Core functionality extracted
- [x] Documentation complete
- [x] Examples added
- [x] GitHub repository created
- [ ] Publish to PyPI
- [ ] Add unit tests
- [ ] Set up CI/CD

### Short-term (v0.2.x)
- [ ] Streaming support for chat completions
- [ ] Function calling support
- [ ] Vision model support (image inputs)
- [ ] Batch API support
- [ ] Enhanced logging options

### Medium-term (v0.3.x)
- [ ] Sync client wrapper (for non-async code)
- [ ] Connection pooling optimization
- [ ] Advanced cost analytics
- [ ] Performance profiling tools
- [ ] Metrics exporter (Prometheus, etc.)

### Long-term (v1.0.x)
- [ ] Support for other Azure AI services
- [ ] Plugin system for custom extensions
- [ ] Advanced caching strategies
- [ ] Multi-region failover
- [ ] Enterprise features (audit logging, etc.)

## Migrating rag-mcp to Use This Library

A comprehensive migration guide is available in `MIGRATION_GUIDE.md`. Key steps:

1. Add `azure-llm-toolkit` as dependency
2. Update imports from `rag_mcp.azure_client` to `azure_llm_toolkit`
3. Replace `AzureClients` with `AzureLLMClient`
4. Update configuration from `Settings` to `AzureConfig`
5. Update API calls (return types changed to dataclasses)
6. Implement custom `CostTracker` if needed for database integration

## Contributing

Contributions are welcome! See `CONTRIBUTING.md` for guidelines.

Key areas for contribution:
- Additional tests
- Documentation improvements
- Example scripts
- Bug fixes
- Feature requests

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- Extracted from the `rag-mcp` project
- Built on the official OpenAI Python SDK
- Uses tiktoken for accurate token counting
- Inspired by production needs for robust Azure OpenAI clients

## Support

- Issues: https://github.com/tsoernes/azure-llm-toolkit/issues
- Discussions: https://github.com/tsoernes/azure-llm-toolkit/discussions
- Email: t.soernes@gmail.com

---

**Created**: 2024-12-08  
**Last Updated**: 2024-12-08  
**Status**: Active Development  
**Maintainer**: Torstein Sørnes (@tsoernes)