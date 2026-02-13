# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.4] - 2026-02-13

### Added

- **Vision model support**
  - Support for vision messages with images (URL and base64)
  - Added `_extract_text_from_content()` helper to safely extract text from message content
  - Handles both simple string content and vision message format with image parts
  - Token counting now works correctly with vision messages
  - Updated type hints to use `Any` for message content to support both text and vision formats
  - Comprehensive test suite for vision integration

### Fixed

- **Token counting error with vision messages**
  - Fixed `TypeError: expected string or buffer` when passing images to models
  - `count_message_tokens()` now properly handles vision message content
  - Rate limiter estimation now works with vision messages

### Changed

- Updated message type hints from `list[dict[str, str]]` to `list[dict[str, Any]]` to support vision content
- Enhanced `count_message_tokens()` to extract and count only text portions of vision messages

## [0.2.3] - 2026-02-13

### Added

- **Automatic GPT-5 parameter conversion**
  - Automatically converts `max_tokens` to `max_completion_tokens` for GPT-5 models
  - Automatically removes `temperature` parameter for GPT-5 models (not supported)
  - Case-insensitive detection of "gpt-5" in model names
  - Logs warnings when parameters are converted or removed
  - Applies to both `chat_completion()` and `chat_completion_stream()` methods
  - Preserves all other parameters unchanged
  - Comprehensive test coverage for parameter conversion

### Changed

- Enhanced error handling with proactive parameter conversion for GPT-5 models
- Improved developer experience by eliminating manual parameter adjustments

## [0.2.2] - 2026-01-09

### Fixed

- **Repository URL corrections**
  - Updated all GitHub URLs from `torsteinsornes` to `tsoernes`
  - Fixed URLs in `pyproject.toml`, `README.md`, `CHANGELOG.md`, and `CONTRIBUTING.md`
  - Ensures consistency across all documentation and package metadata

### Documentation

- Enhanced README with clearer examples section
- Added direct links to example files on GitHub

## [0.2.1] - 2026-01-08

### Added

- **Reasoning token tracking and logging**
  - Added `reasoning_tokens` field to `UsageInfo` type
  - Automatic extraction from `completion_tokens_details.reasoning_tokens`
  - Enhanced DEBUG logs show reasoning tokens: `tokens=X, reasoning_tokens=Y`
  - Metrics system tracks reasoning tokens (OperationMetrics, AggregatedMetrics)
  - Prometheus export includes `token_type='reasoning'`
  - Cost tracking metadata includes reasoning tokens
  - Full backward compatibility (defaults to 0 for non-reasoning models)

- **Comprehensive example**
  - New `examples/reasoning_tokens_example.py` (287 lines)
  - Demonstrates reasoning token tracking with o1/GPT-5 models
  - Shows metrics collection and cost analysis
  - Compares reasoning efforts (low/medium/high)
  - Includes Prometheus export example

### Changed

- Updated README.md to mention reasoning token tracking feature
- Enhanced chat completion logs to include reasoning token counts

## [0.2.0] - 2026-01-08

### ⚠️ Breaking Changes

- **Default API timeout changed from 60 seconds to infinite (`None`)**
  - Recommended for reasoning models (o1, GPT-5) that can take 30+ seconds
  - Prevents false timeout failures on complex requests
  - Migration: Set `AZURE_TIMEOUT_SECONDS=60` in `.env` to restore old behavior

### Added

- **Enhanced logging for timeout debugging and performance monitoring**
  - Client initialization logging shows timeout config, retries, and enabled features
  - Request timing logs track all API request durations (DEBUG level)
  - Performance warnings for slow requests with model-aware thresholds (INFO level)
  - Specific APITimeoutError handling with actionable error messages (ERROR level)
  
- **Improved retry attempt logging**
  - Renamed `wait` to `retry_backoff_delay` for clarity
  - Added `api_timeout` field showing current timeout configuration
  - Clear distinction between retry backoff delay and API timeout duration

- **Comprehensive documentation**
  - New `docs/TIMEOUT_AND_LOGGING.md` guide (449 lines)
  - Covers timeout configuration, logging improvements, best practices
  - Includes troubleshooting guide and example log outputs
  - Breaking change migration instructions

### Changed

- `timeout_seconds` type changed from `float` to `float | None` in `AzureConfig`
- Default `timeout_seconds` value: `60` → `None` (infinite)
- Updated `.env.example` with infinite timeout as default
- Updated `README.md` with new timeout behavior documentation

### Fixed

- Misleading log messages that confused retry backoff delay with API timeout
- Timeout errors now include configuration context for easier debugging

### Performance

- Request timing logs help identify slow requests and performance issues
- Model-aware thresholds: 30s for reasoning models, 10s for others
- Informational warnings distinguish between expected and unexpected slowness

## [0.1.6] - 2026-01-08

### Changed

- Updated GPT-5 pricing to latest Global rates

## [0.1.5] - 2025-12-XX

### Added

- Initial stable release with core features
- Rate limiting with token-based tracking
- Cost estimation and tracking
- Disk-based caching for embeddings and chat completions
- Batch embedding support with Polars
- Retry logic with exponential backoff
- Prometheus metrics integration
- OpenTelemetry support

---

## Version History Summary

- **0.2.1**: Reasoning token tracking and logging for o1/GPT-5 models
- **0.2.0**: Breaking change - infinite timeout default, enhanced logging
- **0.1.6**: Pricing updates
- **0.1.5**: Initial stable release

[0.2.4]: https://github.com/tsoernes/azure-llm-toolkit/compare/v0.2.3...v0.2.4
[0.2.3]: https://github.com/tsoernes/azure-llm-toolkit/compare/v0.2.2...v0.2.3
[0.2.2]: https://github.com/tsoernes/azure-llm-toolkit/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/tsoernes/azure-llm-toolkit/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/tsoernes/azure-llm-toolkit/compare/v0.1.6...v0.2.0
[0.1.6]: https://github.com/tsoernes/azure-llm-toolkit/compare/v0.1.5...v0.1.6
[0.1.5]: https://github.com/tsoernes/azure-llm-toolkit/releases/tag/v0.1.5