# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

- **0.2.0**: Breaking change - infinite timeout default, enhanced logging
- **0.1.6**: Pricing updates
- **0.1.5**: Initial stable release

[0.2.0]: https://github.com/torsteinsornes/azure-llm-toolkit/compare/v0.1.6...v0.2.0
[0.1.6]: https://github.com/torsteinsornes/azure-llm-toolkit/compare/v0.1.5...v0.1.6
[0.1.5]: https://github.com/torsteinsornes/azure-llm-toolkit/releases/tag/v0.1.5