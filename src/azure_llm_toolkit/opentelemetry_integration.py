"""
OpenTelemetry tracing integration helpers for azure-llm-toolkit.

This module provides *optional* OpenTelemetry tracing utilities that can be used
to instrument key operations in `AzureLLMClient` and related components without
introducing a hard dependency on OpenTelemetry.

The design goals are:

- Do not break users who don't have OpenTelemetry installed.
- Provide a simple, opt-in API for tracing.
- Make it easy for library code to create spans without scattering OTEL-specific
  imports everywhere.
- Keep dependencies minimal and isolated in this module.

Typical usage
-------------

Application code (or a thin wrapper around AzureLLMClient) can use this module
as follows:

    from azure_llm_toolkit.opentelemetry_integration import (
        init_tracer_provider,
        get_tracer,
        traced_async,
    )

    # Initialize tracing once at application startup
    init_tracer_provider(service_name="my-azure-llm-app")

    tracer = get_tracer(__name__)

    @traced_async(tracer, "my_operation_name")
    async def my_async_operation(...):
        ...

Library code can also use `get_tracer(__name__)` and then wrap specific calls:

    tracer = get_tracer("azure_llm_toolkit.client")

    async with start_span(tracer, "chat_completion") as span:
        # Call Azure OpenAI here
        ...

Span attributes
---------------

When creating spans for LLM operations, we recommend attaching the following
attributes when available:

- "llm.operation"      : "chat_completion", "embed_text", "rerank", ...
- "llm.model"          : The model name used (e.g., "gpt-4o")
- "llm.success"        : bool (or 0/1)
- "llm.tokens.input"   : int
- "llm.tokens.output"  : int
- "llm.tokens.cached"  : int
- "llm.cost"           : float (if cost tracking is enabled)
- "error.type"         : exception type if an error occurred
- "error.message"      : exception message (non-sensitive)
"""

from __future__ import annotations

import functools
import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Awaitable, Callable, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Optional OpenTelemetry imports
# =============================================================================

_OTEL_AVAILABLE = False
_otel_trace = None
_otel_resource = None
_otel_sdk_trace = None
_otel_simple_exporter = None
_otel_batch_exporter = None

try:  # pragma: no cover - import behavior depends on environment
    from opentelemetry import trace as _otel_trace  # type: ignore
    from opentelemetry.sdk.resources import Resource as _otel_resource  # type: ignore
    from opentelemetry.sdk.trace import TracerProvider as _otel_sdk_trace  # type: ignore
    from opentelemetry.sdk.trace.export import (  # type: ignore
        BatchSpanProcessor as _otel_batch_exporter,
        SimpleSpanProcessor as _otel_simple_exporter,
    )

    _OTEL_AVAILABLE = True
except Exception:  # pragma: no cover - OTEL is optional
    _OTEL_AVAILABLE = False
    _otel_trace = None
    _otel_resource = None
    _otel_sdk_trace = None
    _otel_simple_exporter = None
    _otel_batch_exporter = None


def is_opentelemetry_available() -> bool:
    """
    Check whether OpenTelemetry tracing dependencies are available.

    Returns:
        True if OpenTelemetry imports succeeded, False otherwise.
    """
    return _OTEL_AVAILABLE


# =============================================================================
# Tracer initialization
# =============================================================================


def init_tracer_provider(
    service_name: str = "azure-llm-toolkit",
    use_batch_span_processor: bool = True,
    exporter: Any | None = None,
) -> None:
    """
    Initialize a global TracerProvider for OpenTelemetry.

    This is an *optional* utility. It should typically be called once from
    your application startup code (NOT from library code). If OpenTelemetry
    is not installed, this function becomes a no-op.

    Args:
        service_name:
            Logical service name used in traces (e.g., "my-app").
        use_batch_span_processor:
            Whether to use BatchSpanProcessor (recommended for production) or
            SimpleSpanProcessor (more suitable for local debugging).
        exporter:
            Optional OTEL span exporter instance. If omitted, no exporter is
            attached; you are expected to configure one separately.
            Examples include:
            - opentelemetry.sdk.trace.export.ConsoleSpanExporter()
            - opentelemetry.exporter.otlp.proto.grpc.trace_exporter.OTLPSpanExporter()
    """
    if not _OTEL_AVAILABLE:  # pragma: no cover - environment-dependent
        logger.info("OpenTelemetry is not available; tracer provider not initialized.")
        return

    try:
        resource = _otel_resource.create({"service.name": service_name})  # type: ignore[operator]
        provider = _otel_sdk_trace(resource=resource)  # type: ignore[call-arg]
        _otel_trace.set_tracer_provider(provider)  # type: ignore[call-arg]

        if exporter is not None:
            if use_batch_span_processor and _otel_batch_exporter is not None:
                processor = _otel_batch_exporter(exporter)  # type: ignore[call-arg]
            else:
                processor = _otel_simple_exporter(exporter)  # type: ignore[call-arg]
            provider.add_span_processor(processor)  # type: ignore[attr-defined]

        logger.info("OpenTelemetry tracer provider initialized (service_name=%s).", service_name)
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("Failed to initialize OpenTelemetry tracer provider: %s", e)


def get_tracer(name: str) -> Any:
    """
    Get an OpenTelemetry tracer by name.

    If OpenTelemetry is not available, returns a lightweight no-op tracer
    that exposes the minimal interface used by this module.

    Args:
        name: Name of the tracer, typically `__name__` of the calling module.

    Returns:
        A tracer object with a `start_as_current_span` method.
    """
    if _OTEL_AVAILABLE and _otel_trace is not None:  # pragma: no cover - depends on environment
        return _otel_trace.get_tracer(name)  # type: ignore[call-arg]

    return _NoOpTracer(name)


class _NoOpSpan:
    """Minimal no-op span implementation."""

    def __init__(self, name: str) -> None:
        self.name = name

    def __enter__(self) -> "_NoOpSpan":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        return None

    async def __aenter__(self) -> "_NoOpSpan":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        return None

    def set_attribute(self, key: str, value: Any) -> None:
        return None

    def record_exception(self, exc: BaseException) -> None:
        return None

    def set_status(self, status: Any) -> None:
        return None


class _NoOpSpanContextManager:
    """No-op context manager returned by _NoOpTracer.start_as_current_span."""

    def __init__(self, span_name: str) -> None:
        self._span = _NoOpSpan(span_name)

    def __enter__(self) -> _NoOpSpan:
        return self._span

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        return None

    async def __aenter__(self) -> _NoOpSpan:
        return self._span

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        return None


class _NoOpTracer:
    """Minimal no-op tracer that mimics the OTEL tracer interface."""

    def __init__(self, name: str) -> None:
        self.name = name

    def start_as_current_span(self, span_name: str, *args: Any, **kwargs: Any) -> _NoOpSpanContextManager:
        return _NoOpSpanContextManager(span_name)


# =============================================================================
# Span helpers
# =============================================================================


@asynccontextmanager
async def start_span(
    tracer: Any,
    name: str,
    attributes: Optional[dict[str, Any]] = None,
) -> AsyncGenerator[Any, None]:
    """
    Async context manager to start a span with optional attributes.

    Args:
        tracer:
            Tracer object from `get_tracer`.
        name:
            Span name, e.g. "azure_llm_toolkit.chat_completion".
        attributes:
            Optional attributes to attach to the span.

    Usage:
        tracer = get_tracer("azure_llm_toolkit.client")
        async with start_span(tracer, "chat_completion", {"llm.model": model}) as span:
            ...
    """
    if tracer is None:
        tracer = get_tracer("azure_llm_toolkit")

    # tracer.start_as_current_span works for both real OTEL tracer and _NoOpTracer
    with tracer.start_as_current_span(name) as span:
        if attributes:
            for key, value in attributes.items():
                try:
                    span.set_attribute(key, value)
                except Exception:  # pragma: no cover - defensive
                    pass
        try:
            yield span
        except Exception as exc:
            # Record exception on span if possible
            try:
                span.record_exception(exc)  # type: ignore[attr-defined]
                # In real OTEL we might set status to ERROR, but we avoid importing Status here.
            except Exception:  # pragma: no cover
                pass
            raise


# =============================================================================
# Decorators
# =============================================================================


def traced_async(
    tracer: Any,
    span_name: str,
    get_attributes: Optional[Callable[..., dict[str, Any]]] = None,
) -> Callable[[Callable[..., Awaitable[Any]]], Callable[..., Awaitable[Any]]]:
    """
    Decorator to trace an async function with OpenTelemetry.

    Args:
        tracer:
            Tracer instance obtained from `get_tracer`.
        span_name:
            Name of the span to create for each call.
        get_attributes:
            Optional function that takes the same arguments as the wrapped
            function and returns a dict of span attributes.

    Example:
        tracer = get_tracer("azure_llm_toolkit.client")

        @traced_async(tracer, "chat_completion_call", get_attributes=lambda *a, **k: {"llm.operation": "chat"})
        async def my_chat_call(...):
            ...
    """

    def decorator(func: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            attrs = get_attributes(*args, **kwargs) if get_attributes else None
            async with start_span(tracer, span_name, attrs):
                return await func(*args, **kwargs)

        return wrapper

    return decorator


# =============================================================================
# LLM-specific attribute helpers
# =============================================================================


def llm_span_attributes(
    operation: str,
    model: str | None = None,
    success: bool | None = None,
    tokens_input: int | None = None,
    tokens_output: int | None = None,
    tokens_cached: int | None = None,
    cost: float | None = None,
    error_type: str | None = None,
) -> dict[str, Any]:
    """
    Helper to build standard LLM span attributes.

    Args:
        operation: Operation name (e.g. "chat_completion", "embed_text").
        model: Model identifier.
        success: Whether the operation succeeded.
        tokens_input: Number of input tokens.
        tokens_output: Number of output tokens.
        tokens_cached: Number of cached tokens.
        cost: Monetary cost of the operation (if known).
        error_type: Error type name, if any.

    Returns:
        Dictionary of attributes ready to be passed to `start_span` or similar.
    """
    attrs: dict[str, Any] = {
        "llm.operation": operation,
    }
    if model is not None:
        attrs["llm.model"] = model
    if success is not None:
        attrs["llm.success"] = bool(success)
    if tokens_input is not None:
        attrs["llm.tokens.input"] = int(tokens_input)
    if tokens_output is not None:
        attrs["llm.tokens.output"] = int(tokens_output)
    if tokens_cached is not None:
        attrs["llm.tokens.cached"] = int(tokens_cached)
    if cost is not None:
        attrs["llm.cost"] = float(cost)
    if error_type:
        attrs["error.type"] = error_type
    return attrs


def attach_llm_result_attributes(
    span: Any,
    operation: str,
    model: str,
    usage: Any | None = None,
    cost: float | None = None,
    success: bool = True,
    error_type: str | None = None,
) -> None:
    """
    Attach LLM attributes to an existing span based on operation result.

    Args:
        span:
            Span object (real OTEL span or _NoOpSpan).
        operation:
            Operation name, e.g. "chat_completion".
        model:
            Model name.
        usage:
            Usage object from the OpenAI/Azure API response with attributes
            like `prompt_tokens`, `completion_tokens`, `total_tokens`, and
            optionally `prompt_tokens_details.cached_tokens`.
        cost:
            Monetary cost of the operation (if known).
        success:
            Whether the operation succeeded.
        error_type:
            Error type name, if any.
    """
    try:
        attrs = llm_span_attributes(
            operation=operation,
            model=model,
            success=success,
            tokens_input=getattr(usage, "prompt_tokens", None) if usage else None,
            tokens_output=getattr(usage, "completion_tokens", None) if usage else None,
            tokens_cached=(
                getattr(getattr(usage, "prompt_tokens_details", None), "cached_tokens", None)
                if usage and getattr(usage, "prompt_tokens_details", None) is not None
                else None
            ),
            cost=cost,
            error_type=error_type,
        )
        for key, value in attrs.items():
            try:
                span.set_attribute(key, value)
            except Exception:  # pragma: no cover - defensive
                pass
    except Exception as e:  # pragma: no cover - defensive
        logger.debug("Failed to attach LLM result attributes to span: %s", e)


__all__ = [
    "is_opentelemetry_available",
    "init_tracer_provider",
    "get_tracer",
    "start_span",
    "traced_async",
    "llm_span_attributes",
    "attach_llm_result_attributes",
]
