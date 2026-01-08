"""Metrics and telemetry for Azure LLM operations.

This module provides metrics collection and export capabilities for monitoring
LLM operations, including request counts, latencies, token usage, costs, and errors.

Supports multiple backends:
- Prometheus (via prometheus_client)
- OpenTelemetry
- Custom callbacks
- In-memory aggregation
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

logger = logging.getLogger(__name__)

from opentelemetry import metrics as otel_metrics
from prometheus_client import Counter, Gauge, Histogram, Summary

PROMETHEUS_AVAILABLE = True
OPENTELEMETRY_AVAILABLE = True


@dataclass
class OperationMetrics:
    """Metrics for a single operation."""

    operation: str  # e.g., "embed_text", "chat_completion"
    model: str
    success: bool
    duration_seconds: float
    tokens_input: int = 0
    tokens_output: int = 0
    tokens_cached: int = 0
    tokens_reasoning: int = 0
    cost: float = 0.0
    error_type: str | None = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AggregatedMetrics:
    """Aggregated metrics over a time period."""

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_tokens_input: int = 0
    total_tokens_output: int = 0
    total_tokens_cached: int = 0
    total_tokens_reasoning: int = 0
    total_cost: float = 0.0
    avg_duration_seconds: float = 0.0
    min_duration_seconds: float = float("inf")
    max_duration_seconds: float = 0.0
    errors_by_type: dict[str, int] = field(default_factory=dict)
    requests_by_model: dict[str, int] = field(default_factory=dict)
    requests_by_operation: dict[str, int] = field(default_factory=dict)


class MetricsCollector:
    """
    Base metrics collector.

    Collects metrics in-memory and provides aggregation.
    Can be extended with custom backends.
    """

    def __init__(self, max_history: int = 10000):
        """
        Initialize metrics collector.

        Args:
            max_history: Maximum number of operations to keep in memory
        """
        self.max_history = max_history
        self._operations: list[OperationMetrics] = []
        self._callbacks: list[Callable[[OperationMetrics], None]] = []

    def record(self, metrics: OperationMetrics) -> None:
        """
        Record operation metrics.

        Args:
            metrics: Operation metrics to record
        """
        self._operations.append(metrics)

        # Trim history if needed
        if len(self._operations) > self.max_history:
            self._operations = self._operations[-self.max_history :]

        # Call registered callbacks
        for callback in self._callbacks:
            try:
                callback(metrics)
            except Exception as e:
                logger.warning(f"Metrics callback failed: {e}")

    def add_callback(self, callback: Callable[[OperationMetrics], None]) -> None:
        """
        Add a callback to be called on each metric record.

        Args:
            callback: Function that takes OperationMetrics
        """
        self._callbacks.append(callback)

    def get_aggregated(
        self,
        operation: str | None = None,
        model: str | None = None,
        since: datetime | None = None,
    ) -> AggregatedMetrics:
        """
        Get aggregated metrics.

        Args:
            operation: Optional filter by operation type
            model: Optional filter by model
            since: Optional filter by time (only operations after this time)

        Returns:
            Aggregated metrics
        """
        ops = self._operations

        # Apply filters
        if operation:
            ops = [op for op in ops if op.operation == operation]
        if model:
            ops = [op for op in ops if op.model == model]
        if since:
            ops = [op for op in ops if op.timestamp >= since]

        if not ops:
            return AggregatedMetrics()

        agg = AggregatedMetrics()
        agg.total_requests = len(ops)
        agg.successful_requests = sum(1 for op in ops if op.success)
        agg.failed_requests = sum(1 for op in ops if not op.success)
        agg.total_tokens_input = sum(op.tokens_input for op in ops)
        agg.total_tokens_output = sum(op.tokens_output for op in ops)
        agg.total_tokens_cached = sum(op.tokens_cached for op in ops)
        agg.total_tokens_reasoning = sum(op.tokens_reasoning for op in ops)
        agg.total_cost = sum(op.cost for op in ops)

        durations = [op.duration_seconds for op in ops]
        agg.avg_duration_seconds = sum(durations) / len(durations) if durations else 0.0
        agg.min_duration_seconds = min(durations) if durations else 0.0
        agg.max_duration_seconds = max(durations) if durations else 0.0

        # Count errors by type
        for op in ops:
            if not op.success and op.error_type:
                agg.errors_by_type[op.error_type] = agg.errors_by_type.get(op.error_type, 0) + 1

        # Count requests by model
        for op in ops:
            agg.requests_by_model[op.model] = agg.requests_by_model.get(op.model, 0) + 1

        # Count requests by operation
        for op in ops:
            agg.requests_by_operation[op.operation] = agg.requests_by_operation.get(op.operation, 0) + 1

        return agg

    def clear(self) -> None:
        """Clear all collected metrics."""
        self._operations.clear()

    def get_recent(self, count: int = 100) -> list[OperationMetrics]:
        """
        Get most recent operations.

        Args:
            count: Number of recent operations to return

        Returns:
            List of recent operations
        """
        return self._operations[-count:]


class PrometheusMetrics:
    """
    Prometheus metrics exporter.

    Exports metrics in Prometheus format for scraping.
    """

    def __init__(self, namespace: str = "azure_llm"):
        """
        Initialize Prometheus metrics.

        Args:
            namespace: Metric namespace prefix
        """
        if not PROMETHEUS_AVAILABLE:
            raise ImportError("prometheus_client not installed. Install with: pip install prometheus-client")

        self.namespace = namespace

        # Request counters
        self.requests_total = Counter(
            f"{namespace}_requests_total",
            "Total number of LLM requests",
            ["operation", "model", "status"],
        )

        # Token counters
        self.tokens_total = Counter(
            f"{namespace}_tokens_total",
            "Total number of tokens processed",
            ["operation", "model", "token_type"],
        )

        # Cost counter
        self.cost_total = Counter(
            f"{namespace}_cost_total",
            "Total cost in currency units",
            ["operation", "model"],
        )

        # Duration histogram
        self.duration_seconds = Histogram(
            f"{namespace}_duration_seconds",
            "Request duration in seconds",
            ["operation", "model"],
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
        )

        # Current active requests
        self.active_requests = Gauge(
            f"{namespace}_active_requests",
            "Number of currently active requests",
            ["operation", "model"],
        )

    def record(self, metrics: OperationMetrics) -> None:
        """
        Record metrics in Prometheus format.

        Args:
            metrics: Operation metrics to record
        """
        status = "success" if metrics.success else "error"

        # Record request
        self.requests_total.labels(
            operation=metrics.operation,
            model=metrics.model,
            status=status,
        ).inc()

        # Record tokens
        if metrics.tokens_input > 0:
            self.tokens_total.labels(
                operation=metrics.operation,
                model=metrics.model,
                token_type="input",
            ).inc(metrics.tokens_input)

        if metrics.tokens_output > 0:
            self.tokens_total.labels(
                operation=metrics.operation,
                model=metrics.model,
                token_type="output",
            ).inc(metrics.tokens_output)

        if metrics.tokens_cached > 0:
            self.tokens_total.labels(
                operation=metrics.operation,
                model=metrics.model,
                token_type="cached",
            ).inc(metrics.tokens_cached)

        if metrics.tokens_reasoning > 0:
            self.tokens_total.labels(
                operation=metrics.operation,
                model=metrics.model,
                token_type="reasoning",
            ).inc(metrics.tokens_reasoning)

        # Record cost
        if metrics.cost > 0:
            self.cost_total.labels(
                operation=metrics.operation,
                model=metrics.model,
            ).inc(metrics.cost)

        # Record duration
        self.duration_seconds.labels(
            operation=metrics.operation,
            model=metrics.model,
        ).observe(metrics.duration_seconds)


class MetricsTracker:
    """
    High-level metrics tracker with context manager support.

    Example:
        >>> tracker = MetricsTracker(collector)
        >>>
        >>> async with tracker.track("chat_completion", model="gpt-4o"):
        >>>     result = await client.chat_completion(...)
        >>>     tracker.set_tokens(input=100, output=50)
        >>>     tracker.set_cost(0.05)
    """

    def __init__(self, collector: MetricsCollector):
        """
        Initialize metrics tracker.

        Args:
            collector: Metrics collector to use
        """
        self.collector = collector
        self._operation: str | None = None
        self._model: str | None = None
        self._start_time: float | None = None
        self._tokens_input = 0
        self._tokens_output = 0
        self._tokens_cached = 0
        self._tokens_reasoning = 0
        self._cost = 0.0
        self._metadata: dict[str, Any] = {}

    def track(self, operation: str, model: str) -> MetricsTracker:
        """
        Start tracking an operation.

        Args:
            operation: Operation name
            model: Model name

        Returns:
            Self for context manager
        """
        self._operation = operation
        self._model = model
        self._start_time = time.time()
        self._tokens_input = 0
        self._tokens_output = 0
        self._tokens_cached = 0
        self._tokens_reasoning = 0
        self._cost = 0.0
        self._metadata = {}
        return self

    def set_tokens(self, input: int = 0, output: int = 0, cached: int = 0, reasoning: int = 0) -> None:
        """Set token counts."""
        self._tokens_input = input
        self._tokens_output = output
        self._tokens_cached = cached
        self._tokens_reasoning = reasoning

    def set_cost(self, cost: float) -> None:
        """Set cost."""
        self._cost = cost

    def add_metadata(self, key: str, value: Any) -> None:
        """Add metadata."""
        self._metadata[key] = value

    def __enter__(self):
        """Sync context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Sync context manager exit."""
        if self._operation and self._model and self._start_time:
            duration = time.time() - self._start_time
            metrics = OperationMetrics(
                operation=self._operation,
                model=self._model,
                success=exc_type is None,
                duration_seconds=duration,
                tokens_input=self._tokens_input,
                tokens_output=self._tokens_output,
                tokens_cached=self._tokens_cached,
                tokens_reasoning=self._tokens_reasoning,
                cost=self._cost,
                error_type=exc_type.__name__ if exc_type else None,
                metadata=self._metadata,
            )
            self.collector.record(metrics)
        return False

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._operation and self._model and self._start_time:
            duration = time.time() - self._start_time
            metrics = OperationMetrics(
                operation=self._operation,
                model=self._model,
                success=exc_type is None,
                duration_seconds=duration,
                tokens_input=self._tokens_input,
                tokens_output=self._tokens_output,
                tokens_cached=self._tokens_cached,
                tokens_reasoning=self._tokens_reasoning,
                cost=self._cost,
                error_type=exc_type.__name__ if exc_type else None,
                metadata=self._metadata,
            )
            self.collector.record(metrics)
        return False


def create_collector_with_prometheus(namespace: str = "azure_llm") -> MetricsCollector:
    """
    Create a metrics collector with Prometheus export.

    Args:
        namespace: Prometheus metric namespace

    Returns:
        MetricsCollector configured with Prometheus export
    """
    if not PROMETHEUS_AVAILABLE:
        logger.warning("Prometheus not available, returning basic collector")
        return MetricsCollector()

    collector = MetricsCollector()
    prometheus = PrometheusMetrics(namespace=namespace)
    collector.add_callback(prometheus.record)

    return collector


__all__ = [
    "MetricsCollector",
    "MetricsTracker",
    "OperationMetrics",
    "AggregatedMetrics",
    "PrometheusMetrics",
    "create_collector_with_prometheus",
    "PROMETHEUS_AVAILABLE",
    "OPENTELEMETRY_AVAILABLE",
]
