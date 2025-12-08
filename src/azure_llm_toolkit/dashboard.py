# code/azure-llm-toolkit/src/azure_llm_toolkit/dashboard.py
"""
Simple rate limit and metrics dashboard utilities.

This module provides lightweight, text-based dashboard helpers for:

- Rate limiter statistics (TPM/RPM utilization)
- Circuit breaker state and metrics
- Operation-level metrics (latency, tokens, errors)

The goal is to make it easy to inspect the current state of the system
from a REPL, script, or CLI, without pulling in heavy dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable

from .circuit_breaker import CircuitBreakerMetrics, CircuitState
from .metrics import AggregatedMetrics
from .rate_limiter import RateLimiter, RateLimiterPool


@dataclass
class RateLimiterSnapshot:
    """Snapshot of rate limiter state for a single model."""

    model: str
    rpm_available: float
    rpm_limit: int
    tpm_available: float
    tpm_limit: int
    total_requests: int
    total_tokens: int
    total_wait_time_seconds: float


def _format_header(title: str, width: int = 80) -> str:
    title = title.strip()
    if len(title) + 2 >= width:
        return f"== {title} =="

    pad_total = width - (len(title) + 4)
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    return f"{'=' * pad_left}  {title}  {'=' * pad_right}"


def _format_percentage(value: float | None) -> str:
    if value is None:
        return "   n/a"
    return f"{value:6.1f}%"


def _format_duration(seconds: float | None) -> str:
    if not seconds or seconds <= 0:
        return "0.0s"
    if seconds < 1:
        return f"{seconds * 1000:.1f}ms"
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    rem = seconds % 60
    if minutes < 60:
        return f"{minutes}m {rem:.0f}s"
    hours = minutes // 60
    minutes = minutes % 60
    return f"{hours}h {minutes}m"


def _format_state_duration(durations: dict[str, float]) -> str:
    total = sum(durations.values())
    if total <= 0:
        return "closed:   0.0%, open:   0.0%, half_open:   0.0%"

    def pct(key: str) -> str:
        return _format_percentage((durations.get(key, 0.0) / total) * 100.0)

    return f"closed: {pct('closed')}, open: {pct('open')}, half_open: {pct('half_open')}"


def snapshot_rate_limiter(model: str, limiter: RateLimiter) -> RateLimiterSnapshot:
    """
    Create a snapshot from a RateLimiter instance.

    Args:
        model: Model/deployment identifier
        limiter: RateLimiter instance

    Returns:
        RateLimiterSnapshot with relevant statistics.
    """
    stats = limiter.get_stats()
    return RateLimiterSnapshot(
        model=model,
        rpm_available=stats.get("rpm_available", 0.0),
        rpm_limit=int(stats.get("rpm_limit", limiter.rpm_limit)),
        tpm_available=stats.get("tpm_available", 0.0),
        tpm_limit=int(stats.get("tpm_limit", limiter.tpm_limit)),
        total_requests=int(stats.get("total_requests", 0)),
        total_tokens=int(stats.get("total_tokens", 0)),
        total_wait_time_seconds=float(stats.get("total_wait_time_seconds", 0.0)),
    )


def render_rate_limiter_snapshot(snapshot: RateLimiterSnapshot, width: int = 80) -> str:
    """
    Render a single RateLimiterSnapshot as human-readable text.
    """
    rpm_used = snapshot.rpm_limit - snapshot.rpm_available
    tpm_used = snapshot.tpm_limit - snapshot.tpm_available

    rpm_pct = (rpm_used / snapshot.rpm_limit * 100.0) if snapshot.rpm_limit > 0 else None
    tpm_pct = (tpm_used / snapshot.tpm_limit * 100.0) if snapshot.tpm_limit > 0 else None

    lines: list[str] = []
    lines.append(_format_header(f"Rate Limiter - {snapshot.model}", width))
    lines.append(f"Model:            {snapshot.model}")
    lines.append("")
    lines.append("Requests:")
    lines.append(f"  Total requests: {snapshot.total_requests:,}")
    lines.append(f"  Total tokens:   {snapshot.total_tokens:,}")
    lines.append(f"  Wait time:      {_format_duration(snapshot.total_wait_time_seconds)}")
    lines.append("")
    lines.append("Limits:")
    lines.append(f"  RPM: {rpm_used:,.1f}/{snapshot.rpm_limit:,} ({_format_percentage(rpm_pct)})")
    lines.append(f"  TPM: {tpm_used:,.0f}/{snapshot.tpm_limit:,} ({_format_percentage(tpm_pct)})")
    return "\n".join(lines)


def render_rate_limiter_pool(pool: RateLimiterPool, width: int = 80) -> str:
    """
    Render all limiters from a RateLimiterPool.

    Args:
        pool: RateLimiterPool instance
        width: Line width

    Returns:
        Multi-line string with pool overview.
    """
    stats = pool.get_all_stats()
    lines: list[str] = [_format_header("Rate Limiter Pool", width)]

    if not stats:
        lines.append("No limiters registered yet.")
        return "\n".join(lines)

    for model, _stats in stats.items():
        snapshot = snapshot_rate_limiter(model, pool._limiters[model])
        lines.append("")
        lines.append(render_rate_limiter_snapshot(snapshot, width=width))

    return "\n".join(lines)


def render_circuit_breaker(metrics: CircuitBreakerMetrics, name: str = "default", width: int = 80) -> str:
    """
    Render a circuit breaker dashboard section.

    Args:
        metrics: CircuitBreakerMetrics instance
        name: Logical name (e.g., model or endpoint)
        width: Line width

    Returns:
        Multi-line string with circuit breaker status.
    """
    lines: list[str] = [_format_header(f"Circuit Breaker - {name}", width)]

    lines.append(f"State:            {metrics.state.value.upper()}")
    lines.append(f"Failures:         {metrics.failure_count}")
    lines.append(f"Successes:        {metrics.success_count}")
    lines.append(f"Total requests:   {metrics.total_requests}")
    lines.append(f"Rejected:         {metrics.rejected_requests}")
    if metrics.last_failure_time:
        dt = datetime.fromtimestamp(metrics.last_failure_time)
        lines.append(f"Last failure at:  {dt.isoformat(timespec='seconds')}")
    else:
        lines.append("Last failure at:  never")

    dt_state = datetime.fromtimestamp(metrics.last_state_change)
    lines.append(f"Last state change:{dt_state.isoformat(timespec='seconds')}")
    lines.append("")
    lines.append("State durations:")
    lines.append(f"  {_format_state_duration(metrics.state_durations)}")

    # Simple health indicator
    if metrics.state == CircuitState.CLOSED:
        health = "HEALTHY"
    elif metrics.state == CircuitState.HALF_OPEN:
        health = "DEGRADED"
    else:
        health = "UNHEALTHY"

    lines.append("")
    lines.append(f"Health: {health}")
    return "\n".join(lines)


def render_multi_circuit_breaker(
    all_metrics: dict[str, CircuitBreakerMetrics],
    width: int = 80,
) -> str:
    """
    Render multiple circuit breakers.

    Args:
        all_metrics: Mapping name -> CircuitBreakerMetrics
        width: Line width

    Returns:
        Multi-line string with all circuit breaker statuses.
    """
    lines: list[str] = [_format_header("Circuit Breakers", width)]

    if not all_metrics:
        lines.append("No circuit breakers registered yet.")
        return "\n".join(lines)

    for name, metrics in all_metrics.items():
        lines.append("")
        lines.append(render_circuit_breaker(metrics, name=name, width=width))

    return "\n".join(lines)


def render_operation_metrics(agg: AggregatedMetrics, width: int = 80) -> str:
    """
    Render aggregated operation-level metrics.

    Args:
        agg: AggregatedMetrics from MetricsCollector
        width: Line width

    Returns:
        Multi-line string overview.
    """
    lines: list[str] = [_format_header("Operation Metrics", width)]

    if agg.total_requests == 0:
        lines.append("No metrics recorded yet.")
        return "\n".join(lines)

    lines.append("Requests:")
    lines.append(f"  Total:          {agg.total_requests:,}")
    lines.append(f"  Successful:     {agg.successful_requests:,}")
    lines.append(f"  Failed:         {agg.failed_requests:,}")
    lines.append("")
    lines.append("Tokens:")
    lines.append(f"  Input:          {agg.total_tokens_input:,}")
    lines.append(f"  Output:         {agg.total_tokens_output:,}")
    lines.append(f"  Cached:         {agg.total_tokens_cached:,}")
    lines.append("")
    lines.append("Latency:")
    lines.append(f"  Avg:            {_format_duration(agg.avg_duration_seconds)}")
    lines.append(f"  Min:            {_format_duration(agg.min_duration_seconds)}")
    lines.append(f"  Max:            {_format_duration(agg.max_duration_seconds)}")
    lines.append("")
    lines.append("Requests by model:")
    if agg.requests_by_model:
        for model, count in sorted(agg.requests_by_model.items(), key=lambda kv: kv[1], reverse=True):
            pct = (count / agg.total_requests) * 100.0 if agg.total_requests > 0 else 0.0
            lines.append(f"  {model:25s} {count:8d} ({pct:5.1f}%)")
    else:
        lines.append("  (none)")
    lines.append("")
    lines.append("Requests by operation:")
    if agg.requests_by_operation:
        for op, count in sorted(agg.requests_by_operation.items(), key=lambda kv: kv[1], reverse=True):
            pct = (count / agg.total_requests) * 100.0 if agg.total_requests > 0 else 0.0
            lines.append(f"  {op:25s} {count:8d} ({pct:5.1f}%)")
    else:
        lines.append("  (none)")
    lines.append("")
    lines.append("Errors by type:")
    if agg.errors_by_type:
        for err, count in sorted(agg.errors_by_type.items(), key=lambda kv: kv[1], reverse=True):
            lines.append(f"  {err:25s} {count:8d}")
    else:
        lines.append("  (none)")

    return "\n".join(lines)


def render_full_dashboard(
    *,
    rate_limiter_pool: RateLimiterPool | None = None,
    circuit_metrics: dict[str, CircuitBreakerMetrics] | None = None,
    aggregated_metrics: AggregatedMetrics | None = None,
    width: int = 80,
) -> str:
    """
    Render a full text dashboard combining rate limit, circuit breaker,
    and operation metrics sections.
    """
    sections: list[str] = []

    if rate_limiter_pool is not None:
        sections.append(render_rate_limiter_pool(rate_limiter_pool, width=width))

    if circuit_metrics is not None:
        sections.append(render_multi_circuit_breaker(circuit_metrics, width=width))

    if aggregated_metrics is not None:
        sections.append(render_operation_metrics(aggregated_metrics, width=width))

    if not sections:
        return "No dashboard data available."

    return "\n\n".join(sections)


__all__ = [
    "RateLimiterSnapshot",
    "snapshot_rate_limiter",
    "render_rate_limiter_snapshot",
    "render_rate_limiter_pool",
    "render_circuit_breaker",
    "render_multi_circuit_breaker",
    "render_operation_metrics",
    "render_full_dashboard",
]
