"""
Health checks and readiness probes for Azure LLM Toolkit.

This module provides health check functionality for monitoring the client's
operational status, useful for production deployments, Kubernetes probes,
and service monitoring.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

from .client import AzureLLMClient


class HealthStatus(str, Enum):
    """Health status values."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class ComponentHealth:
    """Health status for a single component."""

    name: str
    status: HealthStatus
    message: str | None = None
    details: dict[str, Any] | None = None
    latency_ms: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result: dict[str, Any] = {
            "name": self.name,
            "status": self.status.value,
        }
        if self.message:
            result["message"] = self.message
        if self.details:
            result["details"] = self.details
        if self.latency_ms is not None:
            result["latency_ms"] = round(self.latency_ms, 2)
        return result


@dataclass
class HealthCheckResult:
    """Overall health check result."""

    status: HealthStatus
    timestamp: datetime
    components: list[ComponentHealth]
    message: str | None = None
    version: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "status": self.status.value,
            "timestamp": self.timestamp.isoformat(),
            "components": [c.to_dict() for c in self.components],
            "message": self.message,
            "version": self.version,
        }

    @property
    def is_healthy(self) -> bool:
        """Check if overall status is healthy."""
        return self.status == HealthStatus.HEALTHY

    @property
    def is_ready(self) -> bool:
        """Check if ready to accept requests (healthy or degraded)."""
        return self.status in (HealthStatus.HEALTHY, HealthStatus.DEGRADED)


class HealthChecker:
    """
    Health checker for AzureLLMClient.

    Performs comprehensive health checks including:
    - API connectivity
    - Rate limiter status
    - Cache status
    - Cost tracker status
    - Overall system health

    Example:
        >>> client = AzureLLMClient(config=config)
        >>> checker = HealthChecker(client)
        >>>
        >>> # Perform health check
        >>> result = await checker.check_health()
        >>> print(result.status)
        'healthy'
        >>>
        >>> # Check readiness
        >>> is_ready = await checker.check_readiness()
        >>> print(is_ready)
        True
    """

    def __init__(self, client: AzureLLMClient, version: str | None = None):
        """
        Initialize health checker.

        Args:
            client: AzureLLMClient instance to monitor
            version: Optional version string to include in health checks
        """
        self.client = client
        self.version = version

    async def check_api_connectivity(self) -> ComponentHealth:
        """
        Check API connectivity with a lightweight request.

        Returns:
            ComponentHealth for API connectivity
        """
        start_time = time.time()

        try:
            # Try to get a very small embedding to test connectivity
            test_text = "health check"
            await self.client.embed_text(test_text, track_cost=False)

            latency_ms = (time.time() - start_time) * 1000

            if latency_ms < 1000:
                status = HealthStatus.HEALTHY
                message = "API is responsive"
            elif latency_ms < 5000:
                status = HealthStatus.DEGRADED
                message = f"API is slow ({latency_ms:.0f}ms)"
            else:
                status = HealthStatus.DEGRADED
                message = f"API is very slow ({latency_ms:.0f}ms)"

            return ComponentHealth(
                name="api_connectivity",
                status=status,
                message=message,
                latency_ms=latency_ms,
            )

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return ComponentHealth(
                name="api_connectivity",
                status=HealthStatus.UNHEALTHY,
                message=f"API error: {str(e)}",
                latency_ms=latency_ms,
            )

    def check_rate_limiter(self) -> ComponentHealth:
        """
        Check rate limiter status.

        Returns:
            ComponentHealth for rate limiter
        """
        if not self.client.enable_rate_limiting or not self.client.rate_limiter_pool:
            return ComponentHealth(
                name="rate_limiter",
                status=HealthStatus.HEALTHY,
                message="Rate limiting disabled",
            )

        try:
            # Get stats from rate limiter pool (access attributes safely)
            pool = self.client.rate_limiter_pool
            limiters = getattr(pool, "limiters", None)  # type: ignore[attr-defined]

            if not limiters:
                return ComponentHealth(
                    name="rate_limiter",
                    status=HealthStatus.HEALTHY,
                    message="No active rate limiters",
                )

            # Aggregate utilization stats across all limiters
            max_rpm = 0.0
            max_tpm = 0.0
            total_requests = 0
            total_tokens = 0

            for limiter in limiters.values():
                try:
                    stats = limiter.get_stats()
                except Exception:
                    # If a particular limiter fails to report, skip it
                    continue

                rpm_util = float(stats.get("rpm_utilization_pct", 0.0))
                tpm_util = float(stats.get("tpm_utilization_pct", 0.0))
                max_rpm = max(max_rpm, rpm_util)
                max_tpm = max(max_tpm, tpm_util)
                total_requests += int(stats.get("total_requests", 0))
                total_tokens += int(stats.get("total_tokens", 0))

            max_util = max(max_rpm, max_tpm)

            if max_util < 70:
                status = HealthStatus.HEALTHY
                message = "Rate limiter has capacity"
            elif max_util < 90:
                status = HealthStatus.DEGRADED
                message = f"Rate limiter under pressure ({max_util:.1f}% utilized)"
            else:
                status = HealthStatus.DEGRADED
                message = f"Rate limiter near capacity ({max_util:.1f}% utilized)"

            return ComponentHealth(
                name="rate_limiter",
                status=status,
                message=message,
                details={
                    "rpm_utilization": round(max_rpm, 1),
                    "tpm_utilization": round(max_tpm, 1),
                    "total_requests": total_requests,
                    "total_tokens": total_tokens,
                },
            )
        except Exception as e:
            return ComponentHealth(
                name="rate_limiter",
                status=HealthStatus.UNHEALTHY,
                message=f"Rate limiter error: {str(e)}",
            )

    def check_cache(self) -> ComponentHealth:
        """
        Check cache status.

        Returns:
            ComponentHealth for cache
        """
        if not self.client.enable_cache or not self.client.cache_manager:
            return ComponentHealth(
                name="cache",
                status=HealthStatus.HEALTHY,
                message="Caching disabled",
            )

        try:
            cache_manager = self.client.cache_manager
            stats = cache_manager.get_stats()

            total_items = sum(stats.values())

            # Cache is healthy if it's accessible
            return ComponentHealth(
                name="cache",
                status=HealthStatus.HEALTHY,
                message=f"Cache operational ({total_items} items)",
                details=stats,
            )

        except Exception as e:
            return ComponentHealth(
                name="cache",
                status=HealthStatus.DEGRADED,
                message=f"Cache error: {str(e)}",
            )

    def check_cost_tracker(self) -> ComponentHealth:
        """
        Check cost tracker status.

        Returns:
            ComponentHealth for cost tracker
        """
        if not self.client.cost_tracker:
            return ComponentHealth(
                name="cost_tracker",
                status=HealthStatus.HEALTHY,
                message="Cost tracking disabled",
            )

        try:
            # Cost tracker is healthy if it's accessible
            return ComponentHealth(
                name="cost_tracker",
                status=HealthStatus.HEALTHY,
                message="Cost tracking operational",
            )

        except Exception as e:
            return ComponentHealth(
                name="cost_tracker",
                status=HealthStatus.DEGRADED,
                message=f"Cost tracker error: {str(e)}",
            )

    async def check_health(self, include_api_check: bool = True) -> HealthCheckResult:
        """
        Perform comprehensive health check.

        Args:
            include_api_check: Whether to include API connectivity check
                             (may be slow, disable for frequent probes)

        Returns:
            HealthCheckResult with overall status and component details
        """
        components: list[ComponentHealth] = []

        # Check API connectivity (optional, can be slow)
        if include_api_check:
            api_health = await self.check_api_connectivity()
            components.append(api_health)

        # Check other components (fast)
        components.extend(
            [
                self.check_rate_limiter(),
                self.check_cache(),
                self.check_cost_tracker(),
            ]
        )

        # Determine overall status
        statuses = [c.status for c in components]

        if HealthStatus.UNHEALTHY in statuses:
            overall_status = HealthStatus.UNHEALTHY
            message = "One or more components are unhealthy"
        elif HealthStatus.DEGRADED in statuses:
            overall_status = HealthStatus.DEGRADED
            message = "One or more components are degraded"
        else:
            overall_status = HealthStatus.HEALTHY
            message = "All components are healthy"

        return HealthCheckResult(
            status=overall_status,
            timestamp=datetime.now(),
            components=components,
            message=message,
            version=self.version,
        )

    async def check_liveness(self) -> bool:
        """
        Liveness probe - checks if the service is alive.

        This is a lightweight check that only verifies the client exists
        and basic components are initialized. Does not make API calls.

        Returns:
            True if alive, False otherwise
        """
        try:
            # Just check if client and basic components exist
            return self.client is not None and self.client.config is not None and self.client.cost_estimator is not None
        except Exception:
            return False

    async def check_readiness(self) -> bool:
        """
        Readiness probe - checks if the service is ready to accept requests.

        This includes checking rate limiter capacity and optionally API connectivity.

        Returns:
            True if ready, False otherwise
        """
        try:
            # Check rate limiter capacity
            if self.client.enable_rate_limiting and self.client.rate_limiter_pool:
                pool = self.client.rate_limiter_pool
                # Access the pool's internal limiter mapping safely; attribute names may differ
                # across implementations and static checkers may not know about them.
                limiters = getattr(pool, "_limiters", None)  # type: ignore[attr-defined]

                if limiters:
                    # Check if any limiter is overloaded
                    for limiter in limiters.values():
                        try:
                            stats = limiter.get_stats()
                        except Exception:
                            # If a limiter fails to report stats, skip it
                            continue

                        rpm_util = stats.get("rpm_utilization_pct", 0)
                        tpm_util = stats.get("tpm_utilization_pct", 0)

                        # Not ready if utilization is too high
                        if max(rpm_util, tpm_util) > 95:
                            return False

            return True

        except Exception:
            return False

    def get_info(self) -> dict[str, Any]:
        """
        Get general information about the client.

        Returns:
            Dictionary with client information
        """
        info: dict[str, Any] = {
            "version": self.version,
            "config": {
                "endpoint": self.client.config.endpoint,
                "chat_deployment": self.client.config.chat_deployment,
                "embedding_deployment": self.client.config.embedding_deployment,
            },
            "features": {
                "rate_limiting": self.client.enable_rate_limiting,
                "caching": self.client.enable_cache,
                "cost_tracking": self.client.cost_tracker is not None,
            },
        }

        return info


__all__ = [
    "HealthStatus",
    "ComponentHealth",
    "HealthCheckResult",
    "HealthChecker",
]
