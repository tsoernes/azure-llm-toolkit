"""Circuit breaker pattern for Azure OpenAI API resilience.

This module implements the circuit breaker pattern to prevent cascading failures
when the API is unavailable or experiencing issues.

Circuit breaker states:
- CLOSED: Normal operation, requests pass through
- OPEN: API is failing, requests fail fast without calling API
- HALF_OPEN: Testing if API has recovered, limited requests allowed
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery


class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open."""

    def __init__(self, message: str, state: CircuitState, failure_count: int):
        super().__init__(message)
        self.state = state
        self.failure_count = failure_count


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""

    failure_threshold: int = 5  # Failures before opening
    success_threshold: int = 2  # Successes in half-open before closing
    timeout_seconds: float = 60.0  # Time to wait before half-open
    half_open_max_requests: int = 3  # Max requests in half-open state


@dataclass
class CircuitBreakerMetrics:
    """Circuit breaker metrics."""

    state: CircuitState
    failure_count: int
    success_count: int
    total_requests: int
    rejected_requests: int
    last_failure_time: float | None
    last_state_change: float
    state_durations: dict[str, float] = field(default_factory=dict)


class CircuitBreaker:
    """
    Circuit breaker for protecting against cascading failures.

    Automatically opens when API failures exceed threshold, preventing
    unnecessary API calls. Automatically attempts recovery after timeout.

    Example:
        >>> breaker = CircuitBreaker()
        >>>
        >>> async def call_api():
        >>>     async with breaker.protect():
        >>>         return await client.embed_text("test")
        >>>
        >>> try:
        >>>     result = await call_api()
        >>> except CircuitBreakerError:
        >>>     # Circuit is open, API is down
        >>>     return fallback_response()
    """

    def __init__(self, config: CircuitBreakerConfig | None = None):
        """
        Initialize circuit breaker.

        Args:
            config: Circuit breaker configuration
        """
        self.config = config or CircuitBreakerConfig()
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._total_requests = 0
        self._rejected_requests = 0
        self._last_failure_time: float | None = None
        self._last_state_change = time.time()
        self._half_open_requests = 0
        self._lock = asyncio.Lock()
        self._state_durations: dict[str, float] = {
            "closed": 0.0,
            "open": 0.0,
            "half_open": 0.0,
        }

    @property
    def state(self) -> CircuitState:
        """Get current state."""
        return self._state

    @property
    def is_closed(self) -> bool:
        """Check if circuit is closed (normal operation)."""
        return self._state == CircuitState.CLOSED

    @property
    def is_open(self) -> bool:
        """Check if circuit is open (rejecting requests)."""
        return self._state == CircuitState.OPEN

    @property
    def is_half_open(self) -> bool:
        """Check if circuit is half-open (testing recovery)."""
        return self._state == CircuitState.HALF_OPEN

    def _update_state_duration(self) -> None:
        """Update duration tracking for current state."""
        now = time.time()
        elapsed = now - self._last_state_change
        self._state_durations[self._state.value] += elapsed
        self._last_state_change = now

    def _transition_to(self, new_state: CircuitState) -> None:
        """
        Transition to a new state.

        Args:
            new_state: State to transition to
        """
        if new_state == self._state:
            return

        old_state = self._state
        self._update_state_duration()
        self._state = new_state

        logger.info(f"Circuit breaker state change: {old_state.value} -> {new_state.value}")

        if new_state == CircuitState.OPEN:
            self._success_count = 0
            self._half_open_requests = 0
        elif new_state == CircuitState.HALF_OPEN:
            self._half_open_requests = 0
            self._success_count = 0
        elif new_state == CircuitState.CLOSED:
            self._failure_count = 0
            self._success_count = 0
            self._half_open_requests = 0

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt recovery."""
        if self._last_failure_time is None:
            return False
        return time.time() - self._last_failure_time >= self.config.timeout_seconds

    async def _check_state(self) -> None:
        """Check and update circuit state if needed."""
        async with self._lock:
            if self._state == CircuitState.OPEN and self._should_attempt_reset():
                self._transition_to(CircuitState.HALF_OPEN)
                logger.info("Circuit breaker attempting recovery (half-open)")

    async def _on_success(self) -> None:
        """Handle successful request."""
        async with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.config.success_threshold:
                    self._transition_to(CircuitState.CLOSED)
                    logger.info("Circuit breaker recovered (closed)")
            elif self._state == CircuitState.CLOSED:
                # Reset failure count on success
                if self._failure_count > 0:
                    self._failure_count = 0

    async def _on_failure(self, exception: Exception) -> None:
        """Handle failed request."""
        async with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()

            logger.warning(
                f"Circuit breaker failure ({self._failure_count}/{self.config.failure_threshold}): {exception}"
            )

            if self._state == CircuitState.HALF_OPEN:
                # Any failure in half-open immediately reopens
                self._transition_to(CircuitState.OPEN)
                logger.warning("Circuit breaker reopened due to failure in half-open state")
            elif self._state == CircuitState.CLOSED:
                if self._failure_count >= self.config.failure_threshold:
                    self._transition_to(CircuitState.OPEN)
                    logger.error(f"Circuit breaker opened after {self._failure_count} consecutive failures")

    async def call(self, func: Callable, *args: Any, **kwargs: Any) -> Any:
        """
        Call a function with circuit breaker protection.

        Args:
            func: Async function to call
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result

        Raises:
            CircuitBreakerError: If circuit is open
            Exception: Original exception if call fails
        """
        await self._check_state()

        # Check if we should allow the request
        async with self._lock:
            self._total_requests += 1

            if self._state == CircuitState.OPEN:
                self._rejected_requests += 1
                raise CircuitBreakerError(
                    f"Circuit breaker is OPEN. API is experiencing failures. "
                    f"Will retry in {self.config.timeout_seconds}s.",
                    state=self._state,
                    failure_count=self._failure_count,
                )

            if self._state == CircuitState.HALF_OPEN:
                if self._half_open_requests >= self.config.half_open_max_requests:
                    self._rejected_requests += 1
                    raise CircuitBreakerError(
                        "Circuit breaker is HALF_OPEN and at max test requests. Waiting for recovery test to complete.",
                        state=self._state,
                        failure_count=self._failure_count,
                    )
                self._half_open_requests += 1

        # Execute the function
        try:
            result = await func(*args, **kwargs)
            await self._on_success()
            return result
        except Exception as e:
            await self._on_failure(e)
            raise

    async def __aenter__(self):
        """Context manager entry."""
        await self._check_state()

        async with self._lock:
            self._total_requests += 1

            if self._state == CircuitState.OPEN:
                self._rejected_requests += 1
                raise CircuitBreakerError(
                    f"Circuit breaker is OPEN. API is experiencing failures. "
                    f"Will retry in {self.config.timeout_seconds}s.",
                    state=self._state,
                    failure_count=self._failure_count,
                )

            if self._state == CircuitState.HALF_OPEN:
                if self._half_open_requests >= self.config.half_open_max_requests:
                    self._rejected_requests += 1
                    raise CircuitBreakerError(
                        "Circuit breaker is HALF_OPEN and at max test requests.",
                        state=self._state,
                        failure_count=self._failure_count,
                    )
                self._half_open_requests += 1

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if exc_type is None:
            # Success
            await self._on_success()
        else:
            # Failure
            await self._on_failure(exc_val)
        return False  # Don't suppress exceptions

    def protect(self):
        """
        Get context manager for protecting a code block.

        Example:
            >>> async with breaker.protect():
            >>>     result = await api_call()
        """
        return self

    def get_metrics(self) -> CircuitBreakerMetrics:
        """
        Get current circuit breaker metrics.

        Returns:
            CircuitBreakerMetrics with current state and statistics
        """
        self._update_state_duration()

        return CircuitBreakerMetrics(
            state=self._state,
            failure_count=self._failure_count,
            success_count=self._success_count,
            total_requests=self._total_requests,
            rejected_requests=self._rejected_requests,
            last_failure_time=self._last_failure_time,
            last_state_change=self._last_state_change,
            state_durations=dict(self._state_durations),
        )

    def reset(self) -> None:
        """Reset circuit breaker to closed state."""
        logger.info("Circuit breaker manually reset")
        self._transition_to(CircuitState.CLOSED)
        self._failure_count = 0
        self._success_count = 0
        self._total_requests = 0
        self._rejected_requests = 0
        self._last_failure_time = None
        self._half_open_requests = 0


class MultiCircuitBreaker:
    """
    Manage multiple circuit breakers for different resources.

    Example:
        >>> breakers = MultiCircuitBreaker()
        >>>
        >>> # Different breakers for different models
        >>> async with breakers.get("gpt-4o").protect():
        >>>     await client.chat_completion(model="gpt-4o", ...)
        >>>
        >>> async with breakers.get("embeddings").protect():
        >>>     await client.embed_text(...)
    """

    def __init__(self, default_config: CircuitBreakerConfig | None = None):
        """
        Initialize multi-circuit breaker.

        Args:
            default_config: Default configuration for new breakers
        """
        self.default_config = default_config or CircuitBreakerConfig()
        self._breakers: dict[str, CircuitBreaker] = {}
        self._lock = asyncio.Lock()

    async def get(self, name: str, config: CircuitBreakerConfig | None = None) -> CircuitBreaker:
        """
        Get or create a circuit breaker by name.

        Args:
            name: Circuit breaker name (e.g., model name, endpoint)
            config: Optional custom configuration

        Returns:
            CircuitBreaker instance
        """
        async with self._lock:
            if name not in self._breakers:
                breaker_config = config or self.default_config
                self._breakers[name] = CircuitBreaker(breaker_config)
                logger.debug(f"Created circuit breaker for '{name}'")
            return self._breakers[name]

    def get_all_metrics(self) -> dict[str, CircuitBreakerMetrics]:
        """
        Get metrics for all circuit breakers.

        Returns:
            Dict mapping breaker name to its metrics
        """
        return {name: breaker.get_metrics() for name, breaker in self._breakers.items()}

    def reset_all(self) -> None:
        """Reset all circuit breakers."""
        for breaker in self._breakers.values():
            breaker.reset()


__all__ = [
    "CircuitBreaker",
    "MultiCircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitBreakerMetrics",
    "CircuitState",
    "CircuitBreakerError",
]
