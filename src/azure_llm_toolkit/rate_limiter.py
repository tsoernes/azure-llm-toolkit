# rag-mcp/src/rag_mcp/rate_limiter.py
"""
Rate limiting for Azure OpenAI API calls.

Implements Token Per Minute (TPM) and Request Per Minute (RPM) rate limiting
to avoid hitting Azure OpenAI quotas and getting rate limit errors.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class RateLimiter:
    """
    Rate limiter that tracks both requests per minute (RPM) and tokens per minute (TPM).

    Uses a sliding window approach with token buckets to enforce limits.
    """

    rpm_limit: int = 3000  # Requests per minute
    tpm_limit: int = 300000  # Tokens per minute (300K for GPT-5)

    # Internal state
    _rpm_bucket: float = field(default=0.0, init=False, repr=False)
    _tpm_bucket: float = field(default=0.0, init=False, repr=False)
    _last_refill: float = field(default_factory=time.time, init=False, repr=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)

    # Tracking for metrics
    _total_requests: int = field(default=0, init=False, repr=False)
    _total_tokens: int = field(default=0, init=False, repr=False)
    _total_wait_time: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self):
        """Initialize buckets to full capacity."""
        self._rpm_bucket = float(self.rpm_limit)
        self._tpm_bucket = float(self.tpm_limit)
        self._last_refill = time.time()

    def _refill_buckets(self) -> None:
        """Refill token buckets based on elapsed time."""
        now = time.time()
        elapsed = now - self._last_refill

        if elapsed <= 0:
            return

        # Refill rate: limits per minute -> per second
        rpm_per_sec = self.rpm_limit / 60.0
        tpm_per_sec = self.tpm_limit / 60.0

        # Add tokens proportional to elapsed time
        self._rpm_bucket = min(self.rpm_limit, self._rpm_bucket + (rpm_per_sec * elapsed))
        self._tpm_bucket = min(self.tpm_limit, self._tpm_bucket + (tpm_per_sec * elapsed))

        self._last_refill = now

    def _calculate_wait_time(self, tokens: int) -> float:
        """
        Calculate how long to wait before a request can proceed.

        Args:
            tokens: Number of tokens the request will consume

        Returns:
            Wait time in seconds (0 if no wait needed)
        """
        self._refill_buckets()

        # Check if we need to wait for RPM
        rpm_wait = 0.0
        if self._rpm_bucket < 1:
            # How many requests short are we?
            shortage = 1 - self._rpm_bucket
            # How long until we have enough? (requests per second = rpm_limit / 60)
            rpm_wait = shortage / (self.rpm_limit / 60.0)

        # Check if we need to wait for TPM
        tpm_wait = 0.0
        if self._tpm_bucket < tokens:
            # How many tokens short are we?
            shortage = tokens - self._tpm_bucket
            # How long until we have enough? (tokens per second = tpm_limit / 60)
            tpm_wait = shortage / (self.tpm_limit / 60.0)

        # Wait for whichever is longer
        return max(rpm_wait, tpm_wait)

    async def acquire(self, tokens: int = 1) -> None:
        """
        Acquire permission to make a request consuming the specified tokens.

        Blocks until the request can proceed without exceeding rate limits.

        Args:
            tokens: Number of tokens the request will consume (estimated)
        """
        async with self._lock:
            wait_time = self._calculate_wait_time(tokens)

            if wait_time > 0:
                logger.debug(
                    f"Rate limit: waiting {wait_time:.2f}s "
                    f"(RPM: {self._rpm_bucket:.1f}/{self.rpm_limit}, "
                    f"TPM: {self._tpm_bucket:.0f}/{self.tpm_limit})"
                )
                await asyncio.sleep(wait_time)
                self._total_wait_time += wait_time
                # Refill after waiting
                self._refill_buckets()

            # Consume tokens
            self._rpm_bucket -= 1
            self._tpm_bucket -= tokens

            # Ensure buckets don't go negative (defensive)
            self._rpm_bucket = max(0, self._rpm_bucket)
            self._tpm_bucket = max(0, self._tpm_bucket)

            # Track metrics
            self._total_requests += 1
            self._total_tokens += tokens

    def update_usage(self, actual_tokens: int, estimated_tokens: int) -> None:
        """
        Update token usage after actual consumption is known.

        If we underestimated, deduct additional tokens. If we overestimated,
        refund the difference.

        Args:
            actual_tokens: Actual tokens consumed by the API call
            estimated_tokens: Tokens we estimated and reserved
        """
        difference = actual_tokens - estimated_tokens
        if difference != 0:
            # Adjust the bucket (negative difference = refund, positive = deduct more)
            self._tpm_bucket -= difference
            self._tpm_bucket = max(0, min(self.tpm_limit, self._tpm_bucket))

            if difference > 0:
                logger.debug(f"Token usage adjustment: underestimated by {difference} tokens")

    def get_stats(self) -> dict[str, float]:
        """
        Get rate limiter statistics.

        Returns:
            Dict with total requests, tokens, wait time, and current bucket levels
        """
        self._refill_buckets()
        return {
            "total_requests": self._total_requests,
            "total_tokens": self._total_tokens,
            "total_wait_time_seconds": self._total_wait_time,
            "rpm_available": self._rpm_bucket,
            "rpm_limit": self.rpm_limit,
            "rpm_utilization_pct": ((self.rpm_limit - self._rpm_bucket) / self.rpm_limit * 100)
            if self.rpm_limit > 0
            else 0,
            "tpm_available": self._tpm_bucket,
            "tpm_limit": self.tpm_limit,
            "tpm_utilization_pct": ((self.tpm_limit - self._tpm_bucket) / self.tpm_limit * 100)
            if self.tpm_limit > 0
            else 0,
        }

    def reset(self) -> None:
        """Reset the rate limiter to initial state."""
        self._rpm_bucket = float(self.rpm_limit)
        self._tpm_bucket = float(self.tpm_limit)
        self._last_refill = time.time()
        self._total_requests = 0
        self._total_tokens = 0
        self._total_wait_time = 0.0


class RateLimiterPool:
    """
    Pool of rate limiters for different models/deployments.

    Automatically creates and manages separate rate limiters per model.
    """

    def __init__(self, default_rpm: int = 3000, default_tpm: int = 300000):
        """
        Initialize rate limiter pool.

        Args:
            default_rpm: Default requests per minute for new limiters
            default_tpm: Default tokens per minute for new limiters
        """
        self.default_rpm = default_rpm
        self.default_tpm = default_tpm
        self._limiters: dict[str, RateLimiter] = {}
        self._lock = asyncio.Lock()

    async def get_limiter(self, model: str, rpm: Optional[int] = None, tpm: Optional[int] = None) -> RateLimiter:
        """
        Get or create a rate limiter for a specific model.

        Args:
            model: Model name/deployment ID
            rpm: Optional custom RPM limit for this model
            tpm: Optional custom TPM limit for this model

        Returns:
            RateLimiter instance for the model
        """
        async with self._lock:
            if model not in self._limiters:
                rpm = rpm or self.default_rpm
                tpm = tpm or self.default_tpm
                self._limiters[model] = RateLimiter(rpm_limit=rpm, tpm_limit=tpm)
                logger.info(f"Created rate limiter for '{model}': {rpm} RPM, {tpm} TPM")
            return self._limiters[model]

    def get_all_stats(self) -> dict[str, dict[str, float]]:
        """
        Get statistics for all rate limiters in the pool.

        Returns:
            Dict mapping model name to its statistics
        """
        return {model: limiter.get_stats() for model, limiter in self._limiters.items()}

    def reset_all(self) -> None:
        """Reset all rate limiters in the pool."""
        for limiter in self._limiters.values():
            limiter.reset()


# Default rate limiter pool singleton
_default_pool: Optional[RateLimiterPool] = None


def get_rate_limiter_pool() -> RateLimiterPool:
    """
    Get the default rate limiter pool singleton.

    Returns:
        Global RateLimiterPool instance
    """
    global _default_pool
    if _default_pool is None:
        _default_pool = RateLimiterPool()
    return _default_pool
