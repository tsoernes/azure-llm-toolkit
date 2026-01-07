"""Cost tracking and estimation utilities for Azure OpenAI API calls."""

from __future__ import annotations

import logging
from typing import Any, Protocol

from .types import CostInfo, UsageInfo

logger = logging.getLogger(__name__)


class CostEstimator:
    """
    Estimate costs for Azure OpenAI API calls based on model pricing.

    Pricing is configurable per model and supports:
    - Input tokens (non-cached)
    - Output tokens
    - Cached input tokens (prompt caching)

    All prices are per 1 million tokens by default.
    """

    def __init__(self, currency: str = "kr") -> None:
        """
        Initialize cost estimator.

        Args:
            currency: Currency code (e.g., "kr", "usd", "eur")
        """
        self.currency = currency
        self._pricing: dict[str, dict[str, float]] = {}
        self._load_default_pricing()

    def _load_default_pricing(self) -> None:
        """Load default pricing for common Azure OpenAI models (NOK per 1M tokens)."""
        # GPT-4o models
        self._pricing["gpt-4o"] = {
            "input": 41.25,
            "output": 165.00,
            "cached_input": 20.63,
        }
        self._pricing["gpt-4o-mini"] = {
            "input": 1.24,
            "output": 4.95,
            "cached_input": 0.62,
        }

        # GPT-4 Turbo
        self._pricing["gpt-4-turbo"] = {
            "input": 82.50,
            "output": 247.50,
            "cached_input": 41.25,
        }

        # GPT-3.5 Turbo
        self._pricing["gpt-3.5-turbo"] = {
            "input": 4.13,
            "output": 12.38,
            "cached_input": 2.06,
        }

        # o1 models (reasoning models)
        self._pricing["o1-preview"] = {
            "input": 123.75,
            "output": 495.00,
            "cached_input": 61.88,
        }
        self._pricing["o1-mini"] = {
            "input": 24.75,
            "output": 99.00,
            "cached_input": 12.38,
        }

        # GPT-5 models (Global pricing, NOK per 1M tokens)
        self._pricing["gpt-5"] = {
            "input": 12.55,
            "output": 100.37,
            "cached_input": 1.26,
        }
        self._pricing["gpt-5-mini"] = {
            "input": 2.51,
            "output": 20.08,
            "cached_input": 0.26,
        }

        # Embedding models (per 1M tokens)
        # text-embedding-3-small: 0.000226 kr per 1K tokens -> 0.226 kr per 1M
        self._pricing["text-embedding-3-small"] = {
            "input": 0.226,
            "output": 0.0,
            "cached_input": 0.0,
        }
        # text-embedding-3-large: 0.001466 kr per 1K tokens -> 1.466 kr per 1M
        self._pricing["text-embedding-3-large"] = {
            "input": 1.466,
            "output": 0.0,
            "cached_input": 0.0,
        }
        self._pricing["text-embedding-ada-002"] = {
            "input": 0.83,
            "output": 0.0,
            "cached_input": 0.0,
        }

    def set_model_pricing(
        self,
        model: str,
        input_price: float,
        output_price: float,
        cached_input_price: float = 0.0,
    ) -> None:
        """
        Set custom pricing for a model.

        Args:
            model: Model name
            input_price: Price per 1M input tokens
            output_price: Price per 1M output tokens
            cached_input_price: Price per 1M cached input tokens
        """
        self._pricing[model] = {
            "input": input_price,
            "output": output_price,
            "cached_input": cached_input_price,
        }

    def get_model_pricing(self, model: str) -> dict[str, float] | None:
        """
        Get pricing for a model.

        Args:
            model: Model name

        Returns:
            Dict with input, output, and cached_input prices, or None if not found
        """
        # Try exact match first
        if model in self._pricing:
            return self._pricing[model]

        # Try to match by prefix (e.g., "gpt-4o-2024-05-13" matches "gpt-4o")
        for known_model in self._pricing:
            if model.startswith(known_model):
                logger.debug(f"Using pricing for '{known_model}' for model '{model}'")
                return self._pricing[known_model]

        return None

    def estimate_cost(
        self,
        model: str,
        tokens_input: int = 0,
        tokens_output: int = 0,
        tokens_cached_input: int = 0,
    ) -> float:
        """
        Estimate cost for an API call.

        Args:
            model: Model name
            tokens_input: Number of input tokens (non-cached)
            tokens_output: Number of output tokens
            tokens_cached_input: Number of cached input tokens

        Returns:
            Estimated cost in the configured currency
        """
        pricing = self.get_model_pricing(model)
        if pricing is None:
            logger.warning(f"No pricing found for model '{model}', returning 0.0")
            return 0.0

        cost = (
            (tokens_input * pricing["input"])
            + (tokens_output * pricing["output"])
            + (tokens_cached_input * pricing["cached_input"])
        ) / 1_000_000.0

        return float(cost)

    def estimate_cost_from_usage(self, model: str, usage: UsageInfo) -> float:
        """
        Estimate cost from a UsageInfo object.

        Args:
            model: Model name
            usage: Usage information

        Returns:
            Estimated cost in the configured currency
        """
        return self.estimate_cost(
            model=model,
            tokens_input=usage.prompt_tokens,
            tokens_output=usage.completion_tokens,
            tokens_cached_input=usage.cached_prompt_tokens,
        )

    def create_cost_info(
        self,
        model: str,
        tokens_input: int = 0,
        tokens_output: int = 0,
        tokens_cached_input: int = 0,
    ) -> CostInfo:
        """
        Create a CostInfo object with estimated cost.

        Args:
            model: Model name
            tokens_input: Number of input tokens
            tokens_output: Number of output tokens
            tokens_cached_input: Number of cached input tokens

        Returns:
            CostInfo object with estimated cost
        """
        amount = self.estimate_cost(
            model=model,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            tokens_cached_input=tokens_cached_input,
        )

        return CostInfo(
            currency=self.currency,
            amount=amount,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            tokens_cached_input=tokens_cached_input,
            model=model,
        )


class CostTracker(Protocol):
    """
    Protocol for cost tracking implementations.

    This allows users to implement their own cost tracking backend
    (e.g., database, file, in-memory) while maintaining compatibility.
    """

    def record_cost(
        self,
        category: str,
        model: str,
        tokens_input: int,
        tokens_output: int,
        tokens_cached_input: int,
        currency: str,
        amount: float,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Record a cost entry.

        Args:
            category: Cost category (e.g., "embedding", "chat", "indexing")
            model: Model name
            tokens_input: Number of input tokens
            tokens_output: Number of output tokens
            tokens_cached_input: Number of cached input tokens
            currency: Currency code
            amount: Cost amount
            metadata: Optional additional metadata
        """
        ...

    def get_total_cost(self, category: str | None = None) -> float:
        """
        Get total cost, optionally filtered by category.

        Args:
            category: Optional category filter

        Returns:
            Total cost amount
        """
        ...


class InMemoryCostTracker:
    """Simple in-memory cost tracker for testing or temporary tracking.

    This tracker keeps entries in memory and (optionally) appends them to a JSONL file
    so that cost and usage data is persisted across runs.
    """

    def __init__(self, currency: str = "kr", file_path: str | None = None) -> None:
        """
        Initialize in-memory cost tracker.

        Args:
            currency: Default currency code
            file_path: Optional path to a JSONL file for persisting entries.
        """
        self.currency = currency
        self._entries: list[dict[str, Any]] = []
        self._file_path = file_path

    def record_cost(
        self,
        category: str,
        model: str,
        tokens_input: int,
        tokens_output: int,
        tokens_cached_input: int,
        currency: str,
        amount: float,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record a cost entry."""
        import json
        from datetime import datetime, timezone
        from pathlib import Path

        entry = {
            "category": category,
            "model": model,
            "tokens_input": tokens_input,
            "tokens_output": tokens_output,
            "tokens_cached_input": tokens_cached_input,
            "currency": currency,
            "amount": amount,
            "metadata": metadata or {},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._entries.append(entry)

        # Append to JSONL file if configured
        if self._file_path:
            try:
                path = Path(self._file_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(entry) + "\n")
            except Exception as e:
                # Persistence is best-effort; do not fail the main flow
                logger.warning(f"Failed to persist cost entry to file '{self._file_path}': {e}")

    def get_total_cost(self, category: str | None = None) -> float:
        """Get total cost, optionally filtered by category."""
        entries = self._entries
        if category is not None:
            entries = [e for e in entries if e["category"] == category]
        return sum(e["amount"] for e in entries)

    def get_entries(self, category: str | None = None) -> list[dict[str, Any]]:
        """Get all entries, optionally filtered by category."""
        if category is None:
            return list(self._entries)
        return [e for e in self._entries if e["category"] == category]

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of costs by category and model."""
        by_category: dict[str, float] = {}
        by_model: dict[str, float] = {}
        total_tokens_input = 0
        total_tokens_output = 0
        total_tokens_cached = 0

        for entry in self._entries:
            category = entry["category"]
            model = entry["model"]
            amount = entry["amount"]

            by_category[category] = by_category.get(category, 0.0) + amount
            by_model[model] = by_model.get(model, 0.0) + amount

            total_tokens_input += entry["tokens_input"]
            total_tokens_output += entry["tokens_output"]
            total_tokens_cached += entry["tokens_cached_input"]

        return {
            "total_cost": sum(e["amount"] for e in self._entries),
            "currency": self.currency,
            "by_category": by_category,
            "by_model": by_model,
            "total_entries": len(self._entries),
            "total_tokens_input": total_tokens_input,
            "total_tokens_output": total_tokens_output,
            "total_tokens_cached": total_tokens_cached,
        }

    def reset(self) -> None:
        """Clear all recorded entries."""
        self._entries.clear()


__all__ = [
    "CostEstimator",
    "CostTracker",
    "InMemoryCostTracker",
]
