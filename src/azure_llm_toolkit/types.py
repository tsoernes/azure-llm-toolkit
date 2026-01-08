"""Shared types and data structures for azure-llm-toolkit."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class UsageInfo:
    """Token usage information from API calls."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_prompt_tokens: int = 0
    reasoning_tokens: int = 0

    @classmethod
    def from_openai_usage(cls, usage: Any) -> UsageInfo:
        """Create UsageInfo from OpenAI API response usage object."""
        if usage is None:
            return cls()

        # Handle prompt_tokens_details which may be a structured object
        cached = 0
        prompt_details = getattr(usage, "prompt_tokens_details", None)
        if prompt_details is not None:
            # Newer OpenAI clients return a PromptTokensDetails object
            cached = getattr(prompt_details, "cached_tokens", 0) or 0

        # Handle completion_tokens_details for reasoning tokens
        reasoning = 0
        completion_details = getattr(usage, "completion_tokens_details", None)
        if completion_details is not None:
            # Extract reasoning_tokens from CompletionTokensDetails
            reasoning = getattr(completion_details, "reasoning_tokens", 0) or 0

        return cls(
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            total_tokens=getattr(usage, "total_tokens", 0) or 0,
            cached_prompt_tokens=cached,
            reasoning_tokens=reasoning,
        )

    def to_dict(self) -> dict[str, int]:
        """Convert to dictionary."""
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cached_prompt_tokens": self.cached_prompt_tokens,
            "reasoning_tokens": self.reasoning_tokens,
        }


@dataclass
class CostInfo:
    """Cost information for API calls."""

    currency: str
    amount: float
    tokens_input: int
    tokens_output: int
    tokens_cached_input: int
    model: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "currency": self.currency,
            "amount": self.amount,
            "tokens_input": self.tokens_input,
            "tokens_output": self.tokens_output,
            "tokens_cached_input": self.tokens_cached_input,
            "model": self.model,
        }


@dataclass
class ChatCompletionResult:
    """Result from a chat completion call with usage tracking."""

    content: str
    usage: UsageInfo
    model: str
    finish_reason: str | None = None
    raw_response: Any | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "content": self.content,
            "usage": self.usage.to_dict(),
            "model": self.model,
            "finish_reason": self.finish_reason,
        }


@dataclass
class EmbeddingResult:
    """Result from an embedding call."""

    embeddings: list[list[float]]
    model: str
    usage: UsageInfo

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "embeddings": self.embeddings,
            "model": self.model,
            "usage": self.usage.to_dict(),
        }


__all__ = [
    "UsageInfo",
    "CostInfo",
    "ChatCompletionResult",
    "EmbeddingResult",
]
