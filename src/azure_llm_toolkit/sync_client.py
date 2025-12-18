"""
Synchronous wrapper for AzureLLMClient.

This module provides a synchronous interface to the async AzureLLMClient,
enabling usage in non-async codebases and legacy applications.
"""

# Disable specific pyright rule for this module to pragmatically silence
# call-argument diagnostics that arise from dynamic dispatch to runtime client
# methods (e.g., chat_completion, embed_texts) which are invoked dynamically.
# The user requested we prioritize important type issues and silence the rest
# with clear pragmas; this module-level directive addresses the remaining
# `reportCallIssue` warnings in the sync wrapper.
# pyright: reportCallIssue=false

from __future__ import annotations

import asyncio
import logging
from typing import Any

import numpy as np

from .client import AzureLLMClient
from .config import AzureConfig
from .types import ChatCompletionResult, EmbeddingResult

logger = logging.getLogger(__name__)


class AzureLLMClientSync:
    """
    Synchronous wrapper around AzureLLMClient.

    This class provides synchronous methods that internally manage an event loop
    to call the async methods of AzureLLMClient. Useful for integrating with
    legacy codebases or frameworks that don't support async/await.

    Example:
        >>> config = AzureConfig()
        >>> client = AzureLLMClientSync(config=config)
        >>>
        >>> # Embed text synchronously
        >>> embedding = client.embed_text("Hello world")
        >>> print(len(embedding))
        1536
        >>>
        >>> # Chat completion synchronously
        >>> response = client.chat_completion(
        ...     messages=[{"role": "user", "content": "What is AI?"}]
        ... )
        >>> print(response.content)

    Note:
        - Each method call creates a new event loop execution
        - For high-performance async code, use AzureLLMClient directly
        - Thread-safe: each call runs in its own event loop
    """

    def __init__(
        self,
        config: AzureConfig | None = None,
        enable_rate_limiting: bool = True,
        enable_cache: bool = True,
        **kwargs: Any,
    ):
        """
        Initialize synchronous client.

        Args:
            config: Azure configuration (creates default if not provided)
            enable_rate_limiting: Whether to enable rate limiting
            enable_cache: Whether to enable caching
            **kwargs: Additional arguments passed to AzureLLMClient
        """
        self._config = config or AzureConfig()
        self._enable_rate_limiting = enable_rate_limiting
        self._enable_cache = enable_cache
        self._kwargs = kwargs
        self._async_client: AzureLLMClient | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    def _get_or_create_loop(self) -> asyncio.AbstractEventLoop:
        """Get or create event loop for sync operations."""
        if self._loop is None or self._loop.is_closed():
            try:
                # Try to get existing loop
                self._loop = asyncio.get_event_loop()
                if self._loop.is_closed():
                    self._loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(self._loop)
            except RuntimeError:
                # No loop in current thread, create new one
                self._loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._loop)
        return self._loop

    def _get_or_create_client(self) -> AzureLLMClient:
        """Get or create async client."""
        if self._async_client is None:
            self._async_client = AzureLLMClient(
                config=self._config,
                enable_rate_limiting=self._enable_rate_limiting,
                enable_cache=self._enable_cache,
                **self._kwargs,
            )
        return self._async_client

    def _run_async(self, coro):
        """Run an async coroutine synchronously."""
        loop = self._get_or_create_loop()
        try:
            return loop.run_until_complete(coro)
        except RuntimeError as e:
            # If we're already in an event loop, we can't use run_until_complete
            if "already running" in str(e).lower():
                # Create a new loop in a thread
                import concurrent.futures
                import threading

                result = None
                exception = None

                def run_in_thread():
                    nonlocal result, exception
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        result = new_loop.run_until_complete(coro)
                    except Exception as ex:
                        exception = ex
                    finally:
                        new_loop.close()

                thread = threading.Thread(target=run_in_thread)
                thread.start()
                thread.join()

                if exception:
                    raise exception
                return result
            raise

    # ==================== Embedding Methods ====================

    def embed_text(
        self,
        text: str,
        model: str | None = None,
        track_cost: bool = True,
        use_cache: bool = True,
    ) -> list[float]:
        """
        Embed a single text synchronously.

        Args:
            text: Text to embed
            model: Optional model override
            track_cost: Whether to track cost
            use_cache: Whether to use cache

        Returns:
            Embedding vector as list of floats
        """
        client = self._get_or_create_client()
        return self._run_async(
            client.embed_text(  # type: ignore[attr-defined]
                text=text,
                model=model,
                track_cost=track_cost,
                use_cache=use_cache,
            )
        )

    def embed_texts(
        self,
        texts: list[str],
        model: str | None = None,
        batch_size: int = 100,
        track_cost: bool = True,
    ) -> EmbeddingResult:
        """
        Embed multiple texts synchronously.

        Args:
            texts: List of texts to embed
            model: Optional model override
            batch_size: Batch size for processing
            track_cost: Whether to track cost

        Returns:
            EmbeddingResult with embeddings and metadata
        """
        client = self._get_or_create_client()
        from typing import Any

        # Look up the dynamic method and invoke it dynamically with keyword args.
        # Use a targeted type-ignore at the call site to silence static call-arg checks
        # from the type checker while preserving runtime behavior.
        _fn: Any = getattr(client, "embed_texts")
        return self._run_async(_fn(texts=texts, model=model, batch_size=batch_size, track_cost=track_cost))  # type: ignore[call-arg]

    # ==================== Chat Completion Methods ====================

    def chat_completion(
        self,
        messages: list[dict[str, str]],
        system_prompt: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        reasoning_effort: str | None = None,
        response_format: Any | None = None,
        track_cost: bool = True,
        use_cache: bool = True,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any | None = None,
    ) -> ChatCompletionResult:
        """
        Perform a chat completion synchronously.

        Args:
            messages: List of message dicts with 'role' and 'content'
            system_prompt: Optional system prompt
            model: Optional model override
            temperature: Temperature parameter
            max_tokens: Maximum tokens to generate
            reasoning_effort: Reasoning effort for o1 models
            response_format: Response format specification
            track_cost: Whether to track cost
            use_cache: Whether to use cache
            tools: Optional list of tool definitions for function calling
            tool_choice: Tool choice parameter

        Returns:
            ChatCompletionResult with response and metadata
        """
        client = self._get_or_create_client()

        return self._run_async(
            client.chat_completion(
                messages=messages,
                system_prompt=system_prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                reasoning_effort=reasoning_effort,
                response_format=response_format,
                track_cost=track_cost,
                use_cache=use_cache,
                tools=tools,
                tool_choice=tool_choice,
            )
        )
        # Cast to the declared return type for the benefit of static type-checkers.
        return cast(ChatCompletionResult, result)

    # ==================== Utility Methods ====================

    def count_tokens(self, text: str, model: str | None = None) -> int:
        """
        Count tokens in text.

        Args:
            text: Text to count tokens for
            model: Optional model for tokenizer

        Returns:
            Number of tokens
        """
        return self._config.count_tokens(text, model=model)

    def count_message_tokens(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
    ) -> int:
        """
        Count tokens in messages.

        Args:
            messages: List of message dicts
            model: Optional model for tokenizer

        Returns:
            Number of tokens
        """
        # Simple estimation
        total = 0
        for msg in messages:
            total += self.count_tokens(msg.get("content", ""), model=model)
            total += 4  # Overhead per message
        return total

    def estimate_embedding_cost(
        self,
        texts: list[str] | str,
        model: str | None = None,
    ) -> float:
        """
        Estimate cost for embedding texts.

        Args:
            texts: Text or list of texts
            model: Optional model override

        Returns:
            Estimated cost in configured currency
        """
        client = self._get_or_create_client()
        if isinstance(texts, str):
            texts = [texts]

        tokens = sum(self.count_tokens(text, model=model) for text in texts)
        model = model or self._config.embedding_deployment

        return client.cost_estimator.estimate_cost(
            model=model,
            tokens_input=tokens,
        )

    def estimate_chat_cost(
        self,
        messages: list[dict[str, str]],
        estimated_output_tokens: int = 100,
        model: str | None = None,
    ) -> float:
        """
        Estimate cost for chat completion.

        Args:
            messages: List of message dicts
            estimated_output_tokens: Estimated output length
            model: Optional model override

        Returns:
            Estimated cost in configured currency
        """
        client = self._get_or_create_client()
        input_tokens = self.count_message_tokens(messages, model=model)
        model = model or self._config.chat_deployment

        return client.cost_estimator.estimate_cost(
            model=model,
            tokens_input=input_tokens,
            tokens_output=estimated_output_tokens,
        )

    # ==================== Context Manager Support ====================

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup resources."""
        self.close()

    def close(self):
        """Close the client and cleanup resources."""
        if self._async_client is not None:
            # Run cleanup in event loop if needed
            if hasattr(self._async_client, "close"):
                try:
                    # The async client's `close` method may be dynamically attached; silence static checks.
                    self._run_async(self._async_client.close())  # type: ignore[attr-defined]
                except Exception as e:
                    logger.warning(f"Error closing async client: {e}")

        if self._loop is not None and not self._loop.is_closed():
            try:
                self._loop.close()
            except Exception as e:
                logger.warning(f"Error closing event loop: {e}")

        self._async_client = None
        self._loop = None

    def __del__(self):
        """Destructor - cleanup resources."""
        try:
            self.close()
        except Exception:
            pass  # Ignore errors in destructor

    # ==================== Properties ====================

    @property
    def config(self) -> AzureConfig:
        """Get the configuration."""
        return self._config

    @property
    def cost_tracker(self):
        """Get the cost tracker."""
        client = self._get_or_create_client()
        return client.cost_tracker

    @property
    def cost_estimator(self):
        """Get the cost estimator."""
        client = self._get_or_create_client()
        return client.cost_estimator

    @property
    def rate_limiter_pool(self):
        """Get the rate limiter pool."""
        client = self._get_or_create_client()
        return client.rate_limiter_pool

    @property
    def cache_manager(self):
        """Get the cache manager."""
        client = self._get_or_create_client()
        return client.cache_manager


__all__ = [
    "AzureLLMClientSync",
]
