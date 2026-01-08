"""Azure OpenAI client wrapper with rate limiting, retry logic, and cost tracking."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from typing import Any

import numpy as np
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncAzureOpenAI,
    BadRequestError,
    RateLimitError,
)
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .cache import CacheManager, ChatCache, EmbeddingCache
from .config import AzureConfig
from .cost_tracker import CostEstimator, CostTracker
from .metrics import MetricsCollector, MetricsTracker
from .rate_limiter import RateLimiter, RateLimiterPool
from .types import ChatCompletionResult, EmbeddingResult, UsageInfo

logger = logging.getLogger(__name__)


def _hash_payload(data: Any) -> str:
    """Generate short hash of payload for retry logging."""
    payload_str = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(payload_str.encode()).hexdigest()[:8]


def _log_retry_attempt(retry_state):
    """Custom callback to log retry attempts with payload hash and timeout info."""
    exception = retry_state.outcome.exception()
    attempt = retry_state.attempt_number
    wait_time = retry_state.next_action.sleep if retry_state.next_action else 0

    # Try to extract payload and client config from args/kwargs
    args = retry_state.args
    kwargs = retry_state.kwargs
    payload_hash = "unknown"
    timeout_info = ""

    # Extract timeout from self (first arg should be the AzureLLMClient instance)
    if args and len(args) > 0 and hasattr(args[0], "config"):
        timeout_seconds = args[0].config.timeout_seconds
        if timeout_seconds is None:
            timeout_info = ", api_timeout=infinite"
        else:
            timeout_info = f", api_timeout={timeout_seconds}s"

    # For embeddings: first arg is batch
    if args and len(args) > 1 and isinstance(args[1], (list, tuple)):
        payload_hash = _hash_payload({"batch_size": len(args[1]), "first_item": args[1][0] if args[1] else None})
    # For chat: messages in kwargs
    elif "messages" in kwargs:
        payload_hash = _hash_payload({"messages": kwargs["messages"]})

    logger.warning(
        f"Retry attempt {attempt} after {exception.__class__.__name__}: {exception} "
        f"(payload_hash={payload_hash}, retry_backoff_delay={wait_time:.1f}s{timeout_info})"
    )


class AzureLLMClient:
    """
    Azure OpenAI client with advanced features:

    - Automatic retry logic with exponential backoff
    - Rate limiting (TPM/RPM)
    - Cost tracking and estimation
    - Batch embedding support
    - Chat completions with reasoning support
    - Token counting
    - Disk-based caching for embeddings and chat completions
    - Optional metrics tracking integration
    - Hooks for streaming and function calling
    """

    def __init__(
        self,
        config: AzureConfig | None = None,
        client: AsyncAzureOpenAI | None = None,
        cost_estimator: CostEstimator | None = None,
        cost_tracker: CostTracker | None = None,
        rate_limiter_pool: RateLimiterPool | None = None,
        enable_rate_limiting: bool = True,
        cache_manager: CacheManager | None = None,
        enable_cache: bool = True,
        metrics_collector: MetricsCollector | None = None,
    ) -> None:
        """
        Initialize Azure LLM client.

        Args:
            config: Azure configuration (will create default if not provided)
            client: Pre-configured AsyncAzureOpenAI client (will create from config if not provided)
            cost_estimator: Cost estimator instance (will create default if not provided)
            cost_tracker: Optional cost tracker for recording costs
            rate_limiter_pool: Rate limiter pool (will create default if not provided)
            enable_rate_limiting: Whether to enable rate limiting
            cache_manager: Cache manager instance (will create default if not provided)
            enable_cache: Whether to enable disk-based caching
        """
        self.config = config or AzureConfig()
        self.client = client or self.config.create_client()
        self.cost_estimator = cost_estimator or CostEstimator(currency="kr")
        self.cost_tracker = cost_tracker
        self.enable_rate_limiting = enable_rate_limiting
        self.enable_cache = enable_cache
        self.metrics_collector = metrics_collector

        # Log timeout configuration
        timeout_str = "infinite" if self.config.timeout_seconds is None else f"{self.config.timeout_seconds}s"
        logger.debug(
            f"Initialized AzureLLMClient (api_timeout={timeout_str}, "
            f"max_retries={self.config.max_retries}, "
            f"rate_limiting={enable_rate_limiting}, "
            f"caching={enable_cache})"
        )

        if enable_rate_limiting:
            self.rate_limiter_pool = rate_limiter_pool or RateLimiterPool()
        else:
            self.rate_limiter_pool = None

        if enable_cache:
            self.cache_manager = cache_manager or CacheManager()
        else:
            self.cache_manager = None

    # ==================== Embeddings ====================
    # Note: For batch embedding, use PolarsBatchEmbedder instead

    @retry(
        reraise=True,
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((APIConnectionError, RateLimitError, APITimeoutError, APIStatusError)),
        before_sleep=_log_retry_attempt,
    )
    async def embed_text(
        self,
        text: str,
        model: str | None = None,
        track_cost: bool = True,
        use_cache: bool = True,
    ) -> list[float]:
        """
        Embed a single text.

        For batch embedding of many texts, use PolarsBatchEmbedder instead,
        which provides intelligent batching, rate limiting, and weighted averaging.

        Args:
            text: Text to embed
            model: Optional model override
            track_cost: Whether to track cost (requires cost_tracker)
            use_cache: Whether to use cache for embedding

        Returns:
            Embedding vector
        """
        model = model or self.config.embedding_deployment
        logger.debug(
            "embed_text called",
            extra={
                "model": model,
                "text_length": len(text),
                "enable_rate_limiting": self.enable_rate_limiting,
                "has_rate_limiter_pool": bool(self.rate_limiter_pool),
                "enable_cache": self.enable_cache,
            },
        )

        # Use metrics tracker as context manager if available
        if self.metrics_collector is not None:
            async with MetricsTracker(self.metrics_collector).track("embed_text", model) as tracker:
                return await self._embed_text_impl(text, model, track_cost, use_cache, tracker)
        else:
            return await self._embed_text_impl(text, model, track_cost, use_cache, None)

    async def _embed_text_impl(
        self,
        text: str,
        model: str,
        track_cost: bool,
        use_cache: bool,
        metrics_tracker: MetricsTracker | None,
    ) -> list[float]:
        """Internal implementation of embed_text."""
        # Check cache if enabled
        if use_cache and self.enable_cache and self.cache_manager:
            cached_embedding = self.cache_manager.embedding_cache.get(text, model)
            if cached_embedding is not None:
                logger.debug("Cache hit for single text embedding")
                embedding_list = cached_embedding.tolist()
                if metrics_tracker is not None:
                    metrics_tracker.set_tokens(input=self.config.count_tokens(text), cached=len(embedding_list))
                    metrics_tracker.set_cost(0.0)
                return embedding_list

        # Rate limiting
        if self.enable_rate_limiting and self.rate_limiter_pool:
            tokens = self.config.count_tokens(text)
            logger.debug(
                "embed_text acquiring rate limiter",
                extra={
                    "model": model,
                    "estimated_tokens": tokens,
                },
            )
            limiter = await self.rate_limiter_pool.get_limiter(model)
            await limiter.acquire(tokens=tokens)

        try:
            start_time = time.time()
            response = await self.client.embeddings.create(model=model, input=[text])
            elapsed = time.time() - start_time

            logger.debug(
                "embed_text received response",
                extra={
                    "model": model,
                    "usage": getattr(response, "usage", None),
                    "elapsed_seconds": f"{elapsed:.2f}",
                },
            )

            # Warn if request took unusually long
            if elapsed > 10.0:
                logger.info(f"Embedding request took {elapsed:.2f}s (model={model})")
        except APITimeoutError as e:
            timeout_str = "infinite" if self.config.timeout_seconds is None else f"{self.config.timeout_seconds}s"
            logger.error(
                f"API timeout on embedding request, configured timeout={timeout_str}, model={model}. "
                f"Consider increasing AZURE_TIMEOUT_SECONDS if needed. Error: {e}"
            )
            raise

        embedding = response.data[0].embedding

        # Update rate limiter with actual usage if available
        if self.enable_rate_limiting and self.rate_limiter_pool and response.usage:
            actual_tokens = response.usage.total_tokens
            estimated_tokens = self.config.count_tokens(text)
            # limiter may be a dynamically-created local variable depending on runtime path;
            # guard access and ignore static attribute/access checks here for the dynamic client.
            try:
                limiter.update_usage(actual_tokens, estimated_tokens)  # type: ignore[attr-defined]
            except Exception:
                # Best-effort: if limiter is not defined or update fails, continue without failing.
                pass

        # Cache the result
        if use_cache and self.enable_cache and self.cache_manager:
            self.cache_manager.embedding_cache.set(text, model, np.array(embedding))

        # Track cost if enabled
        tokens = response.usage.total_tokens if response.usage else self.config.count_tokens(text)
        cost = 0.0
        if track_cost and self.cost_tracker:
            cost = self.cost_estimator.estimate_cost(model=model, tokens_input=tokens)
            self.cost_tracker.record_cost(
                category="embedding",
                model=model,
                tokens_input=tokens,
                tokens_output=0,
                tokens_cached_input=0,
                currency=self.cost_estimator.currency,
                amount=cost,
                metadata={"operation": "embed_text"},
            )

        if metrics_tracker is not None:
            metrics_tracker.set_tokens(input=tokens, output=0, cached=0)
            metrics_tracker.set_cost(cost)

        return embedding

    # ==================== Chat Completions ====================

    @retry(
        reraise=True,
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((APIConnectionError, RateLimitError, APITimeoutError, APIStatusError)),
        before_sleep=_log_retry_attempt,
    )
    async def chat_completion(
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
        Perform a chat completion with usage tracking.

        Args:
            messages: List of message dicts with 'role' and 'content'
            system_prompt: Optional system prompt (prepended to messages)
            model: Optional model override
            temperature: Temperature parameter (not supported by all models)
            max_tokens: Maximum tokens to generate
            reasoning_effort: Reasoning effort for o1/GPT-5 models ("low", "medium", "high")
            response_format: Response format (e.g., {"type": "json_object"})
            track_cost: Whether to track cost
            use_cache: Whether to use cache for chat completions

        Returns:
            ChatCompletionResult with content and usage info
        """
        model = model or self.config.chat_deployment

        # Use metrics tracker as context manager if available
        if self.metrics_collector is not None:
            async with MetricsTracker(self.metrics_collector).track("chat_completion", model) as tracker:
                return await self._chat_completion_impl(
                    messages,
                    system_prompt,
                    model,
                    temperature,
                    max_tokens,
                    reasoning_effort,
                    response_format,
                    track_cost,
                    use_cache,
                    tools,
                    tool_choice,
                    tracker,
                )
        else:
            return await self._chat_completion_impl(
                messages,
                system_prompt,
                model,
                temperature,
                max_tokens,
                reasoning_effort,
                response_format,
                track_cost,
                use_cache,
                tools,
                tool_choice,
                None,
            )

    async def _chat_completion_impl(
        self,
        messages: list[dict[str, str]],
        system_prompt: str | None,
        model: str,
        temperature: float | None,
        max_tokens: int | None,
        reasoning_effort: str | None,
        response_format: Any | None,
        track_cost: bool,
        use_cache: bool,
        tools: list[dict[str, Any]] | None,
        tool_choice: Any | None,
        metrics_tracker: MetricsTracker | None,
    ) -> ChatCompletionResult:
        """Internal implementation of chat_completion."""
        # Build messages
        full_messages: list[dict[str, str]] = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        full_messages.extend(messages)

        # Check cache if enabled
        if use_cache and self.enable_cache and self.cache_manager:
            cached_response = self.cache_manager.chat_cache.get(
                messages=full_messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            if cached_response:
                logger.debug("Cache hit for chat completion")
                return ChatCompletionResult(
                    content=cached_response["content"],
                    usage=UsageInfo(**cached_response["usage"]),
                    model=cached_response["model"],
                    finish_reason=cached_response.get("finish_reason"),
                    raw_response=None,
                )

        # Estimate tokens for rate limiting
        if self.enable_rate_limiting and self.rate_limiter_pool:
            estimated_input_tokens = sum(self.config.count_tokens(m.get("content", "")) for m in full_messages)
            estimated_output_tokens = max_tokens or 1000  # Conservative estimate
            estimated_total = estimated_input_tokens + estimated_output_tokens

            limiter = await self.rate_limiter_pool.get_limiter(model)
            await limiter.acquire(tokens=estimated_total)

        # Build API kwargs
        kwargs: dict[str, Any] = {"model": model, "messages": full_messages}

        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if reasoning_effort is not None:
            kwargs["reasoning_effort"] = reasoning_effort
        if response_format is not None:
            kwargs["response_format"] = response_format
        if tools is not None:
            kwargs["tools"] = tools
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice

        # Make API call with retry on specific errors
        max_attempts = 3
        last_error: Exception | None = None

        for attempt in range(1, max_attempts + 1):
            try:
                start_time = time.time()
                response = await self.client.chat.completions.create(**kwargs)
                elapsed = time.time() - start_time

                # Log timing info
                logger.debug(f"Chat completion completed in {elapsed:.2f}s (model={model})")

                # Warn if request took unusually long (>30s for reasoning models, >10s for others)
                threshold = 30.0 if any(x in model.lower() for x in ["o1", "gpt-5"]) else 10.0
                if elapsed > threshold:
                    logger.info(
                        f"Chat completion took {elapsed:.2f}s (model={model}, "
                        f"threshold={threshold}s). This is normal for reasoning models."
                    )

                # Extract result
                content = response.choices[0].message.content or ""
                finish_reason = response.choices[0].finish_reason
                usage = UsageInfo.from_openai_usage(response.usage)

                # Update rate limiter with actual usage
                if self.enable_rate_limiting and self.rate_limiter_pool and response.usage:
                    actual_tokens = response.usage.total_tokens
                    # limiter may not be statically visible to the type checker; perform guarded update.
                    try:
                        limiter.update_usage(actual_tokens, estimated_total)  # type: ignore[attr-defined]
                    except Exception:
                        # If limiter isn't present or update fails for some reason, don't crash the flow.
                        pass

                # Track cost
                cost = 0.0
                if track_cost and self.cost_tracker:
                    cost = self.cost_estimator.estimate_cost_from_usage(model, usage)
                    self.cost_tracker.record_cost(
                        category="chat",
                        model=model,
                        tokens_input=usage.prompt_tokens,
                        tokens_output=usage.completion_tokens,
                        tokens_cached_input=usage.cached_prompt_tokens,
                        currency=self.cost_estimator.currency,
                        amount=cost,
                        metadata={"operation": "chat_completion"},
                    )

                if metrics_tracker is not None:
                    metrics_tracker.set_tokens(
                        input=usage.prompt_tokens,
                        output=usage.completion_tokens,
                        cached=usage.cached_prompt_tokens,
                    )
                    metrics_tracker.set_cost(cost)

                # Build the ChatCompletionResult and cache it before returning
                result = ChatCompletionResult(
                    content=content,
                    usage=usage,
                    model=model,
                    finish_reason=finish_reason,
                    raw_response=response,
                )

                # Cache the result if enabled
                if use_cache and self.enable_cache and self.cache_manager:
                    try:
                        cache_data = {
                            "content": content,
                            "usage": usage.to_dict(),
                            "model": model,
                            "finish_reason": finish_reason,
                        }
                        self.cache_manager.chat_cache.set(
                            messages=full_messages,
                            model=model,
                            response=cache_data,
                            temperature=temperature,
                            max_tokens=max_tokens,
                        )
                    except Exception as e:
                        logger.warning(f"Failed to write chat cache: {e}")

                return result

            except BadRequestError as e:
                error_msg = str(e)
                logger.warning(f"[DEBUG] BadRequestError (attempt {attempt}/{max_attempts}): {error_msg}")
                print(f"\n[DEBUG-PRINT] BadRequestError: {error_msg}\n[DEBUG-PRINT] kwargs: {kwargs}\n", flush=True)
                logger.warning(f"[DEBUG] Request kwargs: {kwargs}")

                # Handle parameter conflicts
                if "temperature" in error_msg and "temperature" in kwargs:
                    logger.debug(f"Removing temperature parameter (attempt {attempt}/{max_attempts})")
                    kwargs.pop("temperature", None)
                    continue

                if "max_tokens" in error_msg and "max_tokens" in kwargs:
                    # Some newer models expect 'max_completion_tokens' instead of 'max_tokens'.
                    # If the API message hints at that, map the parameter and retry.
                    if "max_completion_tokens" in error_msg or "Use 'max_completion_tokens' instead" in error_msg:
                        logger.debug(
                            f"Mapping max_tokens -> max_completion_tokens for model {model} (attempt {attempt}/{max_attempts})"
                        )
                        kwargs["max_completion_tokens"] = kwargs.pop("max_tokens", None)
                        continue
                    logger.debug(f"Removing max_tokens parameter (attempt {attempt}/{max_attempts})")
                    kwargs.pop("max_tokens", None)
                    continue

                if "reasoning_effort" in error_msg and "reasoning_effort" in kwargs:
                    logger.debug(f"Removing reasoning_effort parameter (attempt {attempt}/{max_attempts})")
                    kwargs.pop("reasoning_effort", None)
                    continue

                last_error = e
                if attempt < max_attempts:
                    logger.warning(f"Chat completion error (attempt {attempt}/{max_attempts}): {e}")
                    continue
                else:
                    raise

            except APITimeoutError as e:
                # Specific handling for timeout errors with configuration context
                timeout_str = "infinite" if self.config.timeout_seconds is None else f"{self.config.timeout_seconds}s"
                last_error = e
                if attempt < max_attempts:
                    logger.warning(
                        f"API timeout on chat completion (attempt {attempt}/{max_attempts}), "
                        f"configured timeout={timeout_str}, model={model}. Error: {e}"
                    )
                    continue
                else:
                    logger.error(
                        f"API timeout on chat completion after {max_attempts} attempts, "
                        f"configured timeout={timeout_str}, model={model}. "
                        f"Consider increasing AZURE_TIMEOUT_SECONDS if needed."
                    )
                    raise

            except Exception as e:
                last_error = e
                if attempt < max_attempts:
                    logger.warning(f"Chat completion error (attempt {attempt}/{max_attempts}): {e}")
                    continue
                else:
                    raise

        # If we get here, all attempts failed
        if last_error:
            raise last_error
        raise RuntimeError("Chat completion failed after all retry attempts")

    async def chat_completion_stream(
        self,
        messages: list[dict[str, str]],
        system_prompt: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        reasoning_effort: str | None = None,
        response_format: Any | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any | None = None,
    ):
        """
        Stream chat completion tokens as they arrive.

        This method yields chunks of content (strings) as they are produced by the model.
        It is intended for real-time streaming use cases (e.g., terminal UIs, web sockets).

        Args:
            messages: List of message dicts with 'role' and 'content'
            system_prompt: Optional system prompt (prepended to messages)
            model: Optional model override
            temperature: Temperature parameter (not supported by all models)
            max_tokens: Maximum tokens to generate
            reasoning_effort: Reasoning effort for o1/GPT-5 models ("low", "medium", "high")
            response_format: Response format (e.g., {"type": "json_object"})
            tools: Optional list of tool/function definitions for function calling
            tool_choice: Tool choice configuration for function calling

        Yields:
            Content chunks as strings
        """
        model = model or self.config.chat_deployment

        # Build messages
        full_messages: list[dict[str, str]] = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        full_messages.extend(messages)

        # Build API kwargs
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": full_messages,
            "stream": True,
        }

        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if reasoning_effort is not None:
            kwargs["reasoning_effort"] = reasoning_effort
        if response_format is not None:
            kwargs["response_format"] = response_format
        if tools is not None:
            kwargs["tools"] = tools
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice

        # Call streaming API
        async with self.client.chat.completions.stream(**kwargs) as stream:  # type: ignore[attr-defined]
            async for event in stream:
                # Expect chunks with choices and delta content
                try:
                    # Streaming event objects are dynamic and the static type checker may not know their shape.
                    # Use getattr to safely access attributes and silence attribute-access diagnostics.
                    ev = event  # keep original variable name for readability
                    choice = getattr(ev, "choices", [None])[0]
                    delta = getattr(choice, "delta", None) if choice is not None else None
                    if delta and getattr(delta, "content", None):
                        yield delta.content
                except Exception as e:
                    logger.debug(f"Streaming chunk parsing failed: {e}")
                    continue

    # ==================== Token Counting ====================

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.

        Args:
            text: Text to tokenize

        Returns:
            Number of tokens
        """
        return self.config.count_tokens(text)

    def count_message_tokens(self, messages: list[dict[str, str]]) -> int:
        """
        Count tokens in a list of messages.

        Args:
            messages: List of message dicts

        Returns:
            Total number of tokens
        """
        # Approximate token count for messages
        # Format: <role>: <content> with some overhead
        total = 0
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            # Add tokens for role, content, and formatting
            total += self.count_tokens(f"{role}: {content}") + 4
        return total

    # ==================== Utilities ====================

    def estimate_embedding_cost(self, text: str, model: str | None = None) -> float:
        """
        Estimate cost for embedding a single text.

        For batch embedding cost estimation, use PolarsBatchEmbedder.

        Args:
            text: Text to embed
            model: Optional model override

        Returns:
            Estimated cost
        """
        model = model or self.config.embedding_deployment
        tokens = self.count_tokens(text)
        return self.cost_estimator.estimate_cost(model=model, tokens_input=tokens)

    def estimate_chat_cost(
        self,
        messages: list[dict[str, str]],
        estimated_output_tokens: int = 500,
        model: str | None = None,
    ) -> float:
        """
        Estimate cost for a chat completion.

        Args:
            messages: List of message dicts
            estimated_output_tokens: Estimated output tokens
            model: Optional model override

        Returns:
            Estimated cost
        """
        model = model or self.config.chat_deployment
        input_tokens = self.count_message_tokens(messages)
        return self.cost_estimator.estimate_cost(
            model=model,
            tokens_input=input_tokens,
            tokens_output=estimated_output_tokens,
        )


__all__ = [
    "AzureLLMClient",
]
