"""Azure OpenAI client wrapper with rate limiting, retry logic, and cost tracking."""

from __future__ import annotations

import hashlib
import json
import logging
import re
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

from .config import AzureConfig
from .cost_tracker import CostEstimator, CostTracker
from .rate_limiter import RateLimiter, RateLimiterPool
from .types import ChatCompletionResult, EmbeddingResult, QueryRewriteResult, UsageInfo

logger = logging.getLogger(__name__)


def _hash_payload(data: Any) -> str:
    """Generate short hash of payload for retry logging."""
    payload_str = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(payload_str.encode()).hexdigest()[:8]


def _log_retry_attempt(retry_state):
    """Custom callback to log retry attempts with payload hash."""
    exception = retry_state.outcome.exception()
    attempt = retry_state.attempt_number
    wait_time = retry_state.next_action.sleep if retry_state.next_action else 0

    # Try to extract payload from args/kwargs
    args = retry_state.args
    kwargs = retry_state.kwargs
    payload_hash = "unknown"

    # For embeddings: first arg is batch
    if args and len(args) > 1 and isinstance(args[1], (list, tuple)):
        payload_hash = _hash_payload({"batch_size": len(args[1]), "first_item": args[1][0] if args[1] else None})
    # For chat: messages in kwargs
    elif "messages" in kwargs:
        payload_hash = _hash_payload({"messages": kwargs["messages"]})

    logger.warning(
        f"Retry attempt {attempt} after {exception.__class__.__name__}: {exception} "
        f"(payload_hash={payload_hash}, wait={wait_time:.1f}s)"
    )


class AzureLLMClient:
    """
    Azure OpenAI client with advanced features:

    - Automatic retry logic with exponential backoff
    - Rate limiting (TPM/RPM)
    - Cost tracking and estimation
    - Batch embedding support
    - Chat completions with reasoning support
    - Query rewriting utilities
    - Token counting
    """

    def __init__(
        self,
        config: AzureConfig | None = None,
        client: AsyncAzureOpenAI | None = None,
        cost_estimator: CostEstimator | None = None,
        cost_tracker: CostTracker | None = None,
        rate_limiter_pool: RateLimiterPool | None = None,
        enable_rate_limiting: bool = True,
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
        """
        self.config = config or AzureConfig()
        self.client = client or self.config.create_client()
        self.cost_estimator = cost_estimator or CostEstimator(currency="kr")
        self.cost_tracker = cost_tracker
        self.enable_rate_limiting = enable_rate_limiting

        if enable_rate_limiting:
            self.rate_limiter_pool = rate_limiter_pool or RateLimiterPool()
        else:
            self.rate_limiter_pool = None

    # ==================== Embeddings ====================

    @retry(
        reraise=True,
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((APIConnectionError, RateLimitError, APITimeoutError, APIStatusError)),
        before_sleep=_log_retry_attempt,
    )
    async def _embed_batch(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        """
        Embed a batch of texts with retry logic.

        Args:
            texts: List of texts to embed
            model: Optional model override (uses config default if not provided)

        Returns:
            List of embedding vectors
        """
        model = model or self.config.embedding_deployment

        # Rate limiting
        if self.enable_rate_limiting and self.rate_limiter_pool:
            # Estimate tokens for rate limiting
            total_tokens = sum(self.config.count_tokens(t) for t in texts)
            limiter = await self.rate_limiter_pool.get_limiter(model)
            await limiter.acquire(tokens=total_tokens)

        response = await self.client.embeddings.create(model=model, input=texts)

        embeddings = [item.embedding for item in response.data]

        # Update rate limiter with actual usage if available
        if self.enable_rate_limiting and self.rate_limiter_pool and response.usage:
            actual_tokens = response.usage.total_tokens
            limiter.update_usage(actual_tokens, total_tokens)

        return embeddings

    async def embed_texts(
        self,
        texts: list[str],
        batch_size: int = 100,
        model: str | None = None,
        track_cost: bool = True,
    ) -> EmbeddingResult:
        """
        Embed multiple texts with automatic batching.

        Args:
            texts: List of texts to embed
            batch_size: Maximum batch size for API calls
            model: Optional model override
            track_cost: Whether to track cost (requires cost_tracker)

        Returns:
            EmbeddingResult with embeddings and usage info
        """
        if not texts:
            return EmbeddingResult(embeddings=[], model=model or self.config.embedding_deployment, usage=UsageInfo())

        model = model or self.config.embedding_deployment
        all_embeddings: list[list[float]] = []
        total_tokens = 0

        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch_embeddings = await self._embed_batch(batch, model=model)
            all_embeddings.extend(batch_embeddings)

            # Estimate tokens for this batch
            batch_tokens = sum(self.config.count_tokens(t) for t in batch)
            total_tokens += batch_tokens

        usage = UsageInfo(prompt_tokens=total_tokens, total_tokens=total_tokens)

        # Track cost if enabled
        if track_cost and self.cost_tracker:
            cost = self.cost_estimator.estimate_cost_from_usage(model, usage)
            self.cost_tracker.record_cost(
                category="embedding",
                model=model,
                tokens_input=usage.prompt_tokens,
                tokens_output=0,
                tokens_cached_input=0,
                currency=self.cost_estimator.currency,
                amount=cost,
            )

        return EmbeddingResult(embeddings=all_embeddings, model=model, usage=usage)

    async def embed_text(self, text: str, model: str | None = None) -> list[float]:
        """
        Embed a single text.

        Args:
            text: Text to embed
            model: Optional model override

        Returns:
            Embedding vector
        """
        result = await self.embed_texts([text], model=model)
        return result.embeddings[0]

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

        Returns:
            ChatCompletionResult with content and usage info
        """
        model = model or self.config.chat_deployment

        # Build messages
        full_messages: list[dict[str, str]] = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        full_messages.extend(messages)

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

        # Make API call with retry on specific errors
        max_attempts = 3
        last_error: Exception | None = None

        for attempt in range(1, max_attempts + 1):
            try:
                response = await self.client.chat.completions.create(**kwargs)

                # Extract result
                content = response.choices[0].message.content or ""
                finish_reason = response.choices[0].finish_reason
                usage = UsageInfo.from_openai_usage(response.usage)

                # Update rate limiter with actual usage
                if self.enable_rate_limiting and self.rate_limiter_pool and response.usage:
                    actual_tokens = response.usage.total_tokens
                    limiter.update_usage(actual_tokens, estimated_total)

                # Track cost
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
                    )

                return ChatCompletionResult(
                    content=content,
                    usage=usage,
                    model=model,
                    finish_reason=finish_reason,
                    raw_response=response,
                )

            except BadRequestError as e:
                error_msg = str(e)

                # Handle parameter conflicts
                if "temperature" in error_msg and "temperature" in kwargs:
                    logger.debug(f"Removing temperature parameter (attempt {attempt}/{max_attempts})")
                    kwargs.pop("temperature", None)
                    continue

                if "max_tokens" in error_msg and "max_tokens" in kwargs:
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

    async def generate_answer(
        self,
        question: str,
        context: str,
        system_prompt: str | None = None,
        model: str | None = None,
        track_cost: bool = True,
    ) -> ChatCompletionResult:
        """
        Generate an answer to a question given context (RAG-style).

        Args:
            question: User question
            context: Retrieved context
            system_prompt: Optional system prompt
            model: Optional model override
            track_cost: Whether to track cost

        Returns:
            ChatCompletionResult with answer
        """
        default_system = (
            "You are a helpful assistant. Answer the user's question based on the provided context. "
            "If the context doesn't contain relevant information, say so clearly."
        )

        messages = [
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}",
            }
        ]

        return await self.chat_completion(
            messages=messages,
            system_prompt=system_prompt or default_system,
            model=model,
            track_cost=track_cost,
        )

    # ==================== Query Rewriting ====================

    async def rewrite_query(
        self,
        query: str,
        model: str | None = None,
        track_cost: bool = True,
    ) -> QueryRewriteResult:
        """
        Rewrite a query for better retrieval.

        Args:
            query: Original query
            model: Optional model override
            track_cost: Whether to track cost

        Returns:
            QueryRewriteResult with original and rewritten queries
        """
        system_prompt = (
            "You are an expert at reformulating search queries to improve retrieval. "
            "Rewrite the user's query to be more specific and search-friendly. "
            "Output ONLY the rewritten query, nothing else."
        )

        messages = [{"role": "user", "content": query}]

        result = await self.chat_completion(
            messages=messages,
            system_prompt=system_prompt,
            model=model,
            max_tokens=200,
            track_cost=track_cost,
        )

        rewritten = result.content.strip()

        return QueryRewriteResult(
            original=query,
            rewritten=rewritten,
            raw_response=result.content,
        )

    # ==================== Metadata Extraction ====================

    async def extract_metadata_from_filename(
        self,
        filename: str,
        model: str | None = None,
        track_cost: bool = True,
    ) -> dict[str, Any]:
        """
        Extract metadata from a filename using LLM.

        Args:
            filename: Filename to analyze
            model: Optional model override
            track_cost: Whether to track cost

        Returns:
            Extracted metadata as dict
        """
        system_prompt = (
            "Extract metadata from the given filename. Return a JSON object with relevant fields like "
            "title, date, author, document_type, etc. Only include fields you can infer with confidence."
        )

        messages = [{"role": "user", "content": f"Filename: {filename}"}]

        result = await self.chat_completion(
            messages=messages,
            system_prompt=system_prompt,
            model=model,
            response_format={"type": "json_object"},
            track_cost=track_cost,
        )

        try:
            metadata = json.loads(result.content)
            return metadata
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse metadata JSON from filename: {filename}")
            return {}

    async def extract_metadata_from_content(
        self,
        content: str,
        filename: str | None = None,
        max_content_length: int = 4000,
        model: str | None = None,
        track_cost: bool = True,
    ) -> dict[str, Any]:
        """
        Extract metadata from document content using LLM.

        Args:
            content: Document content
            filename: Optional filename for context
            max_content_length: Maximum content length to send (will truncate)
            model: Optional model override
            track_cost: Whether to track cost

        Returns:
            Extracted metadata as dict
        """
        # Truncate content if too long
        if len(content) > max_content_length:
            content = content[:max_content_length] + "..."

        system_prompt = (
            "Extract metadata from the given document content. Return a JSON object with relevant fields like "
            "title, summary, topics, key_entities, document_type, etc. Only include fields you can infer with confidence."
        )

        user_content = f"Content:\n{content}"
        if filename:
            user_content = f"Filename: {filename}\n\n{user_content}"

        messages = [{"role": "user", "content": user_content}]

        result = await self.chat_completion(
            messages=messages,
            system_prompt=system_prompt,
            model=model,
            response_format={"type": "json_object"},
            track_cost=track_cost,
        )

        try:
            metadata = json.loads(result.content)
            return metadata
        except json.JSONDecodeError:
            logger.warning("Failed to parse metadata JSON from content")
            return {}

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
        Estimate cost for embedding a text.

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
