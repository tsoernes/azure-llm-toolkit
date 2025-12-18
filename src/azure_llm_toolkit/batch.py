"""
Logical batch runner for chat and embeddings (Feature #4 - partial implementation).

This module provides a high-level, *logical* batch runner that uses the existing
AzureLLMClient to execute many chat or embedding requests efficiently with:

- Automatic batching
- Concurrency control
- Integration with existing rate limiting and cost tracking
- Progress reporting hooks
- Simple, composable API

It does *not* implement the full Azure OpenAI Batch REST API (with Azure Blob
storage etc.). Instead, it offers an ergonomic way to process many requests
using the normal chat/embedding endpoints while respecting rate limits.

Example usage
-------------

    from azure_llm_toolkit import AzureConfig, AzureLLMClient
    from azure_llm_toolkit.batch import (
        ChatBatchItem,
        ChatBatchRunner,
        EmbeddingBatchRunner,
        EmbeddingBatchItem,
    )

    config = AzureConfig()
    client = AzureLLMClient(config=config)

    # Chat batch
    items = [
        ChatBatchItem(
            id=f"q{i}",
            messages=[{"role": "user", "content": f"Question {i}"}],
        )
        for i in range(100)
    ]

    runner = ChatBatchRunner(client, max_concurrent=5)
    results = await runner.run(items)

    # Embedding batch
    embed_items = [
        EmbeddingBatchItem(id=f"d{i}", text=f"Document {i}") for i in range(1000)
    ]
    embed_runner = EmbeddingBatchRunner(client, batch_size=64, max_concurrent=4)
    embed_results = await embed_runner.run(embed_items)



Design notes
------------

This module is intentionally "logical batch" only. It does *not* handle:

- Uploading JSONL files to Azure Blob storage
- Creating server-side batch jobs via dedicated REST endpoints
- Polling batch job status

Those are significantly more involved and environment-specific. This
implementation focuses on:

- High-level ergonomics for bulk processing
- Leveraging existing rate limiting & cost tracking
- Keeping code local and testable
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Coroutine, Dict, Iterable, List, Optional

from .client import AzureLLMClient
from .types import ChatCompletionResult, EmbeddingResult

logger = logging.getLogger(__name__)


# =============================================================================
# Shared types
# =============================================================================


class BatchStatus(str, Enum):
    """Status for a batch item."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class BatchError:
    """Represents an error for a single batch item."""

    message: str
    exception: Optional[Exception] = None

    def __str__(self) -> str:
        if self.exception:
            return f"{self.message}: {self.exception!r}"
        return self.message


@dataclass
class BaseBatchItem:
    """
    Base class for batch items.

    Attributes:
        id: Logical identifier for the item.
        metadata: Optional metadata passed through to results.
    """

    id: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BaseBatchResult:
    """
    Base class for batch results.

    Attributes:
        id: Identifier matching the input item.
        status: Final status of the item.
        error: Error information if status == FAILED.
        metadata: Metadata from the input item (copied through).
    """

    id: str
    status: BatchStatus
    error: Optional[BatchError] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Chat batch runner
# =============================================================================


@dataclass
class ChatBatchItem(BaseBatchItem):
    """
    Represents a single chat completion request in a batch.

    Attributes:
        messages: List of messages (same format as AzureLLMClient.chat_completion).
        system_prompt: Optional system prompt override.
        model: Optional model override.
        temperature: Optional temperature.
        max_tokens: Optional max tokens.
        reasoning_effort: Optional reasoning effort.
        response_format: Optional response format.
        tools: Optional tools for function calling.
        tool_choice: Optional tool choice.
    """

    messages: List[Dict[str, str]] = field(default_factory=list)
    system_prompt: Optional[str] = None
    model: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    reasoning_effort: Optional[str] = None
    response_format: Optional[Any] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Any] = None


@dataclass
class ChatBatchResult(BaseBatchResult):
    """
    Result for a single chat batch item.

    Attributes:
        response: ChatCompletionResult on success, None otherwise.
    """

    response: Optional[ChatCompletionResult] = None


ProgressCallback = Callable[[int, int], Awaitable[None]]


class ChatBatchRunner:
    """
    Logical batch runner for chat completions.

    Features:
        - Concurrency control via asyncio.Semaphore
        - Integration with AzureLLMClient retry & rate limiting
        - Optional progress callback
        - Returns per-item results with status and errors
    """

    def __init__(
        self,
        client: AzureLLMClient,
        max_concurrent: int = 8,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> None:
        """
        Initialize chat batch runner.

        Args:
            client: AzureLLMClient instance to use for requests.
            max_concurrent: Maximum number of concurrent chat completions.
            progress_callback: Optional async callback (completed, total).
        """
        if max_concurrent < 1:
            raise ValueError("max_concurrent must be >= 1")

        self.client = client
        self.max_concurrent = max_concurrent
        self.progress_callback = progress_callback

    async def _run_single(
        self,
        item: ChatBatchItem,
        semaphore: asyncio.Semaphore,
    ) -> ChatBatchResult:
        """Execute a single chat item within the concurrency semaphore."""
        async with semaphore:
            try:
                logger.debug("ChatBatchRunner: starting item %s", item.id)

                response = await self.client.chat_completion(
                    messages=item.messages,
                    system_prompt=item.system_prompt,
                    model=item.model,
                    temperature=item.temperature,
                    max_tokens=item.max_tokens,
                    reasoning_effort=item.reasoning_effort,
                    response_format=item.response_format,
                    track_cost=True,
                    use_cache=True,
                    tools=item.tools,
                    tool_choice=item.tool_choice,
                )

                logger.debug("ChatBatchRunner: completed item %s", item.id)

                return ChatBatchResult(
                    id=item.id,
                    status=BatchStatus.SUCCESS,
                    response=response,
                    metadata=item.metadata,
                )

            except Exception as e:
                logger.warning("ChatBatchRunner: item %s failed: %r", item.id, e)
                return ChatBatchResult(
                    id=item.id,
                    status=BatchStatus.FAILED,
                    error=BatchError(message="Chat completion failed", exception=e),
                    metadata=item.metadata,
                )

    async def run(
        self,
        items: Iterable[ChatBatchItem],
    ) -> List[ChatBatchResult]:
        """
        Run a batch of chat completion items.

        Args:
            items: Iterable of ChatBatchItem instances.

        Returns:
            List of ChatBatchResult in the same order as input.
        """
        items_list = list(items)
        total = len(items_list)
        if total == 0:
            return []

        semaphore = asyncio.Semaphore(self.max_concurrent)
        results: List[ChatBatchResult] = [None] * total  # type: ignore
        completed = 0

        async def worker(idx: int, item: ChatBatchItem) -> None:
            nonlocal completed
            result = await self._run_single(item, semaphore)
            results[idx] = result
            completed += 1
            if self.progress_callback:
                try:
                    await self.progress_callback(completed, total)
                except Exception as e:
                    logger.debug("ChatBatchRunner: progress callback failed: %r", e)

        await asyncio.gather(
            *(worker(idx, item) for idx, item in enumerate(items_list)),
            return_exceptions=False,
        )

        return results


# =============================================================================
# Embedding batch runner
# =============================================================================


@dataclass
class EmbeddingBatchItem(BaseBatchItem):
    """
    Represents a single text to embed in a batch.

    Attributes:
        text: Text to embed.
        model: Optional model override.
    """

    text: str = ""
    model: Optional[str] = None


@dataclass
class EmbeddingBatchResult(BaseBatchResult):
    """
    Result for a single embedding batch item.

    Attributes:
        embedding: List[float] for this item (if using single-text embedding),
                   or index into a shared embedding list.
    """

    embedding: Optional[List[float]] = None


class EmbeddingBatchRunner:
    """
    Logical batch runner for embeddings.

    This runner groups items into batches and calls AzureLLMClient.embed_texts
    to leverage the client's internal batching and rate limiting.
    """

    def __init__(
        self,
        client: AzureLLMClient,
        batch_size: int = 64,
        max_concurrent: int = 4,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> None:
        """
        Initialize embedding batch runner.

        Args:
            client: AzureLLMClient instance.
            batch_size: Maximum number of texts per embed_texts() call.
            max_concurrent: Maximum number of concurrent embed_texts calls.
            progress_callback: Optional async callback (completed, total).
        """
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if max_concurrent < 1:
            raise ValueError("max_concurrent must be >= 1")

        self.client = client
        self.batch_size = batch_size
        self.max_concurrent = max_concurrent
        self.progress_callback = progress_callback

    async def _run_batch(
        self,
        batch_items: List[EmbeddingBatchItem],
        semaphore: asyncio.Semaphore,
    ) -> List[EmbeddingBatchResult]:
        """Run a single batch of embedding requests."""
        async with semaphore:
            try:
                logger.debug(
                    "EmbeddingBatchRunner: starting batch of %d items",
                    len(batch_items),
                )

                texts = [item.text for item in batch_items]
                # Use first item's model override if all are the same, otherwise None
                model_overrides = {item.model for item in batch_items if item.model}
                model = model_overrides.pop() if len(model_overrides) == 1 else None

                result: EmbeddingResult = await getattr(self.client, "embed_texts")(
                    texts=texts,
                    model=model,
                    batch_size=len(texts),  # Already batched here
                    track_cost=True,
                )

                embeddings = result.embeddings
                if len(embeddings) != len(batch_items):
                    # Defensive check
                    raise RuntimeError(f"Expected {len(batch_items)} embeddings, got {len(embeddings)}")

                batch_results: List[EmbeddingBatchResult] = []
                for item, emb in zip(batch_items, embeddings):
                    batch_results.append(
                        EmbeddingBatchResult(
                            id=item.id,
                            status=BatchStatus.SUCCESS,
                            embedding=emb,
                            metadata=item.metadata,
                        )
                    )

                logger.debug(
                    "EmbeddingBatchRunner: completed batch of %d items",
                    len(batch_items),
                )
                return batch_results

            except Exception as e:
                logger.warning(
                    "EmbeddingBatchRunner: batch of %d items failed: %r",
                    len(batch_items),
                    e,
                )
                # Mark all items in the batch as failed
                return [
                    EmbeddingBatchResult(
                        id=item.id,
                        status=BatchStatus.FAILED,
                        error=BatchError(
                            message="Embedding batch failed",
                            exception=e,
                        ),
                        metadata=item.metadata,
                    )
                    for item in batch_items
                ]

    async def run(
        self,
        items: Iterable[EmbeddingBatchItem],
    ) -> List[EmbeddingBatchResult]:
        """
        Run a batch of embedding items.

        Args:
            items: Iterable of EmbeddingBatchItem instances.

        Returns:
            List of EmbeddingBatchResult in the same order as input.
        """
        items_list = list(items)
        total = len(items_list)
        if total == 0:
            return []

        # Create batches
        batches: List[List[EmbeddingBatchItem]] = []
        for i in range(0, total, self.batch_size):
            batches.append(items_list[i : i + self.batch_size])

        semaphore = asyncio.Semaphore(self.max_concurrent)
        results: List[EmbeddingBatchResult] = [None] * total  # type: ignore
        completed = 0

        async def batch_worker(batch_index: int, batch_items: List[EmbeddingBatchItem]) -> None:
            nonlocal completed
            start = batch_index * self.batch_size
            batch_results = await self._run_batch(batch_items, semaphore)
            for offset, br in enumerate(batch_results):
                results[start + offset] = br
                completed += 1
                if self.progress_callback:
                    try:
                        await self.progress_callback(completed, total)
                    except Exception as e:
                        logger.debug(
                            "EmbeddingBatchRunner: progress callback failed: %r",
                            e,
                        )

        await asyncio.gather(
            *(batch_worker(idx, batch) for idx, batch in enumerate(batches)),
            return_exceptions=False,
        )

        return results


__all__ = [
    "BatchStatus",
    "BatchError",
    "BaseBatchItem",
    "BaseBatchResult",
    "ChatBatchItem",
    "ChatBatchResult",
    "ChatBatchRunner",
    "EmbeddingBatchItem",
    "EmbeddingBatchResult",
    "EmbeddingBatchRunner",
]
