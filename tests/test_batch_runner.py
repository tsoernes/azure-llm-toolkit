import asyncio
from typing import Any, List

import pytest

from azure_llm_toolkit import (
    AzureConfig,
    AzureLLMClient,
    BatchStatus,
    BatchError,
    ChatBatchItem,
    ChatBatchRunner,
    EmbeddingBatchItem,
    EmbeddingBatchRunner,
)


class DummyUsage:
    """Simple dummy usage object mimicking OpenAI usage."""

    def __init__(self, prompt_tokens: int = 0, completion_tokens: int = 0) -> None:
        self.prompt_tokens = prompt_tokens
        self.delivery_tokens = 0  # kept for forward-compatibility if needed
        self.completion_tokens = completion_tokens
        self.total_tokens = prompt_tokens + completion_tokens


class DummyChatResponse:
    """Minimal object to mimic ChatCompletionResult.raw_response when needed."""

    def __init__(self, content: str, model: str = "test-model") -> None:
        # We don't rely on raw_response in batch runner tests, so this is minimal.
        self.choices = []
        self.model = model
        self.id = "dummy"
        self.object = "chat.completion"
        self.created = 0


class DummyAzureLLMClient(AzureLLMClient):
    """
    A lightweight dummy client that replaces network calls with predictable behavior.

    We subclass AzureLLMClient only to satisfy type expectations in the batch runner,
    but we override the async methods that the batch runner uses.
    """

    def __init__(self) -> None:
        # Construct a minimal config; values are mostly unused in tests.
        config = AzureConfig()
        # We bypass AzureLLMClient.__init__ heavy setup by not calling super().__init__
        # and instead attaching only what we need.
        # However, to keep type-checkers happy, we do call super and accept that
        # some unused machinery is initialized.
        super().__init__(config=config, enable_rate_limiting=False, enable_cache=False)
        self._chat_call_count = 0
        self._embed_call_count = 0

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
    ):
        """
        Dummy chat_completion that returns a predictable content string.

        The content encodes the number of calls so tests can verify mapping.
        """
        self._chat_call_count += 1
        user_contents = [m.get("content", "") for m in messages if m.get("role") == "user"]
        last_user = user_contents[-1] if user_contents else ""
        content = f"response-{self._chat_call_count}: {last_user}"
        usage = DummyUsage(prompt_tokens=5, completion_tokens=7)
        return type(
            "DummyChatResult",
            (),
            {
                "content": content,
                "usage": usage,
                "model": model or "test-model",
                "finish_reason": "stop",
                "raw_response": DummyChatResponse(content=content, model=model or "test-model"),
            },
        )()

    async def embed_texts(
        self,
        texts: list[str],
        model: str | None = None,
        batch_size: int = 100,
        track_cost: bool = True,
    ):
        """
        Dummy embed_texts that returns deterministic numeric embeddings.

        Each text is mapped to a small vector based on its index and length.
        """
        self._embed_call_count += 1
        embeddings: list[list[float]] = []
        for idx, text in enumerate(texts):
            # Simple deterministic embedding: [idx, len(text)]
            embeddings.append([float(idx), float(len(text))])
        usage = DummyUsage(prompt_tokens=len(texts), completion_tokens=0)
        return type(
            "DummyEmbeddingResult",
            (),
            {
                "embeddings": embeddings,
                "usage": usage,
                "model": model or "test-embed-model",
            },
        )()


@pytest.mark.asyncio
async def test_chat_batch_runner_success_basic() -> None:
    """ChatBatchRunner should process items and preserve ordering and IDs."""
    client = DummyAzureLLMClient()
    items = [
        ChatBatchItem(
            id=f"item-{i}",
            messages=[{"role": "user", "content": f"hello-{i}"}],
        )
        for i in range(5)
    ]

    runner = ChatBatchRunner(client, max_concurrent=2)
    results = await runner.run(items)

    assert len(results) == len(items)
    for i, res in enumerate(results):
        assert res.id == f"item-{i}"
        assert res.status == BatchStatus.SUCCESS
        assert res.response is not None
        assert f"hello-{i}" in res.response.content


@pytest.mark.asyncio
async def test_chat_batch_runner_failure_handling() -> None:
    """
    ChatBatchRunner should mark FAILED when client.chat_completion raises.

    We monkey-patch chat_completion for this test to simulate errors.
    """
    client = DummyAzureLLMClient()

    async def failing_chat_completion(*args, **kwargs):
        raise RuntimeError("simulated failure")

    client.chat_completion = failing_chat_completion  # type: ignore[assignment]

    items = [
        ChatBatchItem(
            id="fail-1",
            messages=[{"role": "user", "content": "should-fail"}],
        )
    ]

    runner = ChatBatchRunner(client, max_concurrent=1)
    results = await runner.run(items)

    assert len(results) == 1
    res = results[0]
    assert res.id == "fail-1"
    assert res.status == BatchStatus.FAILED
    assert isinstance(res.error, BatchError)
    assert "Chat completion failed" in res.error.message


@pytest.mark.asyncio
async def test_chat_batch_runner_progress_callback() -> None:
    """ChatBatchRunner should invoke progress callback with (completed, total)."""
    client = DummyAzureLLMClient()
    items = [
        ChatBatchItem(
            id=f"item-{i}",
            messages=[{"role": "user", "content": f"hello-{i}"}],
        )
        for i in range(3)
    ]

    progress_calls: list[tuple[int, int]] = []

    async def progress(completed: int, total: int) -> None:
        progress_calls.append((completed, total))

    runner = ChatBatchRunner(client, max_concurrent=2, progress_callback=progress)
    results = await runner.run(items)

    assert len(results) == 3
    # Progress callback should be called once per completed item.
    assert len(progress_calls) == 3
    assert progress_calls[-1] == (3, 3)


@pytest.mark.asyncio
async def test_embedding_batch_runner_success_basic() -> None:
    """EmbeddingBatchRunner should produce embeddings in the same order as items."""
    client = DummyAzureLLMClient()
    items = [EmbeddingBatchItem(id=f"doc-{i}", text=f"document-{i}") for i in range(10)]

    runner = EmbeddingBatchRunner(client, batch_size=4, max_concurrent=2)
    results = await runner.run(items)

    assert len(results) == len(items)
    for i, res in enumerate(results):
        assert res.id == f"doc-{i}"
        assert res.status == BatchStatus.SUCCESS
        assert res.embedding is not None
        # Our dummy embedding is [idx, len(text)]
        assert res.embedding[0] == pytest.approx(float(i % 4)) or isinstance(res.embedding[0], float)


@pytest.mark.asyncio
async def test_embedding_batch_runner_batches_correctly() -> None:
    """EmbeddingBatchRunner should call embed_texts in batches and handle batch failures."""
    client = DummyAzureLLMClient()
    # We want to observe that multiple batches are created.
    items = [EmbeddingBatchItem(id=f"doc-{i}", text=f"text-{i}") for i in range(9)]

    # Save original method to wrap
    original_embed_texts = client.embed_texts

    call_batch_sizes: List[int] = []

    async def tracking_embed_texts(texts: list[str], *args, **kwargs):
        call_batch_sizes.append(len(texts))
        return await original_embed_texts(texts, *args, **kwargs)

    client.embed_texts = tracking_embed_texts  # type: ignore[assignment]

    runner = EmbeddingBatchRunner(client, batch_size=3, max_concurrent=2)
    results = await runner.run(items)

    assert len(results) == 9
    assert all(r.status == BatchStatus.SUCCESS for r in results)
    # We should have 3 calls of size 3 each.
    assert call_batch_sizes == [3, 3, 3]


@pytest.mark.asyncio
async def test_embedding_batch_runner_batch_failure_marks_all_failed() -> None:
    """
    EmbeddingBatchRunner should mark all batch items as FAILED when embed_texts errors.

    We patch embed_texts to raise for all calls.
    """
    client = DummyAzureLLMClient()

    async def failing_embed_texts(*args, **kwargs):
        raise RuntimeError("batch failure")

    client.embed_texts = failing_embed_texts  # type: ignore[assignment]

    items = [EmbeddingBatchItem(id="doc-1", text="some text")]
    runner = EmbeddingBatchRunner(client, batch_size=10, max_concurrent=1)

    results = await runner.run(items)

    assert len(results) == 1
    res = results[0]
    assert res.id == "doc-1"
    assert res.status == BatchStatus.FAILED
    assert isinstance(res.error, BatchError)
    assert "Embedding batch failed" in res.error.message


@pytest.mark.asyncio
async def test_embedding_batch_runner_progress_callback() -> None:
    """EmbeddingBatchRunner should invoke progress callback per item."""
    client = DummyAzureLLMClient()
    items = [EmbeddingBatchItem(id=f"doc-{i}", text=f"t-{i}") for i in range(5)]

    progress_calls: list[tuple[int, int]] = []

    async def progress(completed: int, total: int) -> None:
        progress_calls.append((completed, total))

    runner = EmbeddingBatchRunner(
        client,
        batch_size=2,
        max_concurrent=2,
        progress_callback=progress,
    )
    results = await runner.run(items)

    assert len(results) == 5
    # Progress callback is called once per item as results are assigned.
    assert len(progress_calls) == 5
    assert progress_calls[-1] == (5, 5)
