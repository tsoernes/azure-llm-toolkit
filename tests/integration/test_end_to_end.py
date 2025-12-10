# code/azure-llm-toolkit/tests/integration/test_end_to_end.py
"""
End-to-end integration tests combining core components of azure-llm-toolkit.

These tests are designed to validate that major building blocks work together:

- AzureLLMClient (monkeypatched) for chat and embeddings
- ConversationManager for multi-turn interactions and summarization
- ChatBatchRunner and EmbeddingBatchRunner for logical batching
- StructuredOutputManager for validated structured outputs
- HealthChecker for health/readiness checks

All tests monkeypatch AzureLLMClient methods to avoid real network calls.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List

import pytest
from pydantic import BaseModel

from azure_llm_toolkit import (
    AzureConfig,
    AzureLLMClient,
    BatchStatus,
    ChatBatchItem,
    ChatBatchRunner,
    ConversationConfig,
    ConversationManager,
    EmbeddingBatchItem,
    EmbeddingBatchRunner,
    HealthChecker,
    StructuredOutputManager,
    ValidationRetryExhaustedError,
)


class DummyUsage:
    """Simple dummy usage mimicking OpenAI usage."""

    def __init__(self, prompt_tokens: int = 0, completion_tokens: int = 0) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = prompt_tokens + completion_tokens


class DummyChatResponse:
    """Minimal raw_response-like object (not heavily used in tests)."""

    def __init__(self, content: str, model: str = "test-model") -> None:
        self.choices = []
        self.model = model
        self.id = "dummy"
        self.object = "chat.completion"
        self.created = 0


# =============================================================================
# Pydantic models for structured output
# =============================================================================


class Person(BaseModel):
    name: str
    age: int
    occupation: str


# =============================================================================
# Integration tests
# =============================================================================


@pytest.mark.asyncio
async def test_conversation_manager_end_to_end() -> None:
    """
    ConversationManager integrates with a client, maintains history, and summarizes.

    Flow:
    - Create AzureLLMClient and ConversationManager
    - Send multiple user messages
    - Force summarization by lowering thresholds
    - Verify that a summary is created and history is trimmed
    """
    config_obj = AzureConfig()
    client = AzureLLMClient(config=config_obj, enable_rate_limiting=False, enable_cache=False)

    async def fake_chat_completion(
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
        user_contents = [m.get("content", "") for m in messages if m.get("role") == "user"]
        last_user = user_contents[-1] if user_contents else ""
        if "Summarize the conversation" in last_user or "running summary" in last_user:
            content = "This is a concise summary focusing on recent turns."
        else:
            content = f"dummy-response: {last_user[:80]}"
        usage = DummyUsage(prompt_tokens=10, completion_tokens=5)
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

    client.chat_completion = fake_chat_completion  # type: ignore[assignment]

    config = ConversationConfig(
        max_history_messages=6,
        max_history_tokens=200,
        auto_summarize=True,
        summarize_threshold_messages=3,
        summarize_threshold_tokens=50,
        system_prompt="You are a helpful assistant.",
        model="gpt-4o",
    )
    manager = ConversationManager(client=client, config=config)

    # Simulate a short conversation
    await manager.send_message("Hi, I'm testing the conversation manager.")
    await manager.send_message("I have a question about Azure OpenAI.")
    await manager.send_message("Can you summarize what we've discussed so far?")

    # At this point, auto_summarize should have kicked in at least once
    # We can't assert exact summary text (since it's dummy), but we can assert:
    # - Summary exists
    # - Message count does not exceed max_history_messages
    assert manager.summary is not None
    assert len(manager.messages) <= config.max_history_messages

    history = manager.get_history(include_metadata=True)
    assert len(history) == len(manager.messages)
    # Last message should be from assistant (dummy client)
    assert history[-1]["role"] == "assistant"


@pytest.mark.asyncio
async def test_batch_runner_and_embeddings_end_to_end() -> None:
    """
    ChatBatchRunner and EmbeddingBatchRunner work with DummyAzureLLMClient together.

    Flow:
    - Run a small chat batch
    - Use the results as text for embedding batch
    - Verify order, IDs, and basic semantics
    """
    config_obj = AzureConfig()
    client = AzureLLMClient(config=config_obj, enable_rate_limiting=False, enable_cache=False)

    async def fake_chat_completion(
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
        user_contents = [m.get("content", "") for m in messages if m.get("role") == "user"]
        last_user = user_contents[-1] if user_contents else ""
        if "Summarize the conversation" in last_user or "running summary" in last_user:
            content = "This is a concise summary focusing on recent turns."
        elif response_format and isinstance(response_format, dict) and response_format.get("type") == "json_object":
            content = '{"name": "Alice", "age": 30, "occupation": "Engineer"}'
        else:
            content = f"dummy-response: {last_user[:80]}"
        usage = DummyUsage(prompt_tokens=10, completion_tokens=5)
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

    async def fake_embed_texts(
        texts: list[str],
        model: str | None = None,
        batch_size: int = 100,
        track_cost: bool = True,
    ):
        embeddings: list[list[float]] = []
        for idx, text in enumerate(texts):
            embeddings.append([float(len(text)), float(idx)])
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

    client.chat_completion = fake_chat_completion  # type: ignore[assignment]
    client.embed_texts = fake_embed_texts  # type: ignore[assignment]

    # --- Chat batch ---
    chat_items = [
        ChatBatchItem(
            id=f"q{i}",
            messages=[{"role": "user", "content": f"Question {i} about embeddings"}],
        )
        for i in range(4)
    ]
    chat_runner = ChatBatchRunner(client, max_concurrent=2)
    chat_results = await chat_runner.run(chat_items)

    assert len(chat_results) == len(chat_items)
    for i, res in enumerate(chat_results):
        assert res.id == f"q{i}"
        assert res.status == BatchStatus.SUCCESS
        assert res.response is not None
        assert "Question" in res.response.content

    # --- Embedding batch based on chat results ---
    embed_items = [
        EmbeddingBatchItem(
            id=res.id,
            text=res.response.content if res.response else "",
        )
        for res in chat_results
    ]
    embed_runner = EmbeddingBatchRunner(client, batch_size=2, max_concurrent=2)
    embed_results = await embed_runner.run(embed_items)

    assert len(embed_results) == len(embed_items)
    for i, res in enumerate(embed_results):
        assert res.id == f"q{i}"
        assert res.status == BatchStatus.SUCCESS
        assert res.embedding is not None
        # Our dummy embedding encodes length in position 0
        assert res.embedding[0] == float(len(embed_items[i].text))


@pytest.mark.asyncio
async def test_structured_output_manager_end_to_end() -> None:
    """
    StructuredOutputManager works end-to-end with AzureLLMClient (monkeypatched).

    Flow:
    - Use StructuredOutputManager with Person model
    - Extract structured data from text
    - Then run extract_batch on multiple texts
    """
    config_obj = AzureConfig()
    client = AzureLLMClient(config=config_obj, enable_rate_limiting=False, enable_cache=False)

    async def fake_chat_completion(
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
        user_contents = [m.get("content", "") for m in messages if m.get("role") == "user"]
        last_user = user_contents[-1] if user_contents else ""

        # When JSON structured output is requested, return valid JSON
        if response_format and isinstance(response_format, dict) and response_format.get("type") == "json_object":
            content = '{"name": "Alice", "age": 30, "occupation": "Engineer"}'
        else:
            content = f"dummy-response: {last_user[:80]}"

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

    async def fake_embed_texts(
        texts: list[str],
        model: str | None = None,
        batch_size: int = 100,
        track_cost: bool = True,
    ):
        embeddings: list[list[float]] = []
        for idx, text in enumerate(texts):
            embeddings.append([float(len(text)), float(idx)])
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

    client.chat_completion = fake_chat_completion  # type: ignore[assignment]
    client.embed_texts = fake_embed_texts  # type: ignore[assignment]
    manager = StructuredOutputManager(client)

    text = "Alice is a 30-year-old engineer."
    person = await manager.extract(text, Person, use_cache=True)
    assert isinstance(person, Person)
    assert person.name == "Alice"
    assert person.occupation == "Engineer"

    texts = [
        "Alice is 30 and works as an engineer.",
        "Bob is 40 and works as a teacher.",
    ]
    # Our fake client always returns the same JSON for structured output,
    # but this still exercises the flow.
    people = await manager.extract_batch(texts, Person, use_cache=False)
    assert len(people) == 2
    assert all(isinstance(p, Person) for p in people)


@pytest.mark.asyncio
async def test_structured_output_validation_failure_propagates() -> None:
    """
    StructuredOutputManager surfaces validation exhaustion properly.

    Flow:
    - Use a client that returns invalid JSON for structured extraction
    - Expect ValidationRetryExhaustedError
    """

    config_obj = AzureConfig()
    real_client = AzureLLMClient(config=config_obj, enable_rate_limiting=False, enable_cache=False)

    async def bad_chat_completion(
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
        # Always return invalid JSON, regardless of response_format
        content = "not-json"
        usage = DummyUsage(prompt_tokens=5, completion_tokens=5)
        return type(
            "BadChatResult",
            (),
            {
                "content": content,
                "usage": usage,
                "model": model or "test-model",
                "finish_reason": "stop",
                "raw_response": DummyChatResponse(content=content, model=model or "test-model"),
            },
        )()

    config_obj = AzureConfig()
    real_client = AzureLLMClient(config=config_obj, enable_rate_limiting=False, enable_cache=False)

    async def bad_chat_completion(*args: Any, **kwargs: Any):
        content = "not-json"
        usage = DummyUsage(prompt_tokens=5, completion_tokens=5)
        return type(
            "BadChatResult",
            (),
            {
                "content": content,
                "usage": usage,
                "model": "test-model",
                "finish_reason": "stop",
                "raw_response": DummyChatResponse(content=content, model="test-model"),
            },
        )()

    real_client.chat_completion = bad_chat_completion  # type: ignore[assignment]
    manager = StructuredOutputManager(real_client)
    text = "This will fail."

    with pytest.raises(ValidationRetryExhaustedError):
        await manager.extract(text, Person, use_cache=False, max_retries=2)  # type: ignore[call-arg]


@pytest.mark.asyncio
async def test_health_checker_end_to_end() -> None:
    """
    HealthChecker integrates with DummyAzureLLMClient.

    Flow:
    - Create DummyAzureLLMClient and HealthChecker
    - Run liveness, readiness, and full health checks
    """
    config_obj = AzureConfig()
    client = AzureLLMClient(config=config_obj, enable_rate_limiting=False, enable_cache=False)

    async def fake_chat_completion(
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
        # Always return JSON for structured output
        content = '{"name": "Alice", "age": 30, "occupation": "Engineer"}'
        usage = DummyUsage(prompt_tokens=5, completion_tokens=5)
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

    client.chat_completion = fake_chat_completion  # type: ignore[assignment]
    checker = HealthChecker(client=client, version="test-version")

    is_alive = await checker.check_liveness()
    assert is_alive

    is_ready = await checker.check_readiness()
    assert is_ready

    result = await checker.check_health(include_api_check=False)
    assert result.is_ready
    assert result.status in (result.status.HEALTHY, result.status.DEGRADED)
    info = checker.get_info()
    assert info["version"] == "test-version"
    assert "config" in info
    assert "features" in info


@pytest.mark.asyncio
async def test_full_flow_conversation_batch_structured_health() -> None:
    """
    Full-flow integration:

    - Use ConversationManager with DummyAzureLLMClient
    - Send a few messages and get summary
    - Run a small chat batch and embed the responses
    - Use StructuredOutputManager to parse one result
    - Run HealthChecker to validate overall health
    """
    config_obj = AzureConfig()
    client = AzureLLMClient(config=config_obj, enable_rate_limiting=False, enable_cache=False)

    async def fake_chat_completion(
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
        user_contents = [m.get("content", "") for m in messages if m.get("role") == "user"]
        last_user = user_contents[-1] if user_contents else ""

        # When JSON structured output is requested, return valid JSON
        if response_format and isinstance(response_format, dict) and response_format.get("type") == "json_object":
            content = '{"name": "Alice", "age": 30, "occupation": "Engineer"}'
        else:
            content = f"dummy-response: {last_user[:80]}"

        usage = DummyUsage(prompt_tokens=5, completion_tokens=5)
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

    async def fake_embed_texts(
        texts: list[str],
        model: str | None = None,
        batch_size: int = 100,
        track_cost: bool = True,
    ):
        embeddings: list[list[float]] = []
        for idx, text in enumerate(texts):
            embeddings.append([float(len(text)), float(idx)])
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

    client.chat_completion = fake_chat_completion  # type: ignore[assignment]
    client.embed_texts = fake_embed_texts  # type: ignore[assignment]

    # Conversation
    conv_config = ConversationConfig(
        max_history_messages=5,
        max_history_tokens=200,
        auto_summarize=True,
        summarize_threshold_messages=3,
        summarize_threshold_tokens=40,
        system_prompt="You are an integration-test assistant.",
        model="gpt-4o",
    )
    conv_manager = ConversationManager(client=client, config=conv_config)

    await conv_manager.send_message("Hello, let's test an end-to-end flow.")
    await conv_manager.send_message("First, we'll talk a bit.")
    await conv_manager.send_message("Then, you might summarize this conversation.")

    assert conv_manager.summary is not None

    # Chat batch
    chat_items = [
        ChatBatchItem(
            id="batch-1",
            messages=[{"role": "user", "content": "What did we just do in the conversation?"}],
        )
    ]
    chat_runner = ChatBatchRunner(client, max_concurrent=1)
    chat_results = await chat_runner.run(chat_items)
    assert len(chat_results) == 1
    assert chat_results[0].status == BatchStatus.SUCCESS
    assert chat_results[0].response is not None

    # Embed the chat response
    embed_items = [
        EmbeddingBatchItem(
            id="embed-1",
            text=chat_results[0].response.content if chat_results[0].response else "",
        )
    ]
    embed_runner = EmbeddingBatchRunner(client, batch_size=1, max_concurrent=1)
    embed_results = await embed_runner.run(embed_items)
    assert len(embed_results) == 1
    assert embed_results[0].status == BatchStatus.SUCCESS
    assert embed_results[0].embedding is not None

    # Structured output on synthetic text
    struct_manager = StructuredOutputManager(client)
    person = await struct_manager.extract("Alice is a 30-year-old engineer.", Person, use_cache=False)
    assert person.name == "Alice"

    # Health check
    checker = HealthChecker(client=client, version="integration-test")
    health = await checker.check_health(include_api_check=False)
    assert health.is_ready
