# code/azure-llm-toolkit/tests/test_validation.py
"""
Unit tests for the validation / structured output module.

These tests cover:
- JSON schema generation from Pydantic models
- Extraction prompt creation
- Robust JSON parsing from LLM-style responses
- Structured chat completion with validation and retry behavior (mocked)
- StructuredOutputManager caching and batch extraction

Note:
These tests mock AzureLLMClient.chat_completion to avoid real API calls.
"""

from __future__ import annotations

import json
from typing import Any, Type

import pytest
from pydantic import BaseModel, ValidationError

from azure_llm_toolkit.validation import (
    StructuredOutputError,
    ValidationRetryExhaustedError,
    StructuredOutputManager,
    generate_json_schema,
    create_extraction_prompt,
    parse_json_response,
    chat_completion_structured,
    extract_structured_data,
)
from azure_llm_toolkit.types import ChatCompletionResult


class DummyUsage:
    """Simple dummy usage object mimicking OpenAI usage."""

    def __init__(self, prompt_tokens: int = 0, completion_tokens: int = 0) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = prompt_tokens + completion_tokens


class DummyClient:
    """
    Minimal dummy client to mock AzureLLMClient.chat_completion.

    chat_completion behavior is controlled via:
    - self._responses: list of strings to return as content per call
    - self._index: current response index
    - self._raise_on: optional index to raise exception on
    """

    def __init__(self, responses: list[str] | None = None, raise_on: int | None = None) -> None:
        self._responses = responses or []
        self._index = 0
        self._raise_on = raise_on
        self.config = type("Cfg", (), {"count_tokens": lambda _self, _text, model=None: len(_text.split())})

    async def chat_completion(self, messages: list[dict[str, str]], **kwargs: Any) -> ChatCompletionResult:
        """Return a ChatCompletionResult with mocked content."""
        if self._raise_on is not None and self._index == self._raise_on:
            self._index += 1
            raise RuntimeError("Simulated chat_completion failure")

        if self._index >= len(self._responses):
            content = "{}"
        else:
            content = self._responses[self._index]
            self._index += 1

        usage = DummyUsage(prompt_tokens=10, completion_tokens=5)
        return ChatCompletionResult(
            content=content,
            usage=usage,
            model=kwargs.get("model", "test-model"),
            finish_reason="stop",
            raw_response=None,
        )


# -----------------------------
# Pydantic Models for Tests
# -----------------------------


class Person(BaseModel):
    name: str
    age: int
    occupation: str
    location: str | None = None


class Book(BaseModel):
    title: str
    author: str
    year: int
    pages: int | None = None


# -----------------------------
# Schema Generation Tests
# -----------------------------


def test_generate_json_schema_person() -> None:
    """Schema generation for Person model should include required fields."""
    schema = generate_json_schema(Person)
    assert schema["type"] == "object"
    assert set(schema["required"]) >= {"name", "age", "occupation"}
    assert "properties" in schema
    assert schema["properties"]["name"]["type"] == "string"
    assert schema["properties"]["age"]["type"] in ("integer", "number")


def test_generate_json_schema_book() -> None:
    """Schema generation for Book model should include title, author, year."""
    schema = generate_json_schema(Book)
    assert "title" in schema["properties"]
    assert "author" in schema["properties"]
    assert "year" in schema["properties"]


# -----------------------------
# Extraction Prompt Tests
# -----------------------------


def test_create_extraction_prompt_contains_schema_and_text() -> None:
    """Extraction prompt should contain schema and source text."""
    schema = generate_json_schema(Person)
    text = "John is a 30 year old engineer living in Oslo."
    instructions = "Focus only on people mentioned explicitly."

    prompt = create_extraction_prompt(text, schema, instructions)

    assert "Extract structured information" in prompt
    assert text in prompt
    assert json.dumps(schema, indent=2) in prompt
    assert "Additional instructions" in prompt
    assert instructions in prompt
    assert "JSON output:" in prompt


# -----------------------------
# JSON Parsing Tests
# -----------------------------


@pytest.mark.parametrize(
    "response,expected",
    [
        ('{"a": 1, "b": 2}', {"a": 1, "b": 2}),
        ('```json\n{"a": 1}\n```', {"a": 1}),
        ('```json\n{\n  "x": 10\n}\n```', {"x": 10}),
        ('Some text before\n{"k": 3}\nSome text after', {"k": 3}),
    ],
)
def test_parse_json_response_valid_cases(response: str, expected: dict[str, Any]) -> None:
    """parse_json_response should handle common formatting issues."""
    parsed = parse_json_response(response)
    assert parsed == expected


def test_parse_json_response_invalid_raises() -> None:
    """Invalid JSON should raise JSONDecodeError."""
    with pytest.raises(json.JSONDecodeError):
        parse_json_response("not json at all")


# -----------------------------
# Structured Chat Completion Tests
# -----------------------------


@pytest.mark.asyncio
async def test_chat_completion_structured_success() -> None:
    """chat_completion_structured should validate and return a Pydantic model on success."""
    # Response is already valid JSON for Person
    person_json = json.dumps({"name": "Alice", "age": 28, "occupation": "Engineer", "location": "Oslo"})
    client = DummyClient(responses=[person_json])

    messages = [{"role": "user", "content": "Extract person info: Alice is 28 and works as an engineer in Oslo."}]

    result = await chat_completion_structured(
        client=client,
        messages=messages,
        response_model=Person,
        max_retries=2,
    )

    assert isinstance(result, Person)
    assert result.name == "Alice"
    assert result.age == 28
    assert result.occupation == "Engineer"
    assert result.location == "Oslo"


@pytest.mark.asyncio
async def test_chat_completion_structured_retry_on_bad_json() -> None:
    """chat_completion_structured should retry when JSON is malformed."""
    bad_response = "Not JSON at all"
    good_response = json.dumps({"name": "Bob", "age": 40, "occupation": "Teacher"})
    client = DummyClient(responses=[bad_response, good_response])

    messages = [{"role": "user", "content": "Extract info about Bob"}]

    result = await chat_completion_structured(
        client=client,
        messages=messages,
        response_model=Person,
        max_retries=3,
    )

    assert isinstance(result, Person)
    assert result.name == "Bob"
    assert result.age == 40


@pytest.mark.asyncio
async def test_chat_completion_structured_retry_on_validation_error() -> None:
    """chat_completion_structured should retry when validation fails."""
    # First response missing required field 'occupation'
    invalid = json.dumps({"name": "Carol", "age": 33})
    valid = json.dumps({"name": "Carol", "age": 33, "occupation": "Designer"})
    client = DummyClient(responses=[invalid, valid])

    messages = [{"role": "user", "content": "Extract info about Carol"}]

    result = await chat_completion_structured(
        client=client,
        messages=messages,
        response_model=Person,
        max_retries=3,
    )

    assert result.occupation == "Designer"


@pytest.mark.asyncio
async def test_chat_completion_structured_exhaust_retries() -> None:
    """chat_completion_structured should raise when all retries fail."""
    # All responses invalid
    client = DummyClient(responses=["not json", "still not json", "nope"])

    messages = [{"role": "user", "content": "Extract info from text"}]

    with pytest.raises(ValidationRetryExhaustedError):
        await chat_completion_structured(
            client=client,
            messages=messages,
            response_model=Person,
            max_retries=3,
        )


# -----------------------------
# High-level Extraction Helper Tests
# -----------------------------


@pytest.mark.asyncio
async def test_extract_structured_data_basic() -> None:
    """extract_structured_data should work end-to-end for simple text."""
    text = "The Great Gatsby is a novel by F. Scott Fitzgerald, published in 1925."
    book_json = json.dumps(
        {
            "title": "The Great Gatsby",
            "author": "F. Scott Fitzgerald",
            "year": 1925,
            "pages": 218,
        }
    )
    client = DummyClient(responses=[book_json])

    result = await extract_structured_data(
        client=client,
        text=text,
        response_model=Book,
        max_retries=2,
    )

    assert isinstance(result, Book)
    assert result.title == "The Great Gatsby"
    assert result.author == "F. Scott Fitzgerald"
    assert result.year == 1925
    assert result.pages == 218


# -----------------------------
# StructuredOutputManager Tests
# -----------------------------


@pytest.mark.asyncio
async def test_structured_output_manager_caching() -> None:
    """StructuredOutputManager should cache results when enabled."""
    text = "John is a 30-year-old engineer."
    person_json = json.dumps({"name": "John", "age": 30, "occupation": "Engineer"})
    client = DummyClient(responses=[person_json])

    manager = StructuredOutputManager(client)
    result1 = await manager.extract(text, Person, use_cache=True)
    result2 = await manager.extract(text, Person, use_cache=True)

    assert isinstance(result1, Person)
    assert result1 is result2  # Same object from cache


@pytest.mark.asyncio
async def test_structured_output_manager_no_cache() -> None:
    """StructuredOutputManager should bypass cache when use_cache=False."""
    text = "Alice is a 25-year-old data scientist."
    person_json_1 = json.dumps({"name": "Alice", "age": 25, "occupation": "Data Scientist"})
    person_json_2 = json.dumps({"name": "Alice", "age": 25, "occupation": "ML Engineer"})
    client = DummyClient(responses=[person_json_1, person_json_2])

    manager = StructuredOutputManager(client)
    result1 = await manager.extract(text, Person, use_cache=False)
    result2 = await manager.extract(text, Person, use_cache=False)

    assert result1.occupation == "Data Scientist"
    assert result2.occupation == "ML Engineer"


@pytest.mark.asyncio
async def test_structured_output_manager_extract_batch() -> None:
    """StructuredOutputManager.extract_batch should return list of validated models."""
    texts = [
        "Tom is a 45-year-old doctor.",
        "Linda is a 38-year-old lawyer.",
    ]
    responses = [
        json.dumps({"name": "Tom", "age": 45, "occupation": "Doctor"}),
        json.dumps({"name": "Linda", "age": 38, "occupation": "Lawyer"}),
    ]
    client = DummyClient(responses=responses)

    manager = StructuredOutputManager(client)
    results = await manager.extract_batch(texts, Person, use_cache=False)

    assert len(results) == 2
    assert results[0].name == "Tom"
    assert results[1].name == "Linda"


def test_structured_output_manager_clear_cache() -> None:
    """StructuredOutputManager.clear_cache should remove all cached entries."""
    client = DummyClient()
    manager = StructuredOutputManager(client)
    manager._cache["key"] = Person(name="X", age=1, occupation="Y")
    assert manager._cache
    manager.clear_cache()
    assert not manager._cache
