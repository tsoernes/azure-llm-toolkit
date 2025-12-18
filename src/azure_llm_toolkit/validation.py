"""
Response validation and parsing for structured outputs.

This module provides functionality for getting validated, structured outputs
from LLMs using Pydantic models, with automatic retry on parse failures.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Type, TypeVar

from pydantic import BaseModel, ValidationError

from .client import AzureLLMClient
from .types import ChatCompletionResult

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class StructuredOutputError(Exception):
    """Exception raised when structured output parsing fails."""

    pass


class ValidationRetryExhaustedError(StructuredOutputError):
    """Exception raised when all validation retries are exhausted."""

    pass


def generate_json_schema(model: Type[BaseModel]) -> dict[str, Any]:
    """
    Generate JSON schema from Pydantic model.

    Args:
        model: Pydantic model class

    Returns:
        JSON schema dictionary
    """
    return model.model_json_schema()


def create_extraction_prompt(
    text: str,
    schema: dict[str, Any],
    instructions: str | None = None,
) -> str:
    """
    Create a prompt for structured data extraction.

    Args:
        text: Text to extract data from
        schema: JSON schema for the expected output
        instructions: Optional additional instructions

    Returns:
        Formatted prompt string
    """
    base_prompt = f"""Extract structured information from the following text and return it as valid JSON.

The JSON must conform to this schema:
{json.dumps(schema, indent=2)}

Text to extract from:
{text}

Instructions:
- Return ONLY valid JSON, no additional text
- Ensure all required fields are present
- Follow the exact schema structure
"""

    if instructions:
        base_prompt += f"\nAdditional instructions:\n{instructions}\n"

    base_prompt += "\nJSON output:"

    return base_prompt


def parse_json_response(response: str) -> dict[str, Any]:
    """
    Parse JSON from LLM response, handling common issues.

    Args:
        response: Raw response text from LLM

    Returns:
        Parsed JSON dictionary

    Raises:
        json.JSONDecodeError: If JSON parsing fails
    """
    # Try to extract JSON from response if it contains extra text
    response = response.strip()

    # Remove markdown code blocks if present
    if response.startswith("```"):
        lines = response.split("\n")
        # Remove first line (```json or ```)
        lines = lines[1:]
        # Remove last line if it's ```
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        response = "\n".join(lines).strip()

    # Try to find JSON object in response
    start_idx = response.find("{")
    end_idx = response.rfind("}")

    if start_idx != -1 and end_idx != -1:
        json_str = response[start_idx : end_idx + 1]
        return json.loads(json_str)

    # If no braces found, try parsing the whole response
    return json.loads(response)


async def chat_completion_structured(
    client: AzureLLMClient,
    messages: list[dict[str, str]],
    response_model: Type[T],
    max_retries: int = 3,
    model: str | None = None,
    temperature: float = 0.0,
    **kwargs: Any,
) -> T:
    """
    Get structured output from chat completion with validation.

    This function ensures the LLM response conforms to the specified Pydantic model,
    with automatic retries on validation failures.

    Args:
        client: AzureLLMClient instance
        messages: List of message dictionaries
        response_model: Pydantic model class for response validation
        max_retries: Maximum number of retry attempts on validation failure
        model: Optional model override
        temperature: Temperature (default 0.0 for deterministic output)
        **kwargs: Additional arguments passed to chat_completion

    Returns:
        Validated instance of response_model

    Raises:
        ValidationRetryExhaustedError: If all retries fail
        StructuredOutputError: For other parsing errors

    Example:
        >>> from pydantic import BaseModel
        >>>
        >>> class Person(BaseModel):
        ...     name: str
        ...     age: int
        ...     occupation: str
        >>>
        >>> messages = [{"role": "user", "content": "Extract: John is 30, works as engineer"}]
        >>> person = await chat_completion_structured(
        ...     client=client,
        ...     messages=messages,
        ...     response_model=Person,
        ... )
        >>> print(person.name)
        'John'
    """
    schema = generate_json_schema(response_model)

    # Add JSON format instruction to the last user message
    enhanced_messages = messages.copy()
    if enhanced_messages:
        last_message = enhanced_messages[-1]
        if last_message.get("role") == "user":
            content = last_message["content"]
            schema_instruction = f"\n\nRespond with JSON matching this schema:\n{json.dumps(schema, indent=2)}"
            last_message["content"] = content + schema_instruction

    validation_errors: list[str] = []

    for attempt in range(max_retries):
        try:
            # Get response from LLM
            response = await client.chat_completion(
                messages=enhanced_messages,
                model=model,
                temperature=temperature,
                response_format={"type": "json_object"},  # Request JSON output
                **kwargs,
            )

            # Parse JSON from response
            try:
                parsed_data = parse_json_response(response.content)
            except json.JSONDecodeError as e:
                error_msg = f"JSON parsing failed: {e}"
                logger.warning(f"Attempt {attempt + 1}/{max_retries}: {error_msg}")
                validation_errors.append(error_msg)

                # Add error feedback to messages for retry
                enhanced_messages.append({"role": "assistant", "content": response.content})
                enhanced_messages.append(
                    {
                        "role": "user",
                        "content": f"The response was not valid JSON. Error: {error_msg}\nPlease provide valid JSON output.",
                    }
                )
                continue

            # Validate against Pydantic model
            try:
                validated = response_model.model_validate(parsed_data)
                logger.info(f"Successfully validated structured output on attempt {attempt + 1}")
                return validated

            except ValidationError as e:
                error_msg = f"Validation failed: {e}"
                logger.warning(f"Attempt {attempt + 1}/{max_retries}: {error_msg}")
                validation_errors.append(error_msg)

                # Add validation error feedback for retry
                enhanced_messages.append({"role": "assistant", "content": response.content})
                enhanced_messages.append(
                    {
                        "role": "user",
                        "content": f"The JSON structure was incorrect. Validation errors:\n{e}\n\nPlease provide valid JSON matching the schema.",
                    }
                )
                continue

        except Exception as e:
            error_msg = f"Unexpected error: {e}"
            logger.error(f"Attempt {attempt + 1}/{max_retries}: {error_msg}")
            validation_errors.append(error_msg)

            if attempt < max_retries - 1:
                # Add generic error feedback for retry
                enhanced_messages.append(
                    {
                        "role": "user",
                        "content": "There was an error processing the response. Please try again with valid JSON.",
                    }
                )
            continue

    # All retries exhausted
    error_summary = "\n".join(f"- Attempt {i + 1}: {err}" for i, err in enumerate(validation_errors))
    raise ValidationRetryExhaustedError(
        f"Failed to get valid structured output after {max_retries} attempts:\n{error_summary}"
    )


async def extract_structured_data(
    client: AzureLLMClient,
    text: str,
    response_model: Type[T],
    instructions: str | None = None,
    max_retries: int = 3,
    model: str | None = None,
    **kwargs: Any,
) -> T:
    """
    Extract structured data from text using a Pydantic model.

    Convenience function that creates the extraction prompt automatically.

    Args:
        client: AzureLLMClient instance
        text: Text to extract data from
        response_model: Pydantic model class for response validation
        instructions: Optional additional extraction instructions
        max_retries: Maximum retry attempts on validation failure
        model: Optional model override
        **kwargs: Additional arguments passed to chat_completion

    Returns:
        Validated instance of response_model

    Example:
        >>> class Book(BaseModel):
        ...     title: str
        ...     author: str
        ...     year: int
        >>>
        >>> text = "The Great Gatsby by F. Scott Fitzgerald, published in 1925"
        >>> book = await extract_structured_data(
        ...     client=client,
        ...     text=text,
        ...     response_model=Book,
        ... )
        >>> print(book.title)
        'The Great Gatsby'
    """
    schema = generate_json_schema(response_model)
    prompt = create_extraction_prompt(text, schema, instructions)

    messages = [{"role": "user", "content": prompt}]

    return await chat_completion_structured(
        client=client,
        messages=messages,
        response_model=response_model,
        max_retries=max_retries,
        model=model,
        **kwargs,
    )


class StructuredOutputManager:
    """
    Manager for structured outputs with caching and batch processing.

    Example:
        >>> manager = StructuredOutputManager(client)
        >>>
        >>> # Single extraction
        >>> result = await manager.extract(text, Person)
        >>>
        >>> # Batch extraction
        >>> results = await manager.extract_batch(texts, Person)
    """

    def __init__(self, client: AzureLLMClient):
        """
        Initialize manager.

        Args:
            client: AzureLLMClient instance
        """
        self.client = client
        self._cache: dict[str, BaseModel] = {}

    def _get_cache_key(self, text: str, model_name: str) -> str:
        """Generate cache key for text and model."""
        import hashlib

        text_hash = hashlib.md5(text.encode()).hexdigest()
        return f"{model_name}:{text_hash}"

    async def extract(
        self,
        text: str,
        response_model: Type[T],
        use_cache: bool = True,
        **kwargs: Any,
    ) -> T:
        """
        Extract structured data with optional caching.

        Args:
            text: Text to extract from
            response_model: Pydantic model class
            use_cache: Whether to use cache
            **kwargs: Additional arguments

        Returns:
            Validated model instance
        """
        cache_key = self._get_cache_key(text, response_model.__name__)

        # Check cache
        if use_cache and cache_key in self._cache:
            cached = self._cache[cache_key]
            if isinstance(cached, response_model):
                logger.debug(f"Cache hit for {cache_key}")
                return cached

        # Extract data
        result = await extract_structured_data(
            client=self.client,
            text=text,
            response_model=response_model,
            **kwargs,
        )

        # Cache result
        if use_cache:
            self._cache[cache_key] = result

        return result

    async def extract_batch(
        self,
        texts: list[str],
        response_model: Type[T],
        use_cache: bool = True,
        **kwargs: Any,
    ) -> list[T]:
        """
        Extract structured data from multiple texts in parallel.

        Args:
            texts: List of texts to extract from
            response_model: Pydantic model class
            use_cache: Whether to use cache
            **kwargs: Additional arguments

        Returns:
            List of validated model instances
        """
        import asyncio

        tasks = [self.extract(text, response_model, use_cache=use_cache, **kwargs) for text in texts]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle exceptions
        validated_results: list[T] = []
        from typing import cast

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Failed to extract from text {i}: {result}")
                raise result
            # Cast the result to the expected response_model type for the benefit of static type checkers.
            validated_results.append(cast(T, result))

        return validated_results

    def clear_cache(self) -> None:
        """Clear the extraction cache."""
        self._cache.clear()
        logger.info("Cleared extraction cache")


__all__ = [
    "StructuredOutputError",
    "ValidationRetryExhaustedError",
    "generate_json_schema",
    "create_extraction_prompt",
    "parse_json_response",
    "chat_completion_structured",
    "extract_structured_data",
    "StructuredOutputManager",
]
