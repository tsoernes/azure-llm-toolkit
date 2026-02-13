"""Integration tests for vision model support with images."""

import os
from pathlib import Path

import pytest

from azure_llm_toolkit import AzureConfig, AzureLLMClient


@pytest.mark.asyncio
async def test_vision_message_with_url_image():
    """Test chat completion with a vision message containing an image URL."""
    # Use a public test image
    image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/640px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"

    config = AzureConfig()
    client = AzureLLMClient(config=config, enable_rate_limiting=False, enable_cache=False)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What do you see in this image? Describe it briefly."},
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        }
    ]

    result = await client.chat_completion(messages=messages, max_tokens=100)

    assert result is not None
    assert result.content is not None
    assert len(result.content) > 0
    assert result.usage.total_tokens > 0
    print(f"\nVision model response: {result.content}")
    print(f"Tokens used: {result.usage.total_tokens}")


@pytest.mark.asyncio
async def test_vision_message_with_base64_image():
    """Test chat completion with a vision message containing a base64 encoded image."""
    # Create a simple 1x1 pixel red PNG image in base64
    # This is a valid minimal PNG image
    base64_image = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="

    config = AzureConfig()
    client = AzureLLMClient(config=config, enable_rate_limiting=False, enable_cache=False)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What color is this image?"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
            ],
        }
    ]

    result = await client.chat_completion(messages=messages, max_tokens=50)

    assert result is not None
    assert result.content is not None
    assert len(result.content) > 0
    print(f"\nVision model response: {result.content}")


@pytest.mark.asyncio
async def test_vision_message_token_counting():
    """Test that token counting works correctly with vision messages."""
    image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/320px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"

    config = AzureConfig()
    client = AzureLLMClient(config=config, enable_rate_limiting=False, enable_cache=False)

    # Vision message with text and image
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image in one sentence."},
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        }
    ]

    # This should not raise an error
    token_count = client.count_message_tokens(messages)

    assert token_count > 0
    print(f"\nEstimated tokens for vision message: {token_count}")


@pytest.mark.asyncio
async def test_vision_message_with_multiple_images():
    """Test chat completion with multiple images in a single message."""
    image_url1 = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/320px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
    image_url2 = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/320px-Cat03.jpg"

    config = AzureConfig()
    client = AzureLLMClient(config=config, enable_rate_limiting=False, enable_cache=False)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Compare these two images briefly."},
                {"type": "image_url", "image_url": {"url": image_url1}},
                {"type": "image_url", "image_url": {"url": image_url2}},
            ],
        }
    ]

    result = await client.chat_completion(messages=messages, max_tokens=150)

    assert result is not None
    assert result.content is not None
    assert len(result.content) > 0
    print(f"\nVision model response: {result.content}")


@pytest.mark.asyncio
async def test_vision_message_mixed_with_text_messages():
    """Test chat completion with a mix of text-only and vision messages."""
    image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/320px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"

    config = AzureConfig()
    client = AzureLLMClient(config=config, enable_rate_limiting=False, enable_cache=False)

    messages = [
        {"role": "user", "content": "Hello, I'm going to show you an image."},
        {"role": "assistant", "content": "Sure, I'm ready to see it."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Here it is. What do you see?"},
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        },
    ]

    result = await client.chat_completion(messages=messages, max_tokens=100)

    assert result is not None
    assert result.content is not None
    assert len(result.content) > 0
    print(f"\nVision model response: {result.content}")


@pytest.mark.asyncio
async def test_vision_stream_with_image():
    """Test streaming chat completion with a vision message."""
    image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/320px-Cat03.jpg"

    config = AzureConfig()
    client = AzureLLMClient(config=config, enable_rate_limiting=False, enable_cache=False)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image in one sentence."},
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        }
    ]

    chunks = []
    async for chunk in client.chat_completion_stream(messages=messages, max_tokens=50):
        chunks.append(chunk)

    full_response = "".join(chunks)
    assert len(full_response) > 0
    print(f"\nStreamed vision response: {full_response}")


@pytest.mark.asyncio
async def test_extract_text_from_content_helper():
    """Test the _extract_text_from_content helper function directly."""
    from azure_llm_toolkit.client import _extract_text_from_content

    # Test with string content
    text = _extract_text_from_content("Hello, world!")
    assert text == "Hello, world!"

    # Test with vision message format
    content = [
        {"type": "text", "text": "First part"},
        {"type": "image_url", "image_url": {"url": "http://example.com/image.jpg"}},
        {"type": "text", "text": "Second part"},
    ]
    text = _extract_text_from_content(content)
    assert text == "First part Second part"

    # Test with empty list
    text = _extract_text_from_content([])
    assert text == ""

    # Test with no text parts
    content = [{"type": "image_url", "image_url": {"url": "http://example.com/image.jpg"}}]
    text = _extract_text_from_content(content)
    assert text == ""

    # Test with unexpected format
    text = _extract_text_from_content(123)
    assert text == ""

    print("\nAll _extract_text_from_content tests passed!")


if __name__ == "__main__":
    import asyncio

    print("Running vision integration tests...\n")

    # Run the tests
    asyncio.run(test_vision_message_with_url_image())
    asyncio.run(test_vision_message_with_base64_image())
    asyncio.run(test_vision_message_token_counting())
    asyncio.run(test_vision_message_with_multiple_images())
    asyncio.run(test_vision_message_mixed_with_text_messages())
    asyncio.run(test_vision_stream_with_image())
    asyncio.run(test_extract_text_from_content_helper())

    print("\n✅ All vision integration tests completed!")
