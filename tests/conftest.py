"""Pytest configuration and shared fixtures."""

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

from azure_llm_toolkit import AzureConfig, AzureLLMClient, CacheManager, PolarsBatchEmbedder

# Load environment variables from .env file
load_dotenv()


@pytest.fixture
def config():
    """Create AzureConfig from environment."""
    return AzureConfig()


@pytest.fixture
def client(config):
    """Create AzureLLMClient with caching disabled for tests."""
    return AzureLLMClient(config=config, enable_cache=False)


@pytest.fixture
def client_with_cache(config, tmp_path):
    """Create AzureLLMClient with caching enabled in temporary directory."""
    cache_manager = CacheManager(cache_dir=tmp_path / "cache")
    return AzureLLMClient(config=config, cache_manager=cache_manager, enable_cache=True)


@pytest.fixture
def batch_embedder(config):
    """Create PolarsBatchEmbedder."""
    return PolarsBatchEmbedder(config=config)


@pytest.fixture
def sample_texts():
    """Sample texts for testing."""
    return [
        "Azure OpenAI provides powerful language models.",
        "Machine learning is transforming industries.",
        "Python is great for AI development.",
    ]


@pytest.fixture
def sample_messages():
    """Sample chat messages for testing."""
    return [{"role": "user", "content": "What is 2+2?"}]


@pytest.fixture
def check_credentials():
    """Check that required credentials are set before running tests."""
    required_vars = ["AZURE_OPENAI_API_KEY", "AZURE_ENDPOINT"]
    missing = [
        var for var in required_vars if not os.getenv(var) and not os.getenv(var.replace("AZURE_OPENAI_", "AZURE_"))
    ]

    if missing:
        pytest.skip(f"Missing required environment variables: {', '.join(missing)}")


def pytest_collection_modifyitems(config, items):
    """Mark integration tests that require credentials."""
    skip_no_creds = pytest.mark.skip(reason="Missing Azure credentials")

    # Tests that don't require credentials (unit tests)
    unit_test_patterns = [
        "test_gpt5_parameter_conversion",
        "test_extract_text_from_content_helper",
        "test_validation",
    ]

    required_vars = ["AZURE_OPENAI_API_KEY", "AZURE_ENDPOINT"]
    missing = [
        var for var in required_vars if not os.getenv(var) and not os.getenv(var.replace("AZURE_OPENAI_", "AZURE_"))
    ]

    if missing:
        for item in items:
            # Skip only if it's not a unit test (check both file path and node name)
            is_unit_test = any(pattern in str(item.fspath) or pattern in item.nodeid for pattern in unit_test_patterns)
            if not is_unit_test:
                item.add_marker(skip_no_creds)
