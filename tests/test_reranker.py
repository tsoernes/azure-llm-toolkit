"""
Unit tests for the logprob-based reranker module.

Tests cover:
- Configuration and initialization
- Single document scoring
- Batch reranking
- Error handling and edge cases
- Integration with AzureLLMClient
- Bin probability calculations
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from openai import AsyncAzureOpenAI, BadRequestError, RateLimitError

from azure_llm_toolkit import AzureConfig, AzureLLMClient
from azure_llm_toolkit.rate_limiter import RateLimiter
from azure_llm_toolkit.reranker import (
    LogprobReranker,
    RerankerConfig,
    RerankResult,
    create_reranker,
    _softmax_logprobs,
    _expected_from_bins,
    _build_messages,
)


# -----------------------------
# Utility Function Tests
# -----------------------------


def test_softmax_logprobs_empty():
    """Test softmax with empty input."""
    result = _softmax_logprobs({})
    assert result == {}


def test_softmax_logprobs_single():
    """Test softmax with single value."""
    result = _softmax_logprobs({"0": -2.0})
    assert result == {"0": 1.0}


def test_softmax_logprobs_multiple():
    """Test softmax with multiple values."""
    result = _softmax_logprobs({"0": -2.0, "5": -1.0, "10": -3.0})

    # Check probabilities sum to 1.0
    assert abs(sum(result.values()) - 1.0) < 1e-6

    # Check relative ordering (higher logprob = higher probability)
    assert result["5"] > result["0"] > result["10"]


def test_expected_from_bins_empty():
    """Test expected value with empty bins."""
    result = _expected_from_bins({}, [])
    assert result == 0.0


def test_expected_from_bins_single():
    """Test expected value with single bin."""
    result = _expected_from_bins({"0": 1.0}, ["0"])
    assert result == 1.0


def test_expected_from_bins_uniform():
    """Test expected value with uniform distribution."""
    bins = ["0", "1", "2", "3", "4"]
    probs = {b: 0.2 for b in bins}
    result = _expected_from_bins(probs, bins)

    # Uniform over 5 bins should give expected value of 0.5
    assert abs(result - 0.5) < 1e-6


def test_expected_from_bins_weighted():
    """Test expected value with weighted distribution."""
    bins = ["0", "5", "10"]
    # All probability on middle bin
    probs = {"0": 0.0, "5": 1.0, "10": 0.0}
    result = _expected_from_bins(probs, bins)

    # Should map to 0.5 (middle of 0.0 to 1.0)
    assert abs(result - 0.5) < 1e-6


def test_build_messages_structure():
    """Test message structure for prompts."""
    query = "What is AI?"
    document = "AI is artificial intelligence."
    bins = ["0", "1", "2"]

    messages = _build_messages(query, document, bins)

    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert query in messages[1]["content"]
    assert document in messages[1]["content"]
    assert "0, 1, 2" in messages[1]["content"]


# -----------------------------
# Configuration Tests
# -----------------------------


def test_reranker_config_defaults():
    """Test default configuration values."""
    config = RerankerConfig()

    assert config.model == "gpt-4o-east-US"
    assert config.bins == [str(i) for i in range(11)]
    assert config.top_logprobs == 5
    assert config.logprob_floor == -16.0
    assert config.temperature == 0.2
    assert config.max_tokens == 1
    assert config.timeout == 30.0
    assert config.rpm_limit == 2700
    assert config.tpm_limit == 450000


def test_reranker_config_custom():
    """Test custom configuration."""
    config = RerankerConfig(
        model="gpt-4o",
        bins=["0", "1", "2"],
        top_logprobs=3,
        temperature=0.1,
    )

    assert config.model == "gpt-4o"
    assert config.bins == ["0", "1", "2"]
    assert config.top_logprobs == 3
    assert config.temperature == 0.1


def test_reranker_config_auto_bins():
    """Test automatic bin initialization."""
    config = RerankerConfig(bins=None)
    assert len(config.bins) == 11

    config = RerankerConfig(bins=[])
    assert len(config.bins) == 11


# -----------------------------
# Reranker Initialization Tests
# -----------------------------


def test_reranker_init_with_azure_client():
    """Test initialization with AzureLLMClient."""
    mock_azure_client = AsyncMock(spec=AsyncAzureOpenAI)
    mock_llm_client = MagicMock()
    mock_llm_client.client = mock_azure_client
    mock_llm_client.config = MagicMock(chat_deployment="gpt-4o")

    reranker = LogprobReranker(client=mock_llm_client)

    assert reranker._openai_client == mock_azure_client
    assert reranker.config.model == "gpt-4o-east-US"
    assert reranker.rate_limiter is not None
    assert reranker.rate_limiter.rpm_limit == 2700
    assert reranker.rate_limiter.tpm_limit == 450000


def test_reranker_init_with_openai_client():
    """Test initialization with direct AsyncAzureOpenAI client."""
    mock_client = AsyncMock(spec=AsyncAzureOpenAI)
    config = RerankerConfig(model="gpt-4o")

    reranker = LogprobReranker(client=mock_client, config=config)

    assert reranker._openai_client == mock_client
    assert reranker.config.model == "gpt-4o"


def test_reranker_init_invalid_client():
    """Test initialization with invalid client type."""
    with pytest.raises(TypeError):
        LogprobReranker(client="not a client")


def test_reranker_init_with_default_model():
    """Test initialization uses default model."""
    mock_client = AsyncMock(spec=AsyncAzureOpenAI)

    reranker = LogprobReranker(client=mock_client)
    assert reranker.config.model == "gpt-4o-east-US"
    assert reranker.rate_limiter is not None


# -----------------------------
# Scoring Tests
# -----------------------------


@pytest.mark.asyncio
async def test_score_success():
    """Test successful document scoring."""
    # Create mock response with logprobs
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_logprobs = MagicMock()

    # Mock logprob content
    mock_content_item = MagicMock()
    mock_candidate_1 = MagicMock()
    mock_candidate_1.token = "5"
    mock_candidate_1.logprob = -0.5

    mock_candidate_2 = MagicMock()
    mock_candidate_2.token = "6"
    mock_candidate_2.logprob = -1.0

    mock_content_item.top_logprobs = [mock_candidate_1, mock_candidate_2]
    mock_logprobs.content = [mock_content_item]
    mock_choice.logprobs = mock_logprobs

    mock_response.choices = [mock_choice]

    # Create mock client
    mock_openai_client = AsyncMock(spec=AsyncAzureOpenAI)
    mock_openai_client.chat.completions.create = AsyncMock(return_value=mock_response)

    config = RerankerConfig(model="gpt-4o")
    reranker = LogprobReranker(client=mock_openai_client, config=config)

    score = await reranker.score("test query", "test document")

    # Score should be between 0 and 1
    assert 0.0 <= score <= 1.0
    assert isinstance(score, float)


@pytest.mark.asyncio
async def test_score_with_bin_probs():
    """Test scoring with bin probabilities returned."""
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_logprobs = MagicMock()

    mock_content_item = MagicMock()
    mock_candidate = MagicMock()
    mock_candidate.token = "8"
    mock_candidate.logprob = -0.3

    mock_content_item.top_logprobs = [mock_candidate]
    mock_logprobs.content = [mock_content_item]
    mock_choice.logprobs = mock_logprobs
    mock_response.choices = [mock_choice]

    mock_openai_client = AsyncMock(spec=AsyncAzureOpenAI)
    mock_openai_client.chat.completions.create = AsyncMock(return_value=mock_response)

    config = RerankerConfig(model="gpt-4o")
    reranker = LogprobReranker(client=mock_openai_client, config=config)

    score, bin_probs = await reranker.score("query", "doc", include_bin_probs=True)

    assert isinstance(score, float)
    assert isinstance(bin_probs, dict)
    assert 0.0 <= score <= 1.0
    # Should have probabilities for all bins
    assert len(bin_probs) == len(config.bins)


@pytest.mark.asyncio
async def test_score_api_error():
    """Test handling of API errors during scoring."""
    mock_openai_client = AsyncMock(spec=AsyncAzureOpenAI)

    # Create a proper mock response for the exception
    mock_response = MagicMock()
    mock_request = MagicMock()
    mock_response.request = mock_request

    mock_openai_client.chat.completions.create = AsyncMock(
        side_effect=RateLimitError("Rate limit exceeded", response=mock_response, body=None)
    )

    config = RerankerConfig(model="gpt-4o")
    reranker = LogprobReranker(client=mock_openai_client, config=config)

    score = await reranker.score("query", "doc")

    # Should return 0.0 on error
    assert score == 0.0


@pytest.mark.asyncio
async def test_score_no_logprobs():
    """Test handling when logprobs are not available."""
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_choice.logprobs = None
    mock_response.choices = [mock_choice]

    mock_openai_client = AsyncMock(spec=AsyncAzureOpenAI)
    mock_openai_client.chat.completions.create = AsyncMock(return_value=mock_response)

    config = RerankerConfig(model="gpt-4o")
    reranker = LogprobReranker(client=mock_openai_client, config=config)

    score = await reranker.score("query", "doc")

    # Should still return a score (using floor values)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


@pytest.mark.asyncio
async def test_score_empty_response():
    """Test handling of empty response."""
    mock_response = MagicMock()
    mock_response.choices = []

    mock_openai_client = AsyncMock(spec=AsyncAzureOpenAI)
    mock_openai_client.chat.completions.create = AsyncMock(return_value=mock_response)

    config = RerankerConfig(model="gpt-4o")
    reranker = LogprobReranker(client=mock_openai_client, config=config)

    score = await reranker.score("query", "doc")

    assert score == 0.0


# -----------------------------
# Reranking Tests
# -----------------------------


@pytest.mark.asyncio
async def test_rerank_basic():
    """Test basic reranking functionality."""
    mock_openai_client = AsyncMock(spec=AsyncAzureOpenAI)

    # Track call order to assign consistent scores
    call_order = []

    # Mock different scores for different documents
    async def mock_create(*args, **kwargs):
        # Simulate varying relevance scores
        content = kwargs.get("messages", [{}])[1].get("content", "")

        # Track which document this is
        call_order.append(content)

        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_logprobs = MagicMock()
        mock_content_item = MagicMock()
        mock_candidate = MagicMock()

        # Assign different scores based on document content
        if "Very relevant" in content:
            mock_candidate.token = "9"
            mock_candidate.logprob = -0.2
        elif "Somewhat" in content:
            mock_candidate.token = "5"
            mock_candidate.logprob = -0.6
        else:
            mock_candidate.token = "2"
            mock_candidate.logprob = -1.5

        mock_content_item.top_logprobs = [mock_candidate]
        mock_logprobs.content = [mock_content_item]
        mock_choice.logprobs = mock_logprobs
        mock_response.choices = [mock_choice]

        return mock_response

    mock_openai_client.chat.completions.create = AsyncMock(side_effect=mock_create)

    config = RerankerConfig(model="gpt-4o")
    reranker = LogprobReranker(client=mock_openai_client, config=config)

    documents = [
        "Not relevant document",
        "Very relevant document for the query",
        "Somewhat relevant document",
    ]

    results = await reranker.rerank("test query", documents)

    assert len(results) == 3
    assert all(isinstance(r, RerankResult) for r in results)

    # Results should be sorted by score (descending)
    for i in range(len(results) - 1):
        assert results[i].score >= results[i + 1].score

    # Check that highest scoring document is ranked first
    # The "Very relevant" document should have the highest score (token "9")
    assert results[0].score > results[1].score > results[2].score


@pytest.mark.asyncio
async def test_rerank_empty_documents():
    """Test reranking with empty document list."""
    mock_openai_client = AsyncMock(spec=AsyncAzureOpenAI)
    config = RerankerConfig(model="gpt-4o")
    reranker = LogprobReranker(client=mock_openai_client, config=config)

    results = await reranker.rerank("query", [])

    assert results == []


@pytest.mark.asyncio
async def test_rerank_top_k():
    """Test reranking with top_k limit."""
    mock_openai_client = AsyncMock(spec=AsyncAzureOpenAI)

    # Simple mock that returns valid responses
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_logprobs = MagicMock()
    mock_content_item = MagicMock()
    mock_candidate = MagicMock()
    mock_candidate.token = "5"
    mock_candidate.logprob = -0.5
    mock_content_item.top_logprobs = [mock_candidate]
    mock_logprobs.content = [mock_content_item]
    mock_choice.logprobs = mock_logprobs
    mock_response.choices = [mock_choice]

    mock_openai_client.chat.completions.create = AsyncMock(return_value=mock_response)

    config = RerankerConfig(model="gpt-4o")
    reranker = LogprobReranker(client=mock_openai_client, config=config)

    documents = ["doc1", "doc2", "doc3", "doc4", "doc5"]

    results = await reranker.rerank("query", documents, top_k=3)

    assert len(results) == 3


@pytest.mark.asyncio
async def test_rerank_with_exceptions():
    """Test reranking when some documents fail to score."""
    mock_openai_client = AsyncMock(spec=AsyncAzureOpenAI)

    call_count = 0

    async def mock_create(*args, **kwargs):
        nonlocal call_count
        call_count += 1

        if call_count == 2:
            # Fail second document
            raise BadRequestError("Bad request", response=None, body=None)

        # Success for others
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_logprobs = MagicMock()
        mock_content_item = MagicMock()
        mock_candidate = MagicMock()
        mock_candidate.token = "7"
        mock_candidate.logprob = -0.4
        mock_content_item.top_logprobs = [mock_candidate]
        mock_logprobs.content = [mock_content_item]
        mock_choice.logprobs = mock_logprobs
        mock_response.choices = [mock_choice]
        return mock_response

    mock_openai_client.chat.completions.create = AsyncMock(side_effect=mock_create)

    config = RerankerConfig(model="gpt-4o")
    reranker = LogprobReranker(client=mock_openai_client, config=config)

    documents = ["doc1", "doc2", "doc3"]
    results = await reranker.rerank("query", documents)

    # Should still return all documents, failed one with score 0.0
    assert len(results) == 3
    assert results[-1].score == 0.0  # Failed document should be ranked last


@pytest.mark.asyncio
async def test_rerank_with_bin_probs():
    """Test reranking with bin probability distributions."""
    mock_openai_client = AsyncMock(spec=AsyncAzureOpenAI)

    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_logprobs = MagicMock()
    mock_content_item = MagicMock()
    mock_candidate = MagicMock()
    mock_candidate.token = "8"
    mock_candidate.logprob = -0.3
    mock_content_item.top_logprobs = [mock_candidate]
    mock_logprobs.content = [mock_content_item]
    mock_choice.logprobs = mock_logprobs
    mock_response.choices = [mock_choice]

    mock_openai_client.chat.completions.create = AsyncMock(return_value=mock_response)

    config = RerankerConfig(model="gpt-4o")
    reranker = LogprobReranker(client=mock_openai_client, config=config)

    documents = ["doc1", "doc2"]
    results = await reranker.rerank("query", documents, include_bin_probs=True)

    assert len(results) == 2
    for result in results:
        assert result.bin_probabilities is not None
        assert isinstance(result.bin_probabilities, dict)
        assert len(result.bin_probabilities) > 0


# -----------------------------
# RerankResult Tests
# -----------------------------


def test_rerank_result_creation():
    """Test RerankResult dataclass creation."""
    result = RerankResult(
        index=0,
        document="test document",
        score=0.85,
        bin_probabilities={"5": 0.6, "6": 0.4},
    )

    assert result.index == 0
    assert result.document == "test document"
    assert result.score == 0.85
    assert result.bin_probabilities == {"5": 0.6, "6": 0.4}


def test_rerank_result_to_dict():
    """Test RerankResult.to_dict() method."""
    result = RerankResult(
        index=1,
        document="doc",
        score=0.75,
        bin_probabilities={"3": 0.8, "4": 0.2},
    )

    result_dict = result.to_dict()

    assert result_dict["index"] == 1
    assert result_dict["document"] == "doc"
    assert result_dict["score"] == 0.75
    assert result_dict["bin_probabilities"] == {"3": 0.8, "4": 0.2}


def test_rerank_result_to_dict_no_bin_probs():
    """Test to_dict without bin probabilities."""
    result = RerankResult(
        index=0,
        document="doc",
        score=0.5,
    )

    result_dict = result.to_dict()

    assert "bin_probabilities" not in result_dict


# -----------------------------
# Convenience Function Tests
# -----------------------------


def test_create_reranker():
    """Test create_reranker convenience function."""
    mock_openai_client = AsyncMock(spec=AsyncAzureOpenAI)

    reranker = create_reranker(
        client=mock_openai_client,
        model="gpt-4o",
        temperature=0.1,
    )

    assert isinstance(reranker, LogprobReranker)
    assert reranker.config.model == "gpt-4o"
    assert reranker.config.temperature == 0.1


def test_create_reranker_custom_bins():
    """Test create_reranker with custom bins."""
    mock_openai_client = AsyncMock(spec=AsyncAzureOpenAI)

    custom_bins = ["low", "medium", "high"]
    reranker = create_reranker(
        client=mock_openai_client,
        model="gpt-4o",
        bins=custom_bins,
    )

    assert reranker.config.bins == custom_bins


# -----------------------------
# Integration Tests
# -----------------------------


@pytest.mark.asyncio
async def test_integration_with_azure_llm_client():
    """Test integration with AzureLLMClient."""
    # Create a mock AzureLLMClient
    mock_azure_client = AsyncMock(spec=AsyncAzureOpenAI)
    mock_llm_client = MagicMock(spec=AzureLLMClient)
    mock_llm_client.client = mock_azure_client
    mock_llm_client.config = MagicMock(chat_deployment="gpt-4o")

    # Create reranker
    reranker = LogprobReranker(client=mock_llm_client)

    # Verify client extraction worked
    assert reranker._openai_client == mock_azure_client
    assert reranker.config.model == "gpt-4o-east-US"
    assert reranker.rate_limiter is not None


def test_module_exports():
    """Test that all expected symbols are exported."""
    from azure_llm_toolkit.reranker import __all__

    expected_exports = [
        "LogprobReranker",
        "RerankerConfig",
        "RerankResult",
        "create_reranker",
    ]

    assert set(__all__) == set(expected_exports)


# -----------------------------
# Rate Limiter Tests
# -----------------------------


@pytest.mark.asyncio
async def test_reranker_with_custom_rate_limiter():
    """Test reranker with custom rate limiter."""
    mock_openai_client = AsyncMock(spec=AsyncAzureOpenAI)
    config = RerankerConfig(model="gpt-4o")

    # Create custom rate limiter with lower limits
    custom_limiter = RateLimiter(rpm_limit=100, tpm_limit=10000)

    reranker = LogprobReranker(
        client=mock_openai_client,
        config=config,
        rate_limiter=custom_limiter,
    )

    assert reranker.rate_limiter == custom_limiter
    assert reranker.rate_limiter.rpm_limit == 100
    assert reranker.rate_limiter.tpm_limit == 10000


@pytest.mark.asyncio
async def test_reranker_rate_limiting_in_scoring():
    """Test that rate limiter is invoked during scoring."""
    mock_openai_client = AsyncMock(spec=AsyncAzureOpenAI)

    # Create mock response
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_logprobs = MagicMock()
    mock_content_item = MagicMock()
    mock_candidate = MagicMock()
    mock_candidate.token = "7"
    mock_candidate.logprob = -0.4
    mock_content_item.top_logprobs = [mock_candidate]
    mock_logprobs.content = [mock_content_item]
    mock_choice.logprobs = mock_logprobs
    mock_response.choices = [mock_choice]

    # Add usage info
    mock_usage = MagicMock()
    mock_usage.total_tokens = 100
    mock_response.usage = mock_usage

    mock_openai_client.chat.completions.create = AsyncMock(return_value=mock_response)

    config = RerankerConfig(model="gpt-4o")
    custom_limiter = RateLimiter(rpm_limit=1000, tpm_limit=100000)

    reranker = LogprobReranker(
        client=mock_openai_client,
        config=config,
        rate_limiter=custom_limiter,
    )

    # Get initial stats
    initial_stats = custom_limiter.get_stats()

    # Score a document
    await reranker.score("test query", "test document")

    # Check that rate limiter was used
    final_stats = custom_limiter.get_stats()
    assert final_stats["total_requests"] == initial_stats["total_requests"] + 1
    assert final_stats["total_tokens"] > initial_stats["total_tokens"]


@pytest.mark.asyncio
async def test_reranker_parallel_scoring_with_rate_limiting():
    """Test parallel scoring respects rate limits."""
    mock_openai_client = AsyncMock(spec=AsyncAzureOpenAI)

    # Create mock response
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_logprobs = MagicMock()
    mock_content_item = MagicMock()
    mock_candidate = MagicMock()
    mock_candidate.token = "5"
    mock_candidate.logprob = -0.5
    mock_content_item.top_logprobs = [mock_candidate]
    mock_logprobs.content = [mock_content_item]
    mock_choice.logprobs = mock_logprobs
    mock_response.choices = [mock_choice]

    mock_usage = MagicMock()
    mock_usage.total_tokens = 50
    mock_response.usage = mock_usage

    mock_openai_client.chat.completions.create = AsyncMock(return_value=mock_response)

    config = RerankerConfig(model="gpt-4o")
    # Use high limits to ensure no blocking in test
    custom_limiter = RateLimiter(rpm_limit=10000, tpm_limit=1000000)

    reranker = LogprobReranker(
        client=mock_openai_client,
        config=config,
        rate_limiter=custom_limiter,
    )

    documents = ["doc1", "doc2", "doc3", "doc4", "doc5"]

    # Rerank (scores in parallel)
    results = await reranker.rerank("query", documents)

    # Check all documents were processed
    assert len(results) == 5

    # Check rate limiter tracked all requests
    stats = custom_limiter.get_stats()
    assert stats["total_requests"] == 5


@pytest.mark.asyncio
async def test_create_reranker_with_custom_limits():
    """Test create_reranker with custom rate limits."""
    mock_openai_client = AsyncMock(spec=AsyncAzureOpenAI)

    reranker = create_reranker(
        client=mock_openai_client,
        model="gpt-4o",
        rpm_limit=5000,
        tpm_limit=600000,
    )

    assert reranker.config.rpm_limit == 5000
    assert reranker.config.tpm_limit == 600000
    assert reranker.rate_limiter.rpm_limit == 5000
    assert reranker.rate_limiter.tpm_limit == 600000


@pytest.mark.asyncio
async def test_rate_limiter_defaults():
    """Test that default rate limits are set correctly."""
    mock_openai_client = AsyncMock(spec=AsyncAzureOpenAI)
    config = RerankerConfig(model="gpt-4o")

    reranker = LogprobReranker(client=mock_openai_client, config=config)

    # Check defaults: 2700 RPM, 450k TPM
    assert reranker.rate_limiter.rpm_limit == 2700
    assert reranker.rate_limiter.tpm_limit == 450000
