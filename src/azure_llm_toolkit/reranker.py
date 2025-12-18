"""
Logprob-based reranker for Azure OpenAI chat completions.

This module provides a lightweight, zero-shot reranking component that uses token
log probabilities to compute calibrated relevance scores. It integrates seamlessly
with the azure-llm-toolkit's client infrastructure.

Overview
--------
- Uses a discrete set of "bins" as target output tokens (default: ["0","1",...,"10"])
  mapping evenly onto [0.0..1.0].
- Requests token-level logprobs from the chat completions API.
- Collects per-bin logprobs from the first generated token's top candidates.
- Applies softmax to obtain a probability distribution over bins.
- Computes the expected relevance score as a probability-weighted average of bin values.

Why logprob reranking?
----------------------
- Provides calibrated, uncertainty-aware scoring.
- Avoids free-text parsing and reduces prompt variability.
- Zero-shot and model-agnostic (as long as the chat endpoint exposes logprobs).

Defaults
--------
- Model: "gpt-4o-east-US"
- Bins: ["0","1","2","3","4","5","6","7","8","9","10"] mapping to 0.0..1.0
- top_logprobs: 5
- logprob_floor: -16.0 (used when a bin token is not present in top_logprobs)
- temperature: 0.2
- max_tokens: 1 (only the first token is needed for the bin decision)

Integration Notes
-----------------
- Works with AzureLLMClient or any AsyncAzureOpenAI client.
- Integrates with cost tracking and rate limiting when using AzureLLMClient.
- Can be disabled for deployments that don't support logprobs.

Example Usage
-------------
    from azure_llm_toolkit import AzureConfig, AzureLLMClient
    from azure_llm_toolkit.reranker import LogprobReranker

    config = AzureConfig()
    client = AzureLLMClient(config=config)

    reranker = LogprobReranker(client=client)

    query = "What is machine learning?"
    documents = [
        "Machine learning is a subset of AI...",
        "Python is a programming language...",
        "Deep learning uses neural networks...",
    ]

    results = await reranker.rerank(query, documents, top_k=2)
    for idx, doc, score in results:
        print(f"Score: {score:.3f} - {doc[:50]}")

Limitations
-----------
- Requires the chat endpoint to return token-level logprobs.
- Azure OpenAI supports this for certain deployments (e.g., gpt-4o, gpt-4-turbo).
- If logprobs are missing, returns 0.0 scores with a warning.

License
-------
MIT
"""

from __future__ import annotations

import asyncio
import logging
import math
from dataclasses import dataclass
from typing import Any

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncAzureOpenAI,
    BadRequestError,
    RateLimitError,
)

from .rate_limiter import RateLimiter

logger = logging.getLogger(__name__)

# -----------------------------
# Configuration
# -----------------------------


@dataclass
class RerankerConfig:
    """
    Configuration for the logprob-based reranker.

    Attributes:
        model: Model/deployment name to use for reranking
        bins: List of bin tokens representing relevance levels
        top_logprobs: Number of top logprob candidates to request
        logprob_floor: Floor value for missing bin tokens
        temperature: Sampling temperature for generation
        max_tokens: Maximum tokens to generate (1 for bin selection)
        timeout: Request timeout in seconds
    """

    model: str = "gpt-4o-east-US"
    bins: list[str] | None = None
    top_logprobs: int = 5
    logprob_floor: float = -16.0
    temperature: float = 0.2
    max_tokens: int = 1
    timeout: float = 30.0
    rpm_limit: int = 2700  # Requests per minute (90% of 3000 for safety margin)
    tpm_limit: int = 450000  # Tokens per minute

    def __post_init__(self) -> None:
        """Initialize default bins if not provided."""
        if self.bins is None or len(self.bins) == 0:
            # Default to 11 bins mapping evenly to [0.0..1.0]
            self.bins = [str(i) for i in range(11)]
        if self.model is None:
            self.model = "gpt-4o-east-US"


@dataclass
class RerankResult:
    """
    Result from reranking a single document.

    Attributes:
        index: Original index in the input document list
        document: The document text
        score: Relevance score in [0.0..1.0]
        bin_probabilities: Optional probability distribution over bins
    """

    index: int
    document: str
    score: float
    bin_probabilities: dict[str, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result: dict[str, Any] = {
            "index": self.index,
            "document": self.document,
            "score": self.score,
        }
        if self.bin_probabilities is not None:
            result["bin_probabilities"] = self.bin_probabilities
        return result


# -----------------------------
# Utilities
# -----------------------------


def _softmax_logprobs(logprob_map: dict[str, float]) -> dict[str, float]:
    """
    Apply softmax to convert logprobs to probabilities.

    Args:
        logprob_map: Mapping of tokens to their log probabilities

    Returns:
        Mapping of tokens to probabilities (sums to 1.0)
    """
    if not logprob_map:
        return {}

    vals = list(logprob_map.values())
    max_val = max(vals)

    # Subtract max for numerical stability
    exps = [math.exp(v - max_val) for v in vals]
    total = sum(exps) or 1.0

    return {tok: exp / total for tok, exp in zip(logprob_map.keys(), exps)}


def _expected_from_bins(token_probs: dict[str, float], bins: list[str]) -> float:
    """
    Compute expected relevance score from bin probabilities.

    Bins are mapped evenly to [0.0..1.0] range. For example, with 11 bins:
    - "0" maps to 0.0
    - "5" maps to 0.5
    - "10" maps to 1.0

    Args:
        token_probs: Probability distribution over bin tokens
        bins: Ordered list of bin tokens

    Returns:
        Expected relevance score in [0.0..1.0]
    """
    if not bins:
        return 0.0

    n = len(bins)
    if n == 1:
        return 1.0

    expected = 0.0
    for i, tok in enumerate(bins):
        # Map bin index to [0.0..1.0]
        value = i / (n - 1)
        prob = token_probs.get(tok, 0.0)
        expected += prob * value

    return expected


def _build_messages(query: str, document: str, bins: list[str] | None) -> list[dict[str, str]]:
    """
    Construct prompt messages for relevance scoring.

    Args:
        query: The query/question
        document: The document to score
        bins: Optional list of bin tokens (if None, defaults to 11 bins "0".."10")

    Returns:
        List of message dicts for chat completion
    """
    # Ensure bins is always a list[str] at runtime for downstream code and for type-checkers
    if not bins:
        bins = [str(i) for i in range(11)]

    bins_str = ", ".join(bins)

    system_prompt = (
        "You are a semantic relevance judge. Given a query and a document, "
        "output exactly one token from the provided set representing how relevant "
        "the document is to answering the query."
    )

    user_prompt = (
        f"Output one token from [{bins_str}] representing document relevance on a scale from 0.0 to 1.0.\n\n"
        f"Query: {query}\n\n"
        f"Document: {document}\n\n"
        f"Relevance token:"
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


# -----------------------------
# Logprob-Based Reranker
# -----------------------------


class LogprobReranker:
    """
    Logprob-based reranker using Azure OpenAI chat completions.

    This reranker scores documents by analyzing the log probabilities of relevance
    bin tokens in the model's output. It provides calibrated, probabilistic relevance
    scores without requiring fine-tuning or specialized models.

    The reranker can work with either:
    - An AzureLLMClient instance (recommended for cost tracking and rate limiting)
    - A raw AsyncAzureOpenAI client

    Attributes:
        client: Azure OpenAI client for API calls
        config: Reranker configuration
        rate_limiter: Rate limiter for controlling API call rate
    """

    def __init__(
        self,
        client: Any,  # AzureLLMClient or AsyncAzureOpenAI
        config: RerankerConfig | None = None,
        rate_limiter: RateLimiter | None = None,
    ) -> None:
        """
        Initialize the reranker.

        Args:
            client: AzureLLMClient or AsyncAzureOpenAI instance
            config: Optional reranker configuration
            rate_limiter: Optional rate limiter (creates default if not provided)
        """
        self.client = client
        self.config = config or RerankerConfig()

        # Determine the underlying OpenAI client
        if hasattr(client, "client") and isinstance(client.client, AsyncAzureOpenAI):
            # AzureLLMClient wrapper
            self._openai_client = client.client
            self._use_toolkit_client = True
        elif isinstance(client, AsyncAzureOpenAI):
            # Direct AsyncAzureOpenAI client
            self._openai_client = client
            self._use_toolkit_client = False
        else:
            raise TypeError(f"client must be AzureLLMClient or AsyncAzureOpenAI, got {type(client)}")

        # Set up rate limiter
        if rate_limiter is not None:
            self.rate_limiter = rate_limiter
        else:
            # Create default rate limiter with config values
            self.rate_limiter = RateLimiter(
                rpm_limit=self.config.rpm_limit,
                tpm_limit=self.config.tpm_limit,
            )

    async def score(
        self,
        query: str,
        document: str,
        include_bin_probs: bool = False,
    ) -> float | tuple[float, dict[str, float]]:
        """
        Score a single document for relevance to a query.

        Args:
            query: The query/question
            document: The document to score
            include_bin_probs: If True, return (score, bin_probabilities) tuple

        Returns:
            If include_bin_probs is False: relevance score in [0.0..1.0]
            If include_bin_probs is True: (score, bin_probabilities) tuple
            Returns 0.0 (or (0.0, {})) if logprobs unavailable or API error
        """
        cfg = self.config
        messages = _build_messages(query, document, cfg.bins)  # type: ignore[arg-type]

        # Estimate tokens for rate limiting
        # Approximate: system prompt ~40 tokens, user prompt ~30 tokens, query + doc
        estimated_tokens = 70 + len(query.split()) + len(document.split())

        # Acquire rate limit permission
        await self.rate_limiter.acquire(tokens=estimated_tokens)

        try:
            # Cast messages to Any and use guarded attribute access for the third-party client
            # to satisfy static checkers while preserving runtime behavior.
            from typing import Any, cast

            client_chat = getattr(self._openai_client, "chat", None)  # type: ignore[attr-defined]
            if client_chat is None:
                raise RuntimeError("OpenAI client does not expose 'chat' API")

            # `messages` is a list[dict[str,str]] here; the external client types can be strict,
            # so cast to Any for the call site to silence static complaint while preserving runtime.
            resp = await client_chat.completions.create(  # type: ignore[attr-defined]
                model=cfg.model,
                messages=cast(Any, messages),
                logprobs=True,
                top_logprobs=cfg.top_logprobs,
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
                timeout=cfg.timeout,
            )
        except (
            APIConnectionError,
            APITimeoutError,
            APIStatusError,
            RateLimitError,
            BadRequestError,
        ) as e:
            logger.warning(f"Reranker API call failed ({e.__class__.__name__}): {e}. Returning score 0.0")
            return (0.0, {}) if include_bin_probs else 0.0
        except Exception as e:
            logger.error(f"Unexpected error in reranker.score: {e}")
            return (0.0, {}) if include_bin_probs else 0.0

        # Update rate limiter with actual token usage if available
        if hasattr(resp, "usage") and resp.usage:
            try:
                actual_tokens = int(getattr(resp.usage, "total_tokens", estimated_tokens))
                self.rate_limiter.update_usage(actual_tokens, estimated_tokens)
            except (TypeError, ValueError):
                # If we can't get valid token count, use estimate
                pass

        # Extract logprobs from response
        if not hasattr(resp, "choices") or not resp.choices:
            logger.warning("Reranker response has no choices. Returning score 0.0")
            return (0.0, {}) if include_bin_probs else 0.0

        choice = resp.choices[0]
        logprobs_obj = getattr(choice, "logprobs", None)

        # Collect bin logprobs from first token
        bins_logprob: dict[str, float] = {}

        if logprobs_obj and hasattr(logprobs_obj, "content"):
            content_items = logprobs_obj.content
            if isinstance(content_items, list) and content_items:
                item0 = content_items[0]
                top_list = getattr(item0, "top_logprobs", None)

                if isinstance(top_list, list):
                    for cand in top_list:
                        tok = getattr(cand, "token", None)
                        lp = getattr(cand, "logprob", None)

                        if tok is None or lp is None:
                            continue

                        if tok in cfg.bins:
                            bins_logprob[tok] = float(lp)

        if not bins_logprob:
            logger.debug("No bin tokens found in logprobs. Model may not support logprobs or bins are incorrect.")

        # Fill missing bins with floor value (ensure bins is not None)
        bins_list = cfg.bins or [str(i) for i in range(11)]
        for bin_tok in bins_list:
            if bin_tok not in bins_logprob:
                bins_logprob[bin_tok] = cfg.logprob_floor

        # Convert to probabilities and compute expected score
        token_probs = _softmax_logprobs(bins_logprob)
        score = _expected_from_bins(token_probs, cfg.bins)  # type: ignore[arg-type]

        if include_bin_probs:
            return float(score), token_probs
        return float(score)

    async def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int | None = None,
        include_bin_probs: bool = False,
    ) -> list[RerankResult]:
        """
        Rerank a list of documents by relevance to a query.

        Documents are scored in parallel and returned sorted by score (descending).

        Args:
            query: The query/question
            documents: List of documents to rank
            top_k: If provided, return only top k results
            include_bin_probs: If True, include bin probability distributions in results

        Returns:
            List of RerankResult objects sorted by score (highest first)
        """
        if not documents:
            return []

        # Score all documents concurrently
        tasks = [self.score(query, doc, include_bin_probs=include_bin_probs) for doc in documents]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Defensive: ensure results is a sequence for static type-checkers
        if results is None:
            results = []

        # Build result objects
        rerank_results: list[RerankResult] = []
        for i, doc in enumerate(documents):
            # Safe indexing into results; if results shorter than documents, treat missing as failure
            result = results[i] if i < len(results) else None

            if isinstance(result, Exception):
                logger.warning(f"Document {i} scoring failed: {result.__class__.__name__}: {result}")
                score = 0.0
                bin_probs = None
            elif include_bin_probs and isinstance(result, tuple):
                # result is expected to be (score, bin_probs)
                try:
                    score, bin_probs = result  # type: ignore[assignment]
                    score = float(score)  # coerce to float explicitly
                except Exception:
                    score = 0.0
                    bin_probs = None
            else:
                # `result` can be a numeric value or a dynamic structure; coerce safely.
                try:
                    score = float(result) if result is not None else 0.0  # type: ignore[arg-type]
                except Exception:
                    if isinstance(result, tuple) and len(result) > 0:
                        try:
                            score = float(result[0])  # type: ignore[index]
                        except Exception:
                            score = 0.0
                    else:
                        score = 0.0
                bin_probs = None

            rerank_results.append(
                RerankResult(
                    index=i,
                    document=doc,
                    score=score,
                    bin_probabilities=bin_probs,
                )
            )

        # Sort by score descending
        rerank_results.sort(key=lambda x: x.score, reverse=True)

        # Apply top_k limit if specified
        if top_k is not None and top_k > 0:
            rerank_results = rerank_results[:top_k]

        return rerank_results


# -----------------------------
# Convenience Functions
# -----------------------------


def create_reranker(
    client: Any,
    model: str = "gpt-4o-east-US",
    bins: list[str] | None = None,
    rate_limiter: RateLimiter | None = None,
    rpm_limit: int = 2700,
    tpm_limit: int = 450000,
    **kwargs: Any,
) -> LogprobReranker:
    """
    Create a reranker with optional configuration overrides.

    Args:
        client: AzureLLMClient or AsyncAzureOpenAI instance
        model: Model/deployment name (default: "gpt-4o-east-US")
        bins: Custom bin tokens (optional)
        rate_limiter: Optional rate limiter instance (creates default if not provided)
        rpm_limit: Requests per minute limit (default: 2700)
        tpm_limit: Tokens per minute limit (default: 450000)
        **kwargs: Additional RerankerConfig parameters

    Returns:
        Configured LogprobReranker instance

    Example:
        reranker = create_reranker(
            client=client,
            model="gpt-4o",
            temperature=0.1,
            rpm_limit=3000,
            tpm_limit=500000,
        )
    """
    config = RerankerConfig(model=model, bins=bins, rpm_limit=rpm_limit, tpm_limit=tpm_limit, **kwargs)
    return LogprobReranker(client=client, config=config, rate_limiter=rate_limiter)


__all__ = [
    "LogprobReranker",
    "RerankerConfig",
    "RerankResult",
    "create_reranker",
]
