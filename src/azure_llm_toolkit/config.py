"""Configuration management for Azure OpenAI clients."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import tiktoken
from dotenv import load_dotenv
from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncAzureOpenAI, RateLimitError
from pydantic import BaseModel, Field, field_validator, model_validator
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

# Load .env early
load_dotenv()


def _normalize_endpoint(url: str) -> str:
    """Normalize Azure endpoint by stripping trailing slashes."""
    u = url.strip()
    while u.endswith("/"):
        u = u[:-1]
    return u


def _redact(s: str | None) -> str | None:
    """Redact sensitive strings for logging."""
    if not s:
        return s
    if len(s) <= 8:
        return "****"
    return f"{s[:4]}****{s[-4:]}"


class AzureConfig(BaseModel):
    """
    Configuration for Azure OpenAI clients.

    Supports loading from environment variables with the following precedence:
    1. Explicit constructor arguments
    2. Environment variables
    3. Default values
    """

    # Azure OpenAI credentials
    api_key: str = Field(default_factory=lambda: os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") or "")
    endpoint: str = Field(
        default_factory=lambda: os.getenv("AZURE_ENDPOINT") or os.getenv("AZURE_OPENAI_ENDPOINT") or ""
    )
    api_version: str = Field(default=os.getenv("AZURE_API_VERSION", "2024-12-01-preview"))

    # Deployment names
    chat_deployment: str = Field(default=os.getenv("AZURE_CHAT_DEPLOYMENT", "gpt-5-mini"))
    reranker_deployment: str = Field(default=os.getenv("AZURE_RERANKER_DEPLOYMENT", "gpt-4o-east-US"))
    embedding_deployment: str = Field(default=os.getenv("AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-3-large"))

    # Request configuration (None = infinite timeout, recommended for reasoning models)
    timeout_seconds: float | None = Field(
        default_factory=lambda: float(os.getenv("AZURE_TIMEOUT_SECONDS"))
        if os.getenv("AZURE_TIMEOUT_SECONDS")
        else None
    )
    max_retries: int = Field(default=int(os.getenv("AZURE_MAX_RETRIES", "5")))

    # Tokenizer model (defaults to chat deployment)
    tokenizer_model: str = Field(default=os.getenv("TOKENIZER_MODEL", "gpt-4o"))

    # Optional: cache directory for storing embedding dimensions
    cache_dir: Path | None = Field(default=None)

    @field_validator("endpoint")
    @classmethod
    def _validate_endpoint(cls, v: str) -> str:
        if not v:
            return v
        return _normalize_endpoint(v)

    @model_validator(mode="after")
    def _validate_config(self) -> AzureConfig:
        """Validate required fields are present."""
        if not self.api_key:
            logger.warning("Azure API key not set. Set AZURE_OPENAI_API_KEY or OPENAI_API_KEY environment variable.")
        if not self.endpoint:
            logger.warning("Azure endpoint not set. Set AZURE_ENDPOINT or AZURE_OPENAI_ENDPOINT environment variable.")
        return self

    def ensure_ready(self) -> None:
        """
        Validate that required configuration is present.

        Raises:
            ValueError: If required configuration is missing
        """
        if not self.api_key:
            raise ValueError(
                "Missing Azure API key. Set AZURE_OPENAI_API_KEY (or OPENAI_API_KEY) environment variable."
            )
        if not self.endpoint:
            raise ValueError(
                "Missing Azure endpoint. Set AZURE_ENDPOINT (or AZURE_OPENAI_ENDPOINT) environment variable."
            )

    def get_token_encoder(self) -> tiktoken.Encoding:
        """
        Get the tiktoken encoder for the configured tokenizer model.

        Returns:
            tiktoken.Encoding instance
        """
        try:
            return tiktoken.encoding_for_model(self.tokenizer_model)
        except Exception:
            logger.debug(f"Tokenizer model '{self.tokenizer_model}' not recognized; falling back to cl100k_base.")
            return tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text using the configured tokenizer.

        Args:
            text: Text to tokenize

        Returns:
            Number of tokens
        """
        encoder = self.get_token_encoder()
        return len(encoder.encode(text))

    def create_client(self) -> AsyncAzureOpenAI:
        """
        Create an AsyncAzureOpenAI client with this configuration.

        Returns:
            Configured AsyncAzureOpenAI client
        """
        self.ensure_ready()
        return AsyncAzureOpenAI(
            azure_endpoint=self.endpoint,
            api_key=self.api_key,
            api_version=self.api_version,
            timeout=self.timeout_seconds,
        )

    def summarize(self) -> dict[str, Any]:
        """
        Get a summary of the configuration with redacted credentials.

        Returns:
            Dict with configuration summary
        """
        return {
            "api_key": _redact(self.api_key),
            "endpoint": self.endpoint,
            "api_version": self.api_version,
            "chat_deployment": self.chat_deployment,
            "embedding_deployment": self.embedding_deployment,
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
            "tokenizer_model": self.tokenizer_model,
        }

    def _get_cache_path(self) -> Path | None:
        """Get the cache file path for embedding dimensions."""
        if self.cache_dir is None:
            return None
        cache_dir = Path(self.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / "embedding_dimensions.json"

    def get_cached_embedding_dimension(self) -> int | None:
        """
        Get cached embedding dimension for the current configuration.

        Returns:
            Cached dimension or None if not found
        """
        cache_path = self._get_cache_path()
        if cache_path is None or not cache_path.exists():
            return None

        try:
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            key = f"{self.endpoint}|{self.embedding_deployment}|{self.api_version}"
            dim = data.get(key)
            if isinstance(dim, int) and dim > 0:
                logger.debug(f"Embedding dimension cache hit: {dim} (deployment={self.embedding_deployment})")
                return dim
        except Exception as e:
            logger.debug(f"Failed to read embedding dimension cache: {e}")

        return None

    def cache_embedding_dimension(self, dim: int) -> None:
        """
        Cache embedding dimension for the current configuration.

        Args:
            dim: Embedding dimension to cache
        """
        cache_path = self._get_cache_path()
        if cache_path is None:
            return

        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            key = f"{self.endpoint}|{self.embedding_deployment}|{self.api_version}"

            data: dict[str, int] = {}
            if cache_path.exists():
                try:
                    data = json.loads(cache_path.read_text(encoding="utf-8"))
                except Exception:
                    data = {}

            data[key] = dim
            cache_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
            logger.debug(f"Cached embedding dimension: {dim} (deployment={self.embedding_deployment})")
        except Exception as e:
            logger.warning(f"Failed to cache embedding dimension: {e}")


@retry(
    reraise=True,
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((APIConnectionError, RateLimitError, APITimeoutError, APIStatusError)),
)
async def detect_embedding_dimension(config: AzureConfig, client: AsyncAzureOpenAI | None = None) -> int:
    """
    Detect embedding dimension for the configured deployment.

    Checks cache first, then probes the API if necessary.

    Args:
        config: Azure configuration
        client: Optional pre-configured client (will create one if not provided)

    Returns:
        Embedding dimension

    Raises:
        RuntimeError: If dimension detection fails
    """
    # Check for forced override (useful for testing or offline scenarios)
    forced = os.getenv("FORCE_EMBED_DIM")
    if forced:
        try:
            dim = int(forced)
            if dim > 0:
                config.cache_embedding_dimension(dim)
                logger.warning(
                    f"Embedding dimension override via FORCE_EMBED_DIM={dim} (deployment={config.embedding_deployment})"
                )
                return dim
        except ValueError:
            logger.error(f"FORCE_EMBED_DIM='{forced}' is not a valid integer; ignoring override")

    # Check cache
    cached = config.get_cached_embedding_dimension()
    if cached:
        return cached

    # Probe the API
    client = client or config.create_client()
    logger.debug(
        f"Detecting embedding dimension for deployment '{config.embedding_deployment}' at endpoint '{config.endpoint}'"
    )

    try:
        resp = await client.embeddings.create(
            model=config.embedding_deployment,
            input="dimension-probe",
        )
        dim = len(resp.data[0].embedding)
        if dim <= 0:
            raise RuntimeError(f"Embedding dimension probe returned non-positive dimension: {dim}")

        config.cache_embedding_dimension(dim)
        logger.info(
            f"Embedding dimension detected: {dim} (deployment={config.embedding_deployment}, endpoint={config.endpoint})"
        )
        return dim
    except Exception as e:
        logger.error(
            f"Embedding dimension probe failed (deployment={config.embedding_deployment}, endpoint={config.endpoint}): {e}"
        )
        raise


__all__ = [
    "AzureConfig",
    "detect_embedding_dimension",
]
