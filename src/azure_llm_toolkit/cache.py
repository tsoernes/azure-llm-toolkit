"""Disk-based caching for LLM calls to save costs and improve performance.

This module provides caching mechanisms for embeddings and chat completions,
storing results on disk to avoid redundant API calls.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)


def _compute_hash(data: Any) -> str:
    """
    Compute a stable hash for any data structure.

    Args:
        data: Data to hash (dict, list, str, etc.)

    Returns:
        Hex digest of the hash
    """
    # Convert to stable JSON representation
    json_str = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(json_str.encode()).hexdigest()


class LLMCache:
    """
    Base class for LLM caching.

    Provides disk-based caching for LLM responses to avoid redundant API calls.
    """

    def __init__(self, cache_dir: Path | str = ".llm_cache", enabled: bool = True) -> None:
        """
        Initialize cache.

        Args:
            cache_dir: Directory to store cache files
            enabled: Whether caching is enabled
        """
        self.cache_dir = Path(cache_dir)
        self.enabled = enabled

        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_path(self, key: str, extension: str = "json") -> Path:
        """Get cache file path for a key."""
        return self.cache_dir / f"{key}.{extension}"

    def clear(self) -> int:
        """
        Clear all cached entries.

        Returns:
            Number of files deleted
        """
        if not self.cache_dir.exists():
            return 0

        count = 0
        for file in self.cache_dir.iterdir():
            if file.is_file():
                file.unlink()
                count += 1

        logger.info(f"Cleared {count} cache entries from {self.cache_dir}")
        return count

    def get_stats(self) -> dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dict with cache stats (size, file count, etc.)
        """
        if not self.cache_dir.exists():
            return {"enabled": self.enabled, "size_bytes": 0, "file_count": 0}

        total_size = 0
        file_count = 0

        for file in self.cache_dir.iterdir():
            if file.is_file():
                total_size += file.stat().st_size
                file_count += 1

        return {
            "enabled": self.enabled,
            "size_bytes": total_size,
            "size_mb": total_size / (1024 * 1024),
            "file_count": file_count,
            "cache_dir": str(self.cache_dir),
        }


class EmbeddingCache(LLMCache):
    """
    Cache for embedding results.

    Caches embeddings based on text content and model name.
    """

    def _make_key(self, text: str, model: str) -> str:
        """Create cache key from text and model."""
        data = {"text": text, "model": model}
        return _compute_hash(data)

    def get(self, text: str, model: str) -> npt.NDArray[np.float64] | None:
        """
        Get cached embedding.

        Args:
            text: Text that was embedded
            model: Model name used

        Returns:
            Cached embedding or None if not found
        """
        if not self.enabled:
            return None

        key = self._make_key(text, model)
        cache_path = self._get_cache_path(key, extension="npy")

        if not cache_path.exists():
            return None

        try:
            embedding = np.load(cache_path)
            logger.debug(f"Cache hit for embedding (key={key[:8]}...)")
            return embedding
        except Exception as e:
            logger.warning(f"Failed to load cached embedding: {e}")
            return None

    def set(self, text: str, model: str, embedding: npt.NDArray[np.float64]) -> None:
        """
        Cache an embedding.

        Args:
            text: Text that was embedded
            model: Model name used
            embedding: Embedding vector to cache
        """
        if not self.enabled:
            return

        key = self._make_key(text, model)
        cache_path = self._get_cache_path(key, extension="npy")

        try:
            np.save(cache_path, embedding, allow_pickle=False)
            logger.debug(f"Cached embedding (key={key[:8]}...)")
        except Exception as e:
            logger.warning(f"Failed to cache embedding: {e}")

    def get_batch(self, texts: list[str], model: str) -> tuple[list[npt.NDArray[np.float64] | None], list[int]]:
        """
        Get cached embeddings for a batch of texts.

        Args:
            texts: List of texts
            model: Model name

        Returns:
            Tuple of (embeddings_or_none, missing_indices)
            - embeddings_or_none: List with cached embeddings or None for misses
            - missing_indices: Indices of texts not in cache
        """
        if not self.enabled:
            return [None] * len(texts), list(range(len(texts)))

        embeddings: list[npt.NDArray[np.float64] | None] = []
        missing_indices: list[int] = []

        for i, text in enumerate(texts):
            embedding = self.get(text, model)
            embeddings.append(embedding)
            if embedding is None:
                missing_indices.append(i)

        return embeddings, missing_indices

    def set_batch(self, texts: list[str], model: str, embeddings: list[npt.NDArray[np.float64]]) -> None:
        """
        Cache a batch of embeddings.

        Args:
            texts: List of texts
            model: Model name
            embeddings: List of embeddings
        """
        if not self.enabled:
            return

        for text, embedding in zip(texts, embeddings):
            self.set(text, model, embedding)


class ChatCache(LLMCache):
    """
    Cache for chat completion results.

    Caches responses based on messages, model, and parameters.
    """

    def _make_key(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Create cache key from chat parameters."""
        data = {
            "messages": messages,
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        return _compute_hash(data)

    def get(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any] | None:
        """
        Get cached chat response.

        Args:
            messages: Chat messages
            model: Model name
            temperature: Temperature parameter
            max_tokens: Max tokens parameter

        Returns:
            Cached response dict or None if not found
        """
        if not self.enabled:
            return None

        key = self._make_key(messages, model, temperature, max_tokens)
        cache_path = self._get_cache_path(key, extension="json")

        if not cache_path.exists():
            return None

        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                response = json.load(f)
            logger.debug(f"Cache hit for chat (key={key[:8]}...)")
            return response
        except Exception as e:
            logger.warning(f"Failed to load cached chat response: {e}")
            return None

    def set(
        self,
        messages: list[dict[str, str]],
        model: str,
        response: dict[str, Any],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> None:
        """
        Cache a chat response.

        Args:
            messages: Chat messages
            model: Model name
            response: Response dict to cache
            temperature: Temperature parameter
            max_tokens: Max tokens parameter
        """
        if not self.enabled:
            return

        key = self._make_key(messages, model, temperature, max_tokens)
        cache_path = self._get_cache_path(key, extension="json")

        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(response, f, indent=2)
            logger.debug(f"Cached chat response (key={key[:8]}...)")
        except Exception as e:
            logger.warning(f"Failed to cache chat response: {e}")


class CacheManager:
    """
    Manager for multiple cache instances.

    Provides unified access to embedding and chat caches.
    """

    def __init__(
        self,
        cache_dir: Path | str = ".llm_cache",
        enabled: bool = True,
        enable_embedding_cache: bool = True,
        enable_chat_cache: bool = True,
    ) -> None:
        """
        Initialize cache manager.

        Args:
            cache_dir: Base directory for caches
            enabled: Master switch for all caching
            enable_embedding_cache: Enable embedding cache
            enable_chat_cache: Enable chat cache
        """
        cache_dir = Path(cache_dir)

        self.embedding_cache = EmbeddingCache(
            cache_dir=cache_dir / "embeddings",
            enabled=enabled and enable_embedding_cache,
        )

        self.chat_cache = ChatCache(
            cache_dir=cache_dir / "chat",
            enabled=enabled and enable_chat_cache,
        )

    def clear_all(self) -> dict[str, int]:
        """
        Clear all caches.

        Returns:
            Dict with counts of cleared entries per cache
        """
        return {
            "embeddings": self.embedding_cache.clear(),
            "chat": self.chat_cache.clear(),
        }

    def get_stats(self) -> dict[str, Any]:
        """
        Get statistics for all caches.

        Returns:
            Dict with stats for each cache
        """
        embedding_stats = self.embedding_cache.get_stats()
        chat_stats = self.chat_cache.get_stats()

        total_size = embedding_stats["size_bytes"] + chat_stats["size_bytes"]
        total_files = embedding_stats["file_count"] + chat_stats["file_count"]

        return {
            "total_size_bytes": total_size,
            "total_size_mb": total_size / (1024 * 1024),
            "total_files": total_files,
            "embeddings": embedding_stats,
            "chat": chat_stats,
        }


__all__ = [
    "LLMCache",
    "EmbeddingCache",
    "ChatCache",
    "CacheManager",
]
