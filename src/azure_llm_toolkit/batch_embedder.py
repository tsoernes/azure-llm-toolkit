"""Advanced batch embedding with Polars DataFrames for high-performance processing.

This module provides efficient batch embedding capabilities with:
- Polars DataFrame integration for high-performance data processing
- Intelligent batching based on token and list limits
- Weighted averaging for texts exceeding token limits
- Automatic rate limiting and retry logic
- Progress tracking with tqdm
- Optional disk-based caching of partial results
"""

from __future__ import annotations

import logging
import multiprocessing
from collections.abc import Generator
from datetime import datetime
from typing import Any

import numpy as np
import numpy.typing as npt
import tiktoken
from openai import APIConnectionError, InternalServerError, RateLimitError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from .config import AzureConfig
from .cost_tracker import CostEstimator

logger = logging.getLogger(__name__)

import polars as pl
from tqdm import tqdm


class PolarsBatchEmbedder:
    """
    High-performance batch embedder using Polars DataFrames.

    This embedder is optimized for processing large datasets efficiently:
    - Uses Polars for fast DataFrame operations
    - Tokenizes all texts upfront using multiple CPU cores
    - Intelligently batches based on token and list count limits
    - Handles texts exceeding token limits by splitting and averaging
    - Provides automatic rate limiting and error handling

    Example:
        >>> import polars as pl
        >>> from azure_llm_toolkit import AzureConfig, PolarsBatchEmbedder
        >>>
        >>> config = AzureConfig()
        >>> embedder = PolarsBatchEmbedder(config)
        >>>
        >>> df = pl.DataFrame({"text": ["Hello world", "Another text"]})
        >>> result_df = embedder.embed_dataframe(df, text_column="text")
        >>> print(result_df.columns)
        ['text', 'text.tokens', 'text.token_count', 'text.embedding']
    """

    def __init__(
        self,
        config: AzureConfig,
        max_tokens_per_minute: int = 1_000_000,
        max_tokens_per_row: int = 8190,
        max_lists_per_query: int = 2048,
        sleep_sec: int = 60,
        sleep_inc: int = 5,
        cost_estimator: CostEstimator | None = None,
    ) -> None:
        """
        Initialize the Polars batch embedder.

        Args:
            config: Azure configuration
            max_tokens_per_minute: Maximum tokens per minute (rate limit)
            max_tokens_per_row: Maximum tokens per text (texts exceeding this are split)
            max_lists_per_query: Maximum number of texts per API call
            sleep_sec: Initial sleep time on rate limit error (seconds)
            sleep_inc: Sleep time increment on consecutive errors (seconds)
            cost_estimator: Optional cost estimator for tracking
        """
        self.config = config
        self.max_tokens_per_minute = max_tokens_per_minute
        self.max_tokens_per_row = max_tokens_per_row
        self.max_lists_per_query = max_lists_per_query
        self.sleep_sec = sleep_sec
        self.sleep_inc = sleep_inc
        self.cost_estimator = cost_estimator or CostEstimator(currency="kr")

        # Initialize tokenizer
        try:
            self.encoder = tiktoken.encoding_for_model(config.embedding_deployment)
        except Exception:
            logger.debug(f"Using cl100k_base encoder as fallback for {config.embedding_deployment}")
            self.encoder = tiktoken.get_encoding("cl100k_base")

        # Create Azure client
        self.client = config.create_client()

    def _split_tokens(self, tokens: list[int]) -> list[list[int]]:
        """Split token list into chunks of max_tokens_per_row."""
        return [tokens[i : i + self.max_tokens_per_row] for i in range(0, len(tokens), self.max_tokens_per_row)]

    def _weighted_average_embeddings(
        self,
        embeddings: list[npt.NDArray[np.float64]],
        split_lengths: list[list[int]],
    ) -> list[npt.NDArray[np.float64]]:
        """
        Average embeddings from split texts using weighted average.

        When a text exceeds max_tokens_per_row, it's split into parts,
        each part is embedded separately, then combined using weighted
        average based on the length of each part.

        Args:
            embeddings: List of embedding vectors
            split_lengths: List of lists containing token counts for each split part

        Returns:
            List of averaged embeddings (one per original text)
        """
        averaged_results = []
        index = 0

        for lengths in split_lengths:
            n_parts = len(lengths)
            part_embeddings = embeddings[index : index + n_parts]

            if n_parts == 1:
                # No splitting occurred
                averaged_results.append(part_embeddings[0])
            else:
                # Weighted average based on part lengths
                weights = np.array(lengths) / sum(lengths)
                stacked_embeddings = np.stack(part_embeddings)
                weighted_avg = np.sum(stacked_embeddings * weights[:, np.newaxis], axis=0)
                averaged_results.append(weighted_avg)

            index += n_parts

        return averaged_results

    @retry(
        reraise=True,
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=60),
        retry=retry_if_exception_type((APIConnectionError, RateLimitError, InternalServerError)),
    )
    async def _embed_token_lists(self, token_lists: list[list[int]]) -> list[npt.NDArray[np.float64]]:
        """
        Embed token lists with automatic retry logic.

        Args:
            token_lists: List of token lists to embed

        Returns:
            List of embedding vectors

        Raises:
            ValueError: If input validation fails
        """
        # Validation
        if not token_lists:
            raise ValueError("No token lists provided")

        for i, tokens in enumerate(token_lists):
            if not tokens:
                raise ValueError(f"Empty token list at index {i}")
            if len(tokens) > self.max_tokens_per_row:
                raise ValueError(
                    f"Token list {i} exceeds max_tokens_per_row: {len(tokens)} > {self.max_tokens_per_row}"
                )

        if len(token_lists) > self.max_lists_per_query:
            raise ValueError(f"Too many token lists: {len(token_lists)} > {self.max_lists_per_query}")

        # Call API
        response = await self.client.embeddings.create(
            model=self.config.embedding_deployment,
            input=token_lists,
            encoding_format="float",
        )

        # Extract embeddings as numpy arrays
        embeddings = [np.array(item.embedding, dtype=np.float64) for item in response.data]

        return embeddings

    async def embed_texts(
        self,
        texts: list[str],
        show_progress: bool = True,
    ) -> tuple[list[npt.NDArray[np.float64]], dict[str, Any]]:
        """
        Embed a list of texts with intelligent batching and rate limiting.

        This method automatically:
        - Tokenizes all texts
        - Splits texts exceeding token limits
        - Creates intelligent batches respecting TPM/RPM limits
        - Applies rate limiting between batches
        - Performs weighted averaging for split texts

        Args:
            texts: List of texts to embed
            show_progress: Whether to show progress bar

        Returns:
            Tuple of (embeddings, metadata) where metadata includes token counts and cost

        Example:
            >>> embedder = PolarsBatchEmbedder(config)
            >>> embeddings, metadata = await embedder.embed_texts(["text1", "text2"])
            >>> print(f"Embedded {metadata['num_texts']} texts")
        """
        import asyncio

        if not texts:
            return [], {
                "total_tokens": 0,
                "num_texts": 0,
                "num_splits": 0,
                "estimated_cost": 0.0,
                "currency": self.cost_estimator.currency,
            }

        # Tokenize all texts
        token_lists = self.encoder.encode_batch(texts, num_threads=multiprocessing.cpu_count())

        # Split long texts and track which texts were split
        split_token_lists = []
        split_lengths = []
        total_tokens = 0

        for tokens in token_lists:
            parts = self._split_tokens(tokens)
            split_token_lists.extend(parts)
            split_lengths.append([len(part) for part in parts])
            total_tokens += len(tokens)

        # Create batches respecting TPM and list count limits
        batches = []
        current_batch = []
        current_batch_tokens = 0
        current_batch_lists = 0

        for token_list in split_token_lists:
            token_count = len(token_list)
            n_lists = 1

            # Check if adding this would exceed limits
            if (
                current_batch_tokens + token_count >= self.max_tokens_per_minute
                or current_batch_lists + n_lists >= self.max_lists_per_query
            ):
                if current_batch:
                    batches.append(current_batch)
                current_batch = [token_list]
                current_batch_tokens = token_count
                current_batch_lists = n_lists
            else:
                current_batch.append(token_list)
                current_batch_tokens += token_count
                current_batch_lists += n_lists

        # Add final batch
        if current_batch:
            batches.append(current_batch)

        # Process batches with progress bar and rate limiting
        all_embeddings = []
        iterator = tqdm(batches, desc="Embedding batches", disable=not show_progress)

        for batch_idx, batch in enumerate(iterator):
            try:
                batch_embeddings = await self._embed_token_lists(batch)
                all_embeddings.extend(batch_embeddings)

                # Sleep between batches to respect rate limits (except last batch)
                if batch_idx < len(batches) - 1:
                    await asyncio.sleep(self.sleep_sec)

            except RateLimitError as e:
                logger.warning(f"Rate limited on batch {batch_idx}. Sleeping {self.sleep_sec}s. Error: {e}")
                await asyncio.sleep(self.sleep_sec)
                self.sleep_sec += self.sleep_inc
                # Retry the batch
                batch_embeddings = await self._embed_token_lists(batch)
                all_embeddings.extend(batch_embeddings)

        # Average split embeddings
        final_embeddings = self._weighted_average_embeddings(all_embeddings, split_lengths)

        # Calculate cost
        cost = self.cost_estimator.estimate_cost(
            model=self.config.embedding_deployment,
            tokens_input=total_tokens,
        )

        metadata = {
            "total_tokens": total_tokens,
            "num_texts": len(texts),
            "num_splits": len(split_token_lists),
            "num_batches": len(batches),
            "estimated_cost": cost,
            "currency": self.cost_estimator.currency,
        }

        return final_embeddings, metadata

    async def embed_dataframe(
        self,
        df: pl.DataFrame,
        text_column: str,
        re_embed: bool = False,
        verbose: bool = True,
    ) -> pl.DataFrame:
        """
        Embed texts in a Polars DataFrame.

        Adds columns:
        - {text_column}.tokens: Token IDs
        - {text_column}.token_count: Token counts
        - {text_column}.embedding: Embedding vectors

        Args:
            df: Input DataFrame
            text_column: Name of text column to embed
            re_embed: Re-embed texts that already have embeddings
            verbose: Whether to print progress information

        Returns:
            DataFrame with added embedding columns
        """
        embedding_column = f"{text_column}.embedding"
        token_count_column = f"{text_column}.token_count"

        # Initialize embedding column if it doesn't exist
        if embedding_column not in df.columns:
            df = df.with_columns(pl.lit(None).alias(embedding_column))

        # Filter out rows to embed
        df_to_embed = df.filter(pl.col(text_column).is_not_null())

        if not re_embed:
            df_to_embed = df_to_embed.filter(pl.col(embedding_column).is_null())

        if len(df_to_embed) == 0:
            logger.info("No texts to embed (all already embedded)")
            return df

        # Extract texts to embed
        texts_to_embed = df_to_embed[text_column].to_list()

        if verbose:
            logger.info(f"Embedding {len(texts_to_embed)} texts...")

        # Use embed_texts for actual embedding (handles batching, rate limiting, etc.)
        embeddings, metadata = await self.embed_texts(texts_to_embed, show_progress=verbose)

        if verbose:
            logger.info(
                f"Embedded {metadata['num_texts']} texts. "
                f"Total tokens: {metadata['total_tokens']:,}. "
                f"Estimated cost: {metadata['estimated_cost']:.4f} {metadata['currency']}"
            )

        # Add embeddings and token counts to DataFrame
        embedding_length = len(embeddings[0])

        # Tokenize to get token counts (reuse from embed_texts processing)
        token_counts = [self.config.count_tokens(text) for text in texts_to_embed]

        df_to_embed = df_to_embed.with_columns(
            [
                pl.Series(embedding_column, embeddings, dtype=pl.Array(pl.Float64, embedding_length)),
                pl.Series(token_count_column, token_counts),
            ]
        )

        # Merge back with original DataFrame (for rows that weren't embedded)
        if not re_embed and len(df) > len(df_to_embed):
            # Get rows that weren't embedded
            df_not_embedded = df.filter(pl.col(text_column).is_null() | pl.col(embedding_column).is_not_null())

            # Add missing columns to df_not_embedded
            for col in df_to_embed.columns:
                if col not in df_not_embedded.columns:
                    if col == embedding_column:
                        df_not_embedded = df_not_embedded.with_columns(
                            pl.lit(None).cast(pl.Array(pl.Float64, embedding_length)).alias(col)
                        )
                    else:
                        df_not_embedded = df_not_embedded.with_columns(pl.lit(None).alias(col))

            # Concatenate
            result_df = pl.concat([df_to_embed, df_not_embedded])
        else:
            result_df = df_to_embed

        return result_df


__all__ = [
    "PolarsBatchEmbedder",
]
