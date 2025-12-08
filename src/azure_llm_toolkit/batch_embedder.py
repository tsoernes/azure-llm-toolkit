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
        Embed a list of texts.

        Handles texts exceeding token limits by splitting and averaging.

        Args:
            texts: List of texts to embed
            show_progress: Whether to show progress bar

        Returns:
            Tuple of (embeddings, metadata) where metadata includes token counts and cost
        """
        # Tokenize all texts
        token_lists = self.encoder.encode_batch(texts, num_threads=multiprocessing.cpu_count())

        # Split long texts
        split_token_lists = []
        split_lengths = []
        total_tokens = 0

        for tokens in token_lists:
            parts = self._split_tokens(tokens)
            split_token_lists.extend(parts)
            split_lengths.append([len(part) for part in parts])
            total_tokens += len(tokens)

        # Embed all parts
        embeddings = await self._embed_token_lists(split_token_lists)

        # Average split embeddings
        final_embeddings = self._weighted_average_embeddings(embeddings, split_lengths)

        # Calculate cost
        cost = self.cost_estimator.estimate_cost(
            model=self.config.embedding_deployment,
            tokens_input=total_tokens,
        )

        metadata = {
            "total_tokens": total_tokens,
            "num_texts": len(texts),
            "num_splits": len(split_token_lists),
            "estimated_cost": cost,
            "currency": self.cost_estimator.currency,
        }

        return final_embeddings, metadata

    def _tokenize_dataframe(
        self,
        df: pl.DataFrame,
        text_column: str,
        verbose: bool = True,
    ) -> pl.DataFrame:
        """
        Tokenize text column in DataFrame.

        Adds columns:
        - {text_column}.tokens: List of token IDs
        - {text_column}.token_count: Number of tokens

        Args:
            df: Polars DataFrame
            text_column: Name of text column
            verbose: Whether to print statistics

        Returns:
            DataFrame with added token columns
        """
        tokens_column = f"{text_column}.tokens"
        token_count_column = f"{text_column}.token_count"

        start_time = datetime.now()

        # Tokenize using multiple CPU cores
        texts = df[text_column].to_list()
        tokens = self.encoder.encode_batch(texts, num_threads=multiprocessing.cpu_count())

        # Add token columns
        df = df.with_columns(
            [
                pl.Series(tokens_column, tokens),
            ]
        )

        df = df.with_columns(
            [
                pl.col(tokens_column).list.len().alias(token_count_column),
            ]
        )

        if verbose:
            total = df[token_count_column].sum()
            mean = df[token_count_column].mean()
            min_val = df[token_count_column].min()
            max_val = df[token_count_column].max()
            elapsed = datetime.now() - start_time

            logger.info(f"Tokenized {total:,} tokens in {elapsed}. Mean: {mean:.1f}, Min: {min_val}, Max: {max_val}")

        return df

    def _create_batches(
        self,
        df: pl.DataFrame,
        token_count_column: str,
    ) -> pl.DataFrame:
        """
        Assign batch IDs based on token and list count limits.

        Ensures that:
        - Each batch stays under max_tokens_per_minute
        - Each batch has fewer than max_lists_per_query texts

        Args:
            df: DataFrame with token counts
            token_count_column: Name of token count column

        Returns:
            DataFrame with added 'batch_id' column
        """
        # Calculate number of sub-lists each text will be split into
        df = df.with_columns(
            (pl.col(token_count_column) / self.max_tokens_per_row).ceil().cast(pl.Int32).alias("n_lists")
        )

        batch_ids = np.zeros(len(df), dtype=np.int32)
        batch_n_tokens = [0]
        batch_n_lists = [0]
        batch_id = 0

        # Iterate through rows to assign batch IDs
        for ix, (token_count, n_lists) in enumerate(df.select([token_count_column, "n_lists"]).iter_rows()):
            # Check if we need to start a new batch
            if (
                batch_n_tokens[batch_id] + token_count >= self.max_tokens_per_minute
                or batch_n_lists[batch_id] + n_lists >= self.max_lists_per_query
            ):
                batch_id += 1
                batch_n_tokens.append(token_count)
                batch_n_lists.append(n_lists)
            else:
                batch_n_tokens[batch_id] += token_count
                batch_n_lists[batch_id] += n_lists

            batch_ids[ix] = batch_id

        return df.with_columns(pl.Series("batch_id", batch_ids)).drop("n_lists")

    async def _process_batch(
        self,
        batch: pl.DataFrame,
        tokens_column: str,
        token_count_column: str,
        embedding_column: str,
        retry_count: int = 0,
    ) -> pl.DataFrame:
        """
        Process a single batch of texts.

        Args:
            batch: Batch DataFrame
            tokens_column: Name of tokens column
            token_count_column: Name of token count column
            embedding_column: Name to use for embedding column
            retry_count: Number of retries attempted

        Returns:
            DataFrame with added embedding column
        """
        token_lists = batch[tokens_column].to_list()
        n_tokens = batch[token_count_column].sum()

        assert n_tokens < self.max_tokens_per_minute, f"Batch exceeds token limit: {n_tokens}"

        try:
            # Split long texts
            split_token_lists = []
            split_lengths = []

            for tokens in token_lists:
                parts = self._split_tokens(tokens)
                split_token_lists.extend(parts)
                split_lengths.append([len(part) for part in parts])

            # Embed
            embeddings = await self._embed_token_lists(split_token_lists)

            # Average split embeddings
            final_embeddings = self._weighted_average_embeddings(embeddings, split_lengths)

            # Convert to Polars Series
            embedding_length = len(final_embeddings[0])
            embedding_series = pl.Series(
                embedding_column,
                final_embeddings,
                dtype=pl.Array(pl.Float64, embedding_length),
            )

            return batch.with_columns(embedding_series)

        except RateLimitError as e:
            logger.warning(
                f"Rate limited on batch (retry {retry_count}). "
                f"Sleeping {self.sleep_sec}s and increasing sleep time by {self.sleep_inc}s. "
                f"Error: {e}"
            )
            import asyncio

            await asyncio.sleep(self.sleep_sec)
            self.sleep_sec += self.sleep_inc
            return await self._process_batch(
                batch, tokens_column, token_count_column, embedding_column, retry_count + 1
            )

        except (InternalServerError, APIConnectionError) as e:
            logger.warning(f"API error on batch (retry {retry_count}). Sleeping {self.sleep_sec}s. Error: {e}")
            import asyncio

            await asyncio.sleep(self.sleep_sec)
            return await self._process_batch(
                batch, tokens_column, token_count_column, embedding_column, retry_count + 1
            )

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
        tokens_column = f"{text_column}.tokens"
        token_count_column = f"{text_column}.token_count"
        embedding_column = f"{text_column}.embedding"

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

        # Tokenize
        df_to_embed = self._tokenize_dataframe(df_to_embed, text_column, verbose=verbose)

        # Create batches
        df_to_embed = self._create_batches(df_to_embed, token_count_column)

        # Calculate cost estimate
        total_tokens = df_to_embed[token_count_column].sum()
        estimated_cost = self.cost_estimator.estimate_cost(
            model=self.config.embedding_deployment,
            tokens_input=total_tokens,
        )

        if verbose:
            n_batches = df_to_embed["batch_id"].n_unique()
            logger.info(
                f"Embedding {len(df_to_embed)} texts in {n_batches} batches. "
                f"Total tokens: {total_tokens:,}. "
                f"Estimated cost: {estimated_cost:.4f} {self.cost_estimator.currency}"
            )

        start_time = datetime.now()

        # Process batches
        embedded_batches = []
        for _, batch in tqdm(
            df_to_embed.group_by("batch_id"),
            desc="Embedding batches",
            disable=not verbose,
            total=df_to_embed["batch_id"].n_unique(),
        ):
            embedded_batch = await self._process_batch(
                batch,
                tokens_column,
                token_count_column,
                embedding_column,
            )
            embedded_batches.append(embedded_batch)

        # Concatenate results
        result_df = pl.concat(embedded_batches)

        if verbose:
            elapsed = datetime.now() - start_time
            logger.info(f"Embedded {len(result_df)} texts in {elapsed}")

        # Merge back with original DataFrame (for rows that weren't embedded)
        # This preserves rows with null text or existing embeddings
        if not re_embed and len(df) > len(result_df):
            # Get rows that weren't embedded
            df_not_embedded = df.filter(pl.col(text_column).is_null() | pl.col(embedding_column).is_not_null())

            # Add missing columns to df_not_embedded
            for col in result_df.columns:
                if col not in df_not_embedded.columns:
                    df_not_embedded = df_not_embedded.with_columns(pl.lit(None).alias(col))

            # Concatenate
            result_df = pl.concat([result_df, df_not_embedded])

        return result_df


__all__ = [
    "PolarsBatchEmbedder",
]
