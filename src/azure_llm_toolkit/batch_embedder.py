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
from .rate_limiter import RateLimiter

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
        use_rate_limiting: bool = True,
        rate_limiter: RateLimiter | None = None,
        use_batch_api: bool = False,
        batch_api_client: Any | None = None,
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
            use_rate_limiting: Whether to use the provided rate_limiter for batch calls
            rate_limiter: Optional RateLimiter instance to coordinate batch-level acquires
            use_batch_api: If True, use the (future) Batch API path for embedding operations
            batch_api_client: Optional client/handler for the Batch API (pass None to use default)
            cost_estimator: Optional cost estimator for tracking
        """
        self.config = config
        self.max_tokens_per_minute = max_tokens_per_minute
        self.max_tokens_per_row = max_tokens_per_row
        self.max_lists_per_query = max_lists_per_query
        self.sleep_sec = sleep_sec
        self.sleep_inc = sleep_inc
        self.use_rate_limiting = use_rate_limiting
        self.rate_limiter = rate_limiter
        self.use_batch_api = use_batch_api
        self.batch_api_client = batch_api_client
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

    # ----------------------------
    # Backward-compatible helpers
    # ----------------------------
    def _tokenize_dataframe(self, df: pl.DataFrame, text_column: str, verbose: bool = True) -> pl.DataFrame:
        """
        Tokenize a DataFrame column and add two columns:
        - {text_column}.tokens: list[int] or None for null texts
        - {text_column}.token_count: int token counts (0 for null)

        If the columns already exist, this is a no-op for that column.
        """
        tokens_col = f"{text_column}.tokens"
        token_count_col = f"{text_column}.token_count"

        # If tokens already present, return unchanged
        if tokens_col in df.columns and token_count_col in df.columns:
            return df

        texts = df[text_column].to_list()
        token_lists: list[list[int] | None] = []
        token_counts: list[int] = []

        # Use encoder.encode_batch when available for speed; fall back gracefully
        try:
            # Some tiktoken encoders support encode_batch with num_threads
            token_lists_all = self.encoder.encode_batch(
                [t if t is not None else "" for t in texts], num_threads=multiprocessing.cpu_count()
            )
            for original_text, tl in zip(texts, token_lists_all):
                if original_text is None:
                    token_lists.append(None)
                    token_counts.append(0)
                else:
                    token_lists.append(list(tl))
                    token_counts.append(len(tl))
        except Exception:
            # Fallback: encode individually
            for t in texts:
                if t is None:
                    token_lists.append(None)
                    token_counts.append(0)
                else:
                    try:
                        tl = self.encoder.encode(t)
                        token_lists.append(list(tl))
                        token_counts.append(len(tl))
                    except Exception:
                        # Last resort: estimate tokens using config.count_tokens
                        est = self.config.count_tokens(t)
                        token_lists.append(None)
                        token_counts.append(est)

        # Attach columns (Polars can accept Python lists)
        out_cols = []
        if tokens_col not in df.columns:
            out_cols.append(pl.Series(tokens_col, token_lists))
        if token_count_col not in df.columns:
            out_cols.append(pl.Series(token_count_col, token_counts))

        if out_cols:
            df = df.with_columns(out_cols)

        if verbose:
            logger.debug(f"Tokenized {len(texts)} rows (column={text_column})")

        return df

    def _create_batches(self, df: pl.DataFrame, token_count_column: str) -> pl.DataFrame:
        """
        Create a simple batch id column for the DataFrame.

        The strategy here is intentionally simple and deterministic:
        - Assign sequential batch ids grouping up to self.max_lists_per_query items each.
        - This helper exists to support tests and basic batching inspection.

        Returns DataFrame with additional column `batch_id`.
        """
        batch_col = "batch_id"
        if batch_col in df.columns:
            return df

        n = len(df)
        # Avoid zero division
        group_size = max(1, int(self.max_lists_per_query))
        batch_ids = [i // group_size for i in range(n)]

        df = df.with_columns(pl.Series(batch_col, batch_ids))
        return df

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

    async def _embed_via_batch_api(self, token_lists: list[list[int]]) -> list[npt.NDArray[np.float64]]:
        """
        Attempt to embed token lists using a Batch API if a batch_api_client is configured.
        Falls back to the regular per-request embedding implementation if the batch client
        is not available or the batch job fails.

        This is an opt-in path; set `use_batch_api=True` and provide `batch_api_client`
        when constructing the PolarsBatchEmbedder to enable it.

        The function assumes a minimal, generic async batch client interface:
          - create(model=..., inputs=...) -> returns a job descriptor with `id`
          - get_status(job_id) -> returns an object with `.state` in {'pending','running','succeeded','failed'}
          - get_result(job_id) -> returns a result object with `.data` containing embeddings similar to the per-call API

        The implementation is best-effort and intentionally defensive because concrete batch APIs
        and SDKs differ; the method will log and fall back to per-call embedding on any unexpected errors.
        """
        # If no batch client configured, fall back immediately
        batch_client = getattr(self, "batch_api_client", None)
        if not batch_client:
            return await self._embed_token_lists(token_lists)

        try:
            # Create job (best-effort call signature)
            create_kwargs = {
                "model": self.config.embedding_deployment,
                "inputs": token_lists,
            }
            job = await batch_client.create(**create_kwargs)  # type: ignore[attr-defined]

            # Job descriptor may be an object with attributes or a dict; accept both shapes.
            job_id = None
            # Attribute-style access (object with .id or .job_id)
            job_id = getattr(job, "id", None) or getattr(job, "job_id", None)
            # Dict-style access (mapping with 'id' or 'job_id')
            if job_id is None and isinstance(job, dict):
                job_id = job.get("id") or job.get("job_id")
            if not job_id:
                # Unknown job id shape; fallback to per-call embedding
                logger.debug("Batch API returned job without id; falling back to per-call embed")
                return await self._embed_token_lists(token_lists)

            # Poll for completion with exponential backoff up to a reasonable timeout
            import asyncio as _asyncio

            start = datetime.now()
            timeout_seconds = 60 * 15  # 15 minutes default timeout for batch jobs
            poll_interval = 2.0

            while True:
                status = await batch_client.get_status(job_id)  # type: ignore[attr-defined]
                state = (
                    getattr(status, "state", None) or status.get("state", None) if isinstance(status, dict) else None
                )

                if state in ("succeeded", "finished", "completed"):
                    result = await batch_client.get_result(job_id)  # type: ignore[attr-defined]
                    # Expect result.data similar to response.data where each item has .embedding
                    data_items = (
                        getattr(result, "data", None) or result.get("data", None) if isinstance(result, dict) else None
                    )
                    if not data_items:
                        raise RuntimeError("Batch job completed but no data returned")
                    embeddings = []
                    for item in data_items:
                        emb = (
                            getattr(item, "embedding", None) or item.get("embedding", None)
                            if isinstance(item, dict)
                            else None
                        )
                        if emb is None:
                            raise RuntimeError("Batch result item missing embedding")
                        embeddings.append(np.array(emb, dtype=np.float64))
                    return embeddings

                if state in ("failed", "error"):
                    raise RuntimeError(f"Batch job {job_id} failed (state={state})")

                elapsed = (datetime.now() - start).total_seconds()
                if elapsed > timeout_seconds:
                    raise TimeoutError(f"Batch job {job_id} did not complete within timeout ({timeout_seconds}s)")

                await _asyncio.sleep(poll_interval)
                # increase interval up to a cap
                poll_interval = min(poll_interval * 1.5, 30.0)

        except Exception as e:
            logger.warning(f"Batch API embedding path failed: {e}; falling back to per-call embeddings")
            return await self._embed_token_lists(token_lists)

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
                # If a RateLimiter was provided to the embedder, acquire tokens for the whole batch
                # before issuing the batch request. This integrates with external rate-limiting
                # infrastructure when desired.
                if getattr(self, "use_rate_limiting", False) and getattr(self, "rate_limiter", None):
                    try:
                        estimated_tokens_batch = sum(len(token_list) for token_list in batch)
                        # The RateLimiter API is async; call acquire and ignore any errors (best-effort)
                        await self.rate_limiter.acquire(tokens=estimated_tokens_batch)  # type: ignore[attr-defined]
                    except Exception:
                        # If the external limiter fails, continue without raising to avoid blocking embedding
                        logger.debug("Batch embedder: rate_limiter.acquire failed or not available; continuing")

                # Use batch API path if requested, otherwise fall back to token-list embedding.
                # The batch API is optional and may be provided via `batch_api_client`.
                if getattr(self, "use_batch_api", False) and getattr(self, "batch_api_client", None):
                    try:
                        batch_embeddings = await self._embed_via_batch_api(batch)
                    except Exception as e:
                        logger.warning(f"Batch API path failed ({e}); falling back to per-call embedding")
                        batch_embeddings = await self._embed_token_lists(batch)
                else:
                    batch_embeddings = await self._embed_token_lists(batch)
                all_embeddings.extend(batch_embeddings)

                # Sleep between batches to respect local sleep/backoff settings (except last batch)
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
        tokens_col = f"{text_column}.tokens"

        # Record original presence of columns so we can preserve external schema
        had_embedding_col_original = embedding_column in df.columns
        had_tokens_col_original = tokens_col in df.columns
        had_token_count_col_original = token_count_column in df.columns

        # Initialize embedding column if it doesn't exist
        if not had_embedding_col_original:
            df = df.with_columns(pl.lit(None).alias(embedding_column))

        # Filter out rows to embed
        df_to_embed = df.filter(pl.col(text_column).is_not_null())

        if not re_embed:
            df_to_embed = df_to_embed.filter(pl.col(embedding_column).is_null())

        if len(df_to_embed) == 0:
            logger.info("No texts to embed (all already embedded)")
            return df

        # Ensure tokenization columns exist (backward-compatible)
        df_to_embed = self._tokenize_dataframe(df_to_embed, text_column=text_column, verbose=verbose)

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

        # Add embeddings to DataFrame
        embedding_length = len(embeddings[0])

        df_to_embed = df_to_embed.with_columns(
            [
                pl.Series(embedding_column, embeddings, dtype=pl.Array(pl.Float64, embedding_length)),
            ]
        )

        # Ensure token_count column exists; if not, compute conservative counts
        if token_count_column not in df_to_embed.columns:
            token_counts = [self.config.count_tokens(text) for text in texts_to_embed]
            df_to_embed = df_to_embed.with_columns([pl.Series(token_count_column, token_counts)])

        # Merge back with original DataFrame (for rows that weren't embedded)
        # Preserve the original schema when appropriate (e.g. when input already had an embedding column).
        tokens_col = f"{text_column}.tokens"
        token_count_col = f"{text_column}.token_count"
        # Use the original presence flags recorded earlier (before we possibly added default columns)
        had_embedding_col = had_embedding_col_original
        had_tokens_col = had_tokens_col_original
        had_token_count_col = had_token_count_col_original

        if not re_embed and len(df) > len(df_to_embed):
            # Get rows that weren't embedded (either had embeddings already or null texts)
            df_not_embedded = df.filter(pl.col(text_column).is_null() | pl.col(embedding_column).is_not_null())

            # Add missing columns to df_not_embedded so concat doesn't fail
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

        # If the incoming DataFrame already had an embedding column (common in incremental flows),
        # avoid introducing new token columns that would change the external schema. Only keep
        # token-related columns if they were present in the original input.
        if had_embedding_col:
            # Remove tokens column if it didn't exist originally
            if not had_tokens_col and tokens_col in result_df.columns:
                result_df = result_df.drop(tokens_col)
            # Remove token_count column if it didn't exist originally
            if not had_token_count_col and token_count_col in result_df.columns:
                result_df = result_df.drop(token_count_col)

        return result_df


__all__ = [
    "PolarsBatchEmbedder",
]
