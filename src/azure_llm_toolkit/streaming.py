"""Streaming responses to files, queues, and other sinks.

This module provides utilities for streaming LLM responses to various
destinations including files, queues, callbacks, and custom sinks.
"""

from __future__ import annotations

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncGenerator, Callable

logger = logging.getLogger(__name__)


@dataclass
class StreamChunk:
    """A chunk of streamed content."""

    content: str
    index: int
    finish_reason: str | None = None
    metadata: dict[str, Any] | None = None


class StreamSink(ABC):
    """
    Abstract base class for stream sinks.

    A sink receives streaming chunks and processes them.
    """

    @abstractmethod
    async def write(self, chunk: StreamChunk) -> None:
        """
        Write a chunk to the sink.

        Args:
            chunk: Stream chunk to write
        """
        pass

    async def close(self) -> None:
        """Close the sink and cleanup resources."""
        pass

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
        return False


class FileSink(StreamSink):
    """
    Stream sink that writes to a file.

    Example:
        >>> async with FileSink("output.txt") as sink:
        >>>     async for chunk in stream:
        >>>         await sink.write(chunk)
    """

    def __init__(self, path: Path | str, mode: str = "w", encoding: str = "utf-8"):
        """
        Initialize file sink.

        Args:
            path: File path to write to
            mode: File open mode ('w' for write, 'a' for append)
            encoding: File encoding
        """
        self.path = Path(path)
        self.mode = mode
        self.encoding = encoding
        self._file = None

    async def __aenter__(self):
        """Open file on context entry."""
        self._file = open(self.path, self.mode, encoding=self.encoding)
        return self

    async def write(self, chunk: StreamChunk) -> None:
        """Write chunk content to file."""
        if self._file is None:
            raise RuntimeError("File not opened. Use as context manager.")
        self._file.write(chunk.content)
        self._file.flush()

    async def close(self) -> None:
        """Close the file."""
        if self._file:
            self._file.close()
            self._file = None

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Close file on context exit."""
        await self.close()
        return False


class JSONLSink(StreamSink):
    """
    Stream sink that writes chunks as JSONL (JSON Lines).

    Each chunk is written as a JSON object on a new line.
    """

    def __init__(self, path: Path | str, encoding: str = "utf-8"):
        """
        Initialize JSONL sink.

        Args:
            path: File path to write to
            encoding: File encoding
        """
        self.path = Path(path)
        self.encoding = encoding
        self._file = None

    async def __aenter__(self):
        """Open file on context entry."""
        self._file = open(self.path, "w", encoding=self.encoding)
        return self

    async def write(self, chunk: StreamChunk) -> None:
        """Write chunk as JSON line."""
        if self._file is None:
            raise RuntimeError("File not opened. Use as context manager.")

        data = {
            "content": chunk.content,
            "index": chunk.index,
            "finish_reason": chunk.finish_reason,
            "metadata": chunk.metadata,
        }
        self._file.write(json.dumps(data) + "\n")
        self._file.flush()

    async def close(self) -> None:
        """Close the file."""
        if self._file:
            self._file.close()
            self._file = None

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Close file on context exit."""
        await self.close()
        return False


class QueueSink(StreamSink):
    """
    Stream sink that writes to an asyncio Queue.

    Example:
        >>> queue = asyncio.Queue()
        >>> async with QueueSink(queue) as sink:
        >>>     async for chunk in stream:
        >>>         await sink.write(chunk)
    """

    def __init__(self, queue: asyncio.Queue):
        """
        Initialize queue sink.

        Args:
            queue: Asyncio queue to write to
        """
        self.queue = queue

    async def write(self, chunk: StreamChunk) -> None:
        """Write chunk to queue."""
        await self.queue.put(chunk)

    async def close(self) -> None:
        """Signal end of stream by putting None."""
        await self.queue.put(None)


class CallbackSink(StreamSink):
    """
    Stream sink that calls a callback function for each chunk.

    Example:
        >>> async def on_chunk(chunk):
        >>>     print(chunk.content, end="", flush=True)
        >>>
        >>> async with CallbackSink(on_chunk) as sink:
        >>>     async for chunk in stream:
        >>>         await sink.write(chunk)
    """

    def __init__(self, callback: Callable[[StreamChunk], Any]):
        """
        Initialize callback sink.

        Args:
            callback: Async or sync function to call for each chunk
        """
        self.callback = callback

    async def write(self, chunk: StreamChunk) -> None:
        """Call callback with chunk."""
        result = self.callback(chunk)
        # Handle both sync and async callbacks
        if asyncio.iscoroutine(result):
            await result


class MultiSink(StreamSink):
    """
    Stream sink that writes to multiple sinks simultaneously.

    Example:
        >>> sinks = MultiSink([
        >>>     FileSink("output.txt"),
        >>>     QueueSink(my_queue),
        >>>     CallbackSink(print_chunk),
        >>> ])
        >>> async with sinks:
        >>>     async for chunk in stream:
        >>>         await sinks.write(chunk)
    """

    def __init__(self, sinks: list[StreamSink]):
        """
        Initialize multi sink.

        Args:
            sinks: List of sinks to write to
        """
        self.sinks = sinks

    async def write(self, chunk: StreamChunk) -> None:
        """Write chunk to all sinks."""
        await asyncio.gather(*[sink.write(chunk) for sink in self.sinks])

    async def close(self) -> None:
        """Close all sinks."""
        await asyncio.gather(*[sink.close() for sink in self.sinks])


class BufferedSink(StreamSink):
    """
    Stream sink that buffers chunks before writing.

    Useful for reducing I/O operations or batching network requests.

    Example:
        >>> sink = BufferedSink(FileSink("output.txt"), buffer_size=10)
        >>> async with sink:
        >>>     async for chunk in stream:
        >>>         await sink.write(chunk)
    """

    def __init__(self, inner_sink: StreamSink, buffer_size: int = 10):
        """
        Initialize buffered sink.

        Args:
            inner_sink: The sink to buffer writes to
            buffer_size: Number of chunks to buffer before flushing
        """
        self.inner_sink = inner_sink
        self.buffer_size = buffer_size
        self._buffer: list[StreamChunk] = []

    async def write(self, chunk: StreamChunk) -> None:
        """Buffer chunk and flush if buffer is full."""
        self._buffer.append(chunk)
        if len(self._buffer) >= self.buffer_size:
            await self._flush()

    async def _flush(self) -> None:
        """Flush buffered chunks to inner sink."""
        if not self._buffer:
            return
        for chunk in self._buffer:
            await self.inner_sink.write(chunk)
        self._buffer.clear()

    async def close(self) -> None:
        """Flush remaining buffer and close inner sink."""
        await self._flush()
        await self.inner_sink.close()


class StreamProcessor:
    """
    Process streaming responses and write to sinks.

    Example:
        >>> processor = StreamProcessor()
        >>> sink = FileSink("output.txt")
        >>>
        >>> async with sink:
        >>>     async for chunk in client.chat_completion_stream(...):
        >>>         processed = await processor.process(chunk, index)
        >>>         await sink.write(processed)
    """

    def __init__(self):
        """Initialize stream processor."""
        self._index = 0
        self._accumulated = ""

    async def process(
        self,
        content: str,
        finish_reason: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> StreamChunk:
        """
        Process a content chunk.

        Args:
            content: Content chunk
            finish_reason: Optional finish reason
            metadata: Optional metadata

        Returns:
            StreamChunk with processed content
        """
        chunk = StreamChunk(
            content=content,
            index=self._index,
            finish_reason=finish_reason,
            metadata=metadata,
        )
        self._index += 1
        self._accumulated += content
        return chunk

    def get_accumulated(self) -> str:
        """Get all accumulated content."""
        return self._accumulated

    def reset(self) -> None:
        """Reset processor state."""
        self._index = 0
        self._accumulated = ""


async def stream_to_file(
    stream: AsyncGenerator[str, None],
    path: Path | str,
    append: bool = False,
) -> str:
    """
    Stream content to a file and return accumulated content.

    Args:
        stream: Async generator of content chunks
        path: File path to write to
        append: Whether to append to file (default: overwrite)

    Returns:
        Complete accumulated content

    Example:
        >>> content = await stream_to_file(
        >>>     client.chat_completion_stream(...),
        >>>     "output.txt"
        >>> )
    """
    mode = "a" if append else "w"
    processor = StreamProcessor()

    async with FileSink(path, mode=mode) as sink:
        async for content_chunk in stream:
            chunk = await processor.process(content_chunk)
            await sink.write(chunk)

    return processor.get_accumulated()


async def stream_to_queue(
    stream: AsyncGenerator[str, None],
    queue: asyncio.Queue,
) -> str:
    """
    Stream content to a queue and return accumulated content.

    Args:
        stream: Async generator of content chunks
        queue: Asyncio queue to write to

    Returns:
        Complete accumulated content

    Example:
        >>> queue = asyncio.Queue()
        >>> task = asyncio.create_task(stream_to_queue(stream, queue))
        >>> while True:
        >>>     chunk = await queue.get()
        >>>     if chunk is None:
        >>>         break
        >>>     print(chunk.content)
        >>> content = await task
    """
    processor = StreamProcessor()

    async with QueueSink(queue) as sink:
        async for content_chunk in stream:
            chunk = await processor.process(content_chunk)
            await sink.write(chunk)

    return processor.get_accumulated()


async def stream_with_callback(
    stream: AsyncGenerator[str, None],
    callback: Callable[[StreamChunk], Any],
) -> str:
    """
    Stream content with callback for each chunk.

    Args:
        stream: Async generator of content chunks
        callback: Function to call for each chunk

    Returns:
        Complete accumulated content

    Example:
        >>> def print_chunk(chunk):
        >>>     print(chunk.content, end="", flush=True)
        >>>
        >>> content = await stream_with_callback(stream, print_chunk)
    """
    processor = StreamProcessor()

    async with CallbackSink(callback) as sink:
        async for content_chunk in stream:
            chunk = await processor.process(content_chunk)
            await sink.write(chunk)

    return processor.get_accumulated()


__all__ = [
    "StreamSink",
    "StreamChunk",
    "FileSink",
    "JSONLSink",
    "QueueSink",
    "CallbackSink",
    "MultiSink",
    "BufferedSink",
    "StreamProcessor",
    "stream_to_file",
    "stream_to_queue",
    "stream_with_callback",
]
