"""
Multi-turn conversation manager for Azure LLM Toolkit.

This module provides conversation state management for building chat applications,
including history tracking, automatic summarization, and context window management.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Awaitable

from .client import AzureLLMClient
from .types import ChatCompletionResult

logger = logging.getLogger(__name__)


@dataclass
class ConversationMessage:
    """Represents a single message in a conversation."""

    role: str
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)
    tokens: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to OpenAI message format."""
        return {
            "role": self.role,
            "content": self.content,
        }

    def to_full_dict(self) -> dict[str, Any]:
        """Convert to full dictionary with metadata."""
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "tokens": self.tokens,
        }


@dataclass
class ConversationConfig:
    """Configuration for conversation manager."""

    max_history_messages: int = 20
    max_history_tokens: int = 4000
    auto_summarize: bool = True
    summarize_threshold_messages: int = 10
    summarize_threshold_tokens: int = 3000
    system_prompt: str | None = None
    model: str | None = None
    temperature: float = 0.7
    max_tokens: int | None = None


class ConversationManager:
    """
    Manages multi-turn conversations with automatic history management.

    Features:
    - Message history tracking
    - Automatic token counting
    - Context window management
    - Automatic summarization of old messages
    - Message metadata support
    - Conversation export/import

    Example:
        >>> client = AzureLLMClient(config=config)
        >>> manager = ConversationManager(client)
        >>>
        >>> # Send messages
        >>> response = await manager.send_message("Hello!")
        >>> print(response.content)
        >>>
        >>> # Get history
        >>> history = manager.get_history()
        >>> print(len(history))
    """

    def __init__(
        self,
        client: AzureLLMClient,
        config: ConversationConfig | None = None,
        conversation_id: str | None = None,
    ):
        """
        Initialize conversation manager.

        Args:
            client: AzureLLMClient instance
            config: Optional conversation configuration
            conversation_id: Optional conversation ID for tracking
        """
        self.client = client
        self.config = config or ConversationConfig()
        self.conversation_id = conversation_id or self._generate_conversation_id()

        self.messages: list[ConversationMessage] = []
        self.summary: str | None = None
        self.total_tokens: int = 0
        self.created_at = datetime.now()
        self.updated_at = datetime.now()

    @staticmethod
    def _generate_conversation_id() -> str:
        """Generate a unique conversation ID."""
        import uuid

        return str(uuid.uuid4())

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return self.client.config.count_tokens(text)

    def _should_summarize(self) -> bool:
        """Check if conversation should be summarized."""
        if not self.config.auto_summarize:
            return False

        message_count = len(self.messages)
        token_count = sum(m.tokens or 0 for m in self.messages)

        return (
            message_count >= self.config.summarize_threshold_messages
            or token_count >= self.config.summarize_threshold_tokens
        )

    async def _summarize_history(self) -> str:
        """Summarize conversation history."""
        if not self.messages:
            return ""

        # Create summary prompt where recent turns are more important and the summary is optimized
        # to help continue the conversation naturally.
        history_text = "\n\n".join([f"{m.role}: {m.content}" for m in self.messages])

        summary_prompt = (
            "You are maintaining a running summary of an ongoing conversation.\n\n"
            "Task:\n"
            "- Summarize the conversation so far.\n"
            "- Preserve all key facts, decisions, and user preferences.\n"
            "- Give extra weight to the most recent turns so that the summary is\n"
            "  especially useful for continuing the conversation from where it left off.\n"
            "- Write the summary in a way that the assistant can use it as context\n"
            "  to seamlessly answer the next user message.\n\n"
            "Conversation:\n"
            f"{history_text}\n\n"
            "Provide ONLY the summary, no extra commentary."
        )

        try:
            response = await self.client.chat_completion(
                messages=[{"role": "user", "content": summary_prompt}],
                system_prompt="You maintain summaries that help continue the conversation smoothly.",
                model=self.config.model,
                track_cost=False,
            )

            summary = response.content
            logger.info(f"Summarized {len(self.messages)} messages into summary")
            return summary

        except Exception as e:
            logger.warning(f"Failed to summarize conversation: {e}")
            return ""

    async def _apply_summarization(self) -> None:
        """Apply summarization and trim history."""
        if not self._should_summarize():
            return

        # Summarize current history
        summary = await self._summarize_history()

        if summary:
            # Keep only recent messages
            keep_count = self.config.max_history_messages // 2
            recent_messages = self.messages[-keep_count:] if keep_count > 0 else []

            # Update summary
            if self.summary:
                self.summary = f"{self.summary}\n\nContinued: {summary}"
            else:
                self.summary = summary

            # Replace history with recent messages
            self.messages = recent_messages

            logger.info(f"Applied summarization, kept {len(recent_messages)} recent messages")

    def _trim_history(self) -> None:
        """Trim history to stay within limits."""
        # Trim by message count
        if len(self.messages) > self.config.max_history_messages:
            excess = len(self.messages) - self.config.max_history_messages
            self.messages = self.messages[excess:]
            logger.debug(f"Trimmed {excess} messages from history")

        # Trim by token count
        total_tokens = sum(m.tokens or 0 for m in self.messages)
        while total_tokens > self.config.max_history_tokens and len(self.messages) > 1:
            removed = self.messages.pop(0)
            total_tokens -= removed.tokens or 0
            logger.debug(f"Trimmed message to stay within token limit")

    def _prepare_messages_for_api(self) -> list[dict[str, str]]:
        """Prepare messages for API call."""
        messages = []

        # Add system prompt if configured
        if self.config.system_prompt:
            messages.append({"role": "system", "content": self.config.system_prompt})

        # Add summary as system message if exists
        if self.summary:
            messages.append({"role": "system", "content": f"Previous conversation summary: {self.summary}"})

        # Add conversation history
        messages.extend([m.to_dict() for m in self.messages])

        return messages

    async def send_message(
        self,
        content: str,
        role: str = "user",
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> ChatCompletionResult:
        """
        Send a message and get response.

        Args:
            content: Message content
            role: Message role (default: "user")
            metadata: Optional metadata to attach to message
            **kwargs: Additional arguments passed to chat_completion

        Returns:
            ChatCompletionResult with assistant response
        """
        # Count tokens in user message
        tokens = self._count_tokens(content)

        # Add user message to history
        user_message = ConversationMessage(
            role=role,
            content=content,
            metadata=metadata or {},
            tokens=tokens,
        )
        self.messages.append(user_message)
        self.total_tokens += tokens

        # Check if summarization is needed
        if self.config.auto_summarize:
            await self._apply_summarization()

        # Trim history if needed
        self._trim_history()

        # Prepare messages for API
        api_messages = self._prepare_messages_for_api()

        # Get response from API
        try:
            response = await self.client.chat_completion(
                messages=api_messages,
                model=self.config.model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                **kwargs,
            )

            # Count tokens in assistant response
            assistant_tokens = (
                response.usage.completion_tokens if response.usage else self._count_tokens(response.content)
            )

            # Add assistant response to history
            assistant_message = ConversationMessage(
                role="assistant",
                content=response.content,
                tokens=assistant_tokens,
            )
            self.messages.append(assistant_message)
            self.total_tokens += assistant_tokens

            self.updated_at = datetime.now()

            return response

        except Exception as e:
            logger.error(f"Error sending message: {e}")
            raise

    async def send_message_stream(
        self,
        content: str,
        role: str = "user",
        callback: Callable[[str], None] | Callable[[str], Awaitable[None]] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> ChatCompletionResult:
        """
        Send a message with streaming response.

        Args:
            content: Message content
            role: Message role (default: "user")
            callback: Optional callback for streaming chunks
            metadata: Optional metadata
            **kwargs: Additional arguments

        Returns:
            ChatCompletionResult with complete response
        """
        # For now, use regular send_message
        # TODO: Implement streaming when chat_completion_stream is available
        return await self.send_message(content, role=role, metadata=metadata, **kwargs)

    def add_system_message(self, content: str) -> None:
        """
        Add a system message to the conversation.

        Args:
            content: System message content
        """
        tokens = self._count_tokens(content)
        message = ConversationMessage(
            role="system",
            content=content,
            tokens=tokens,
        )
        self.messages.append(message)
        self.total_tokens += tokens
        self.updated_at = datetime.now()

    def get_history(self, include_metadata: bool = False) -> list[dict[str, Any]]:
        """
        Get conversation history.

        Args:
            include_metadata: Whether to include full metadata

        Returns:
            List of message dictionaries
        """
        if include_metadata:
            return [m.to_full_dict() for m in self.messages]
        return [m.to_dict() for m in self.messages]

    def get_last_message(self) -> ConversationMessage | None:
        """Get the last message in the conversation."""
        return self.messages[-1] if self.messages else None

    def get_message_count(self) -> int:
        """Get the number of messages in the conversation."""
        return len(self.messages)

    def get_token_count(self) -> int:
        """Get total token count for current history."""
        return sum(m.tokens or 0 for m in self.messages)

    def clear_history(self) -> None:
        """Clear conversation history."""
        self.messages = []
        self.summary = None
        self.total_tokens = 0
        self.updated_at = datetime.now()
        logger.info(f"Cleared conversation history for {self.conversation_id}")

    async def regenerate_last_response(self, **kwargs: Any) -> ChatCompletionResult:
        """
        Regenerate the last assistant response.

        Args:
            **kwargs: Additional arguments for chat_completion

        Returns:
            ChatCompletionResult with new response
        """
        if not self.messages:
            raise ValueError("No messages in conversation")

        # Remove last assistant message if exists
        if self.messages[-1].role == "assistant":
            removed = self.messages.pop()
            self.total_tokens -= removed.tokens or 0

        # Get the last user message
        if not self.messages or self.messages[-1].role != "user":
            raise ValueError("Cannot regenerate: no user message to respond to")

        # Prepare messages for API
        api_messages = self._prepare_messages_for_api()

        # Get new response
        response = await self.client.chat_completion(
            messages=api_messages,
            model=self.config.model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            **kwargs,
        )

        # Add new assistant response
        assistant_tokens = response.usage.completion_tokens if response.usage else self._count_tokens(response.content)

        assistant_message = ConversationMessage(
            role="assistant",
            content=response.content,
            tokens=assistant_tokens,
        )
        self.messages.append(assistant_message)
        self.total_tokens += assistant_tokens

        self.updated_at = datetime.now()

        return response

    def export_conversation(self) -> dict[str, Any]:
        """
        Export conversation to dictionary.

        Returns:
            Dictionary containing conversation data
        """
        return {
            "conversation_id": self.conversation_id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "summary": self.summary,
            "total_tokens": self.total_tokens,
            "config": {
                "max_history_messages": self.config.max_history_messages,
                "max_history_tokens": self.config.max_history_tokens,
                "auto_summarize": self.config.auto_summarize,
                "system_prompt": self.config.system_prompt,
                "model": self.config.model,
            },
            "messages": [m.to_full_dict() for m in self.messages],
        }

    @classmethod
    def import_conversation(
        cls,
        client: AzureLLMClient,
        data: dict[str, Any],
    ) -> ConversationManager:
        """
        Import conversation from dictionary.

        Args:
            client: AzureLLMClient instance
            data: Dictionary containing conversation data

        Returns:
            ConversationManager instance
        """
        # Reconstruct config
        config_data = data.get("config", {})
        config = ConversationConfig(
            max_history_messages=config_data.get("max_history_messages", 20),
            max_history_tokens=config_data.get("max_history_tokens", 4000),
            auto_summarize=config_data.get("auto_summarize", True),
            system_prompt=config_data.get("system_prompt"),
            model=config_data.get("model"),
        )

        # Create manager
        manager = cls(
            client=client,
            config=config,
            conversation_id=data.get("conversation_id"),
        )

        # Restore state
        manager.summary = data.get("summary")
        manager.total_tokens = data.get("total_tokens", 0)

        if "created_at" in data:
            manager.created_at = datetime.fromisoformat(data["created_at"])
        if "updated_at" in data:
            manager.updated_at = datetime.fromisoformat(data["updated_at"])

        # Restore messages
        for msg_data in data.get("messages", []):
            message = ConversationMessage(
                role=msg_data["role"],
                content=msg_data["content"],
                timestamp=datetime.fromisoformat(msg_data.get("timestamp", datetime.now().isoformat())),
                metadata=msg_data.get("metadata", {}),
                tokens=msg_data.get("tokens"),
            )
            manager.messages.append(message)

        return manager

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"ConversationManager(id={self.conversation_id}, messages={len(self.messages)}, tokens={self.total_tokens})"
        )


__all__ = [
    "ConversationMessage",
    "ConversationConfig",
    "ConversationManager",
]
