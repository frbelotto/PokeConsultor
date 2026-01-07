"""Conversation memory management service."""

from pokeconsultor.models.llm import ConversationMessage, MessageRole
from pokeconsultor.services.logger import logger


class ConversationMemory:
    """Manages conversation history for multi-turn interactions.

    Maintains a list of messages exchanged in a session, allowing the LLM
    to have context from previous exchanges within the same conversation.
    """

    def __init__(self, max_messages: int = 50) -> None:
        """Initialize conversation memory.

        Args:
            max_messages: Maximum number of messages to keep (oldest removed first).
                         Prevents unbounded memory growth.
        """
        self.messages: list[ConversationMessage] = []
        self.max_messages = max_messages
        logger.debug(
            "ConversationMemory initialized with max_messages=%d", max_messages
        )

    def add_user_message(self, content: str) -> None:
        """Add a user message to history.

        Args:
            content: The user's message content.
        """
        message = ConversationMessage(role=MessageRole.USER, content=content)
        self.messages.append(message)
        self._trim_history()
        logger.debug("User message added (total messages: %d)", len(self.messages))

    def add_assistant_message(self, content: str) -> None:
        """Add an assistant message to history.

        Args:
            content: The assistant's response content.
        """
        message = ConversationMessage(role=MessageRole.ASSISTANT, content=content)
        self.messages.append(message)
        self._trim_history()
        logger.debug("Assistant message added (total messages: %d)", len(self.messages))

    def add_system_message(self, content: str) -> None:
        """Add a system message to history.

        Args:
            content: The system message content.
        """
        message = ConversationMessage(role=MessageRole.SYSTEM, content=content)
        self.messages.append(message)
        logger.debug("System message added")

    def get_messages(self) -> list[ConversationMessage]:
        """Get all messages in conversation history.

        Returns:
            List of all conversation messages.
        """
        return self.messages.copy()

    def get_messages_for_llm(self) -> list[dict[str, str]]:
        """Get messages formatted for LLM API calls.

        Returns:
            List of dictionaries with 'role' and 'content' keys.
        """
        return [
            {"role": msg.role.value, "content": msg.content} for msg in self.messages
        ]

    def clear(self) -> None:
        """Clear all messages from conversation history."""
        message_count = len(self.messages)
        self.messages.clear()
        logger.info("Conversation history cleared (%d messages removed)", message_count)

    def get_summary(self) -> str:
        """Get a summary of conversation statistics.

        Returns:
            String with message count and breakdown by role.
        """
        user_count = sum(1 for msg in self.messages if msg.role == MessageRole.USER)
        assistant_count = sum(
            1 for msg in self.messages if msg.role == MessageRole.ASSISTANT
        )
        system_count = sum(1 for msg in self.messages if msg.role == MessageRole.SYSTEM)

        return (
            f"Conversation has {len(self.messages)} messages: "
            f"{user_count} user, {assistant_count} assistant, {system_count} system"
        )

    def _trim_history(self) -> None:
        """Remove oldest messages if max_messages is exceeded.

        Preserves conversation integrity by keeping most recent messages.
        """
        if len(self.messages) > self.max_messages:
            removed_count = len(self.messages) - self.max_messages
            self.messages = self.messages[-self.max_messages :]
            logger.debug(
                "Conversation history trimmed (removed %d oldest messages)",
                removed_count,
            )
