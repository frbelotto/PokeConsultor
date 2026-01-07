"""LLM request models."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class MessageRole(str, Enum):
    """Enum for message roles in conversation history."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class ConversationMessage(BaseModel):
    """Model for a single message in conversation history.

    Represents a message exchanged between user and assistant,
    with role, content, and timestamp for tracking.
    """

    role: MessageRole = Field(description="Role of the message sender")
    content: str = Field(description="The message content")
    timestamp: datetime = Field(
        default_factory=datetime.now, description="When message was created"
    )

    class Config:
        """Pydantic config."""

        json_encoders = {datetime: lambda v: v.isoformat()}


class LLMRequest(BaseModel):
    """Request model for LLM calls.

    Encapsulates all parameters needed for language model invocation.
    """

    prompt: str = Field(description="The main prompt for the model")
    context: str | None = Field(
        default=None, description="Additional context or background information"
    )
    system_message: str | None = Field(
        default=None, description="System instruction to guide the model behavior"
    )
    messages: list[ConversationMessage] | None = Field(
        default=None, description="Conversation history for multi-turn conversations"
    )
