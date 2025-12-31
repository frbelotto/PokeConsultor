"""LLM request models."""

from __future__ import annotations

from pydantic import BaseModel, Field


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
