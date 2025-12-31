"""Groq LLM implementation."""

from langchain_groq import ChatGroq

from pokeconsultor.config import settings
from pokeconsultor.models import LLMRequest


class GroqLLM:
    """Language model implementation using Groq API."""

    def __init__(self) -> None:
        """Initialize Groq LLM with configuration from settings."""
        self.client = ChatGroq(
            model=settings.LLM_MODEL,
            temperature=settings.LLM_TEMPERATURE,
            max_tokens=settings.LLM_MAX_TOKENS,
            api_key=settings.LLM_API_KEY,
        )

    def generate(self, request: LLMRequest) -> str:
        """Generate a response from Groq LLM.

        Args:
            request: LLMRequest containing prompt, context, and system message.

        Returns:
            The generated response as a string.
        """
        # Build the full prompt with context if provided
        full_prompt = self._build_prompt(request)

        # Prepare messages for the API
        messages = []
        if request.system_message:
            messages.append({"role": "system", "content": request.system_message})

        messages.append({"role": "user", "content": full_prompt})

        response = self.client.invoke(messages)
        return response.content

    @staticmethod
    def _build_prompt(request: LLMRequest) -> str:
        """Build the complete prompt including context.

        Args:
            request: The LLM request with prompt and optional context.

        Returns:
            The formatted prompt string.
        """
        if request.context:
            return f"Context:\n{request.context}\n\nQuestion:\n{request.prompt}"
        return request.prompt
