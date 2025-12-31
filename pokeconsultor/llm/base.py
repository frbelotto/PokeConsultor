"""Protocol definition for Language Models."""

from typing import Protocol

from pokeconsultor.models import LLMRequest


class LLM(Protocol):
    """Protocol for any Language Model implementation.

    Any class with a generate() method conforms to this protocol.
    No inheritance necessary - structural subtyping (duck typing).
    """

    def generate(self, request: LLMRequest) -> str:
        """Generate a response from the given request.

        Args:
            request: LLMRequest containing prompt, context, and system message.

        Returns:
            The generated response as a string.
        """
        ...
