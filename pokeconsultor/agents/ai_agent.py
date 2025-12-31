"""AI Agent for Pokemon consulting."""

from pokeconsultor.llm.base import LLM
from pokeconsultor.models import LLMRequest


class AIAgent:
    """Simple AI Agent with dependency injection for LLM models.

    The agent accepts any object that conforms to the LLM protocol,
    making it easy to swap different language models without inheritance.
    """

    def __init__(self, llm: LLM) -> None:
        """Initialize the AI Agent with an LLM instance.

        Args:
            llm: Any object that conforms to the LLM protocol (has generate method).
        """
        self.llm = llm

    def consult(
        self,
        prompt: str,
        context: str | None = None,
        system_message: str | None = None,
    ) -> str:
        """Process a user query and return a response from the LLM.

        Args:
            prompt: The user's question or request.
            context: Optional background information or context.
            system_message: Optional system instruction for the model.

        Returns:
            The response from the language model.
        """
        request = LLMRequest(
            prompt=prompt,
            context=context,
            system_message=system_message,
        )
        return self.llm.generate(request)
