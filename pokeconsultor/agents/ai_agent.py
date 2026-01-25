from typing import Any

from langchain.chat_models import BaseChatModel, init_chat_model
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from pokeconsultor.llm.base import LLMProfile

from pokeconsultor.services.memory import ConversationMemory

from langchain.messages import HumanMessage, AIMessage
from langchain_core.messages import BaseMessage


class AIAgent(BaseModel):
    """Thin wrapper that instantiates a LangChain chat model with memory."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    llm: LLMProfile
    memory: ConversationMemory = Field(default_factory=ConversationMemory)
    _agent: BaseChatModel = PrivateAttr()

    def model_post_init(self, _context: Any) -> None:
        """Instantiate the chat model passing the provider API key when available."""
        api_key = self.llm.api_key.get_secret_value()

        init_kwargs: dict[str, Any] = {
            "temperature": self.llm.temperature,
            "max_tokens": self.llm.max_tokens,
            "api_key": api_key,
        }

        self._agent = init_chat_model(
            f"{self.llm.provider}:{self.llm.model}",
            **init_kwargs,
        )

    @property
    def agent(self) -> BaseChatModel:
        """Get the instantiated chat model."""
        return self._agent

    def respond(self, prompt: list[BaseMessage]) -> AIMessage:
        # memory stores BaseMessage instances (HumanMessage/AIMessage)
        try:
            history_msgs = self.memory.get_messages()
            prompt.extend(history_msgs)
        except Exception:
            # If memory fails, continue without it
            history_msgs = []

        # Append the current user prompt
        humanmessages = [m for m in prompt if isinstance(m, HumanMessage)]

        # Persist user message
        try:
            self.memory.add_user_message(humanmessages)
        except Exception:
            pass

        # Invoke the chat model
        response = self._agent.invoke(prompt)

        # Persist assistant reply
        try:
            self.memory.add_assistant_message([response])
        except Exception:
            pass

        return response
