from __future__ import annotations

from typing import Any

from langchain.chat_models import BaseChatModel, init_chat_model
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field

from pokeconsultor.llm.base import LLMProfile
from pokeconsultor.models.llm import LLMRequest
from pokeconsultor.services.memory import ConversationMemory


class AIAgent(BaseModel):
    """Thin wrapper that instantiates a LangChain chat model with memory."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    llm: LLMProfile
    memory: ConversationMemory = Field(default_factory=ConversationMemory)
    agent: BaseChatModel | None = None

    def model_post_init(self, _context: Any) -> None:
        """Instantiate the chat model passing the provider API key when available."""
        api_key = self.llm.api_key.get_secret_value() if self.llm.api_key else ""

        init_kwargs: dict[str, Any] = {
            "temperature": self.llm.temperature,
            "max_tokens": self.llm.max_tokens,
        }

        if api_key:
            init_kwargs["api_key"] = api_key

        self.agent = init_chat_model(
            f"{self.llm.provider}:{self.llm.model}",
            **init_kwargs,
        )

    def respond(self, request: LLMRequest) -> str:
        """Generate a chat response using memory context and optional RAG.

        Args:
            request: LLMRequest containing prompt, system message, and optional context

        Returns:
            The assistant's response as a string.

        Includes previous conversation history from memory to provide context
        for multi-turn interactions.
        """

        assert self.agent is not None, "Chat model not initialized"

        messages: list[BaseMessage] = []

        if request.system_message:
            # Instructions FIRST to ensure LLM adheres to them,
            # then context so LLM has material to work with
            sys_content = request.system_message
            
            # If no context, explicitly state it to LLM as a safeguard
            if request.context:
                sys_content = f"{request.system_message}\n\n## CONTEXTO RELEVANTE\n{request.context}"
            else:
                # Enforce "no context = no guessing" by being explicit
                sys_content = f"{request.system_message}\n\n⚠️ AVISO: Você NÃO TEM CONTEXTO para esta pergunta. Você DEVE responder: 'Não tenho essa informação no contexto fornecido.'"
                
            messages.append(SystemMessage(content=sys_content))
        elif request.context:
            messages.append(SystemMessage(content=f"Context:\n{request.context}"))

        # Include conversation history from memory
        for msg in self.memory.get_messages():
            if msg.role.value == "user":
                messages.append(HumanMessage(content=msg.content))
            elif msg.role.value == "assistant":
                messages.append(AIMessage(content=msg.content))
            elif msg.role.value == "system":
                messages.append(SystemMessage(content=msg.content))

        # Add current user prompt
        messages.append(HumanMessage(content=request.prompt))

        response = self.agent.invoke(messages)

        # Persist conversation
        self.memory.add_user_message(request.prompt)

        # Ensure response.content is converted to string
        response_text = (
            str(response.content)
            if not isinstance(response.content, str)
            else response.content
        )
        self.memory.add_assistant_message(response_text)

        return response_text
