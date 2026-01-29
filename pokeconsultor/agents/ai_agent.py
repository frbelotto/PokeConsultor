from typing import Any

from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langgraph.graph.state import CompiledStateGraph

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from pokeconsultor.llm.base import LLMProfile

from pokeconsultor.services.memory import ConversationMemory

from langchain.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain.messages import SystemMessage
from pokeconsultor.services.logger import logger


class AIAgent(BaseModel):
    """Thin wrapper that instantiates a LangChain chat model with memory."""
    
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")
    systemprompt: SystemMessage
    llm: LLMProfile
    memory: ConversationMemory = Field(default_factory=ConversationMemory)
    _agent: CompiledStateGraph = PrivateAttr()
    
    def model_post_init(self, _context: Any) -> None:
        """Instantiate the chat model passing the provider API key when available."""
        
        api_key = self.llm.api_key.get_secret_value()
        
        llm_instance = init_chat_model(
            model=self.llm.model,
            model_provider=self.llm.provider,
            temperature=self.llm.temperature,
            max_tokens=self.llm.max_tokens,
            api_key=api_key,
        )

        self._agent = create_agent(
            llm_instance, system_prompt=self.systemprompt)

    @property
    def agent(self) -> CompiledStateGraph:
        """Get the instantiated chat model."""
        return self._agent

    def respond(self, prompt: HumanMessage, ragcontext: HumanMessage) -> str:
        """Send prompt and optional RAG context to the agent and return the parsed response as plain text.

        The agent expects a list of message dicts with 'role' and 'content'. We build that
        list from conversation history, optional RAG context, and the current user prompt.
        """

        # Retrieve history as list of {'role','content'} dicts
        try:
            history = self.memory.get_history()
        except Exception:
            history = []

        # Persist the current user message into memory (expects a list of HumanMessage)
        try:
            self.memory.add_user_message([prompt])
        except Exception:
            pass

        # Build messages for the agent invoke: history followed by optional rag context and user prompt
        messages = []
        messages.extend(history)

        if ragcontext and getattr(ragcontext, "content", ""):
            # Include RAG context as a system message so the model sees it as grounding info
            messages.append({"role": "system", "content": ragcontext.content})

        # Current user prompt
        messages.append({"role": "user", "content": prompt.content})
        logger.debug(f"Agent messages: {messages}")

        parser = StrOutputParser()


        raw = self._agent.invoke({"messages": messages})
        parsed = parser.parse(raw)

        def _normalize_parsed(obj: Any) -> str:
            """Normalize parser/agent outputs to plain string."""
            if isinstance(obj, str):
                return obj
            if isinstance(obj, dict):
                for k in ("text", "output", "content"):
                    v = obj.get(k)
                    if v:
                        return str(v)
                msgs = obj.get("messages")
                if isinstance(msgs, list) and msgs:
                    last = msgs[-1]
                    if isinstance(last, dict):
                        return str(last.get("content") or last.get("text") or last)
                    return str(getattr(last, "content", last))
                return str(obj)
            return str(obj)

        final_text = _normalize_parsed(parsed)


        ai_msg = AIMessage(content=final_text)
        try:
            self.memory.add_assistant_message([ai_msg])
        except Exception:
            pass

        return final_text
