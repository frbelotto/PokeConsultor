from typing import Any

from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langgraph.graph.state import CompiledStateGraph

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from pokeconsultor.llm.base import LLMProfile

from pokeconsultor.services.memory import ConversationMemory

from langchain.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.messages import BaseMessage


class AIAgent(BaseModel):
    """Thin wrapper that instantiates a LangChain chat model with memory."""
    
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")
    
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

        self._agent = create_agent(llm_instance)

    @property
    def agent(self) -> CompiledStateGraph:
        """Get the instantiated chat model."""
        return self._agent

    def respond(self, prompt: HumanMessage, ragcontext:HumanMessage) -> AIMessage:
        
        
        try:
            history_msgs = self.memory.get_messages()
        except Exception:
            history_msgs = []

        # # Append the current user prompt
        # humanmessages = [m for m in prompt if isinstance(m, HumanMessage)]

        # Persist user message
        try:
            self.memory.add_user_message(prompt)
        except Exception:
            pass

        # Invoke the chat model
        response = self._agent.invoke(
            {'messages' : {
                'input': prompt,
                'context': ragcontext,  
            }
            })
            

        # Persist assistant reply
        try:
            self.memory.add_assistant_message([response])
        except Exception:
            pass

        return response
