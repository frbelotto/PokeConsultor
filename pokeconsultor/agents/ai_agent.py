from typing import Any

from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langgraph.graph.state import CompiledStateGraph

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from pokeconsultor.llm.base import LLMProfile
from pokeconsultor.services.rag.rag_tool import create_retrieve_context_tool


from langchain.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain.messages import SystemMessage
from pokeconsultor.services.logger import logger
from pokeconsultor.services.memory import checkpointer, middleware
from langgraph.checkpoint.memory import InMemorySaver
import threading


class AIAgent(BaseModel):
    """Thin wrapper that instantiates a LangChain chat model with memory and tools."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")
    systemprompt: SystemMessage
    llm: LLMProfile
    rag_service: Any = Field(
        description="RAGService instance for tool-calling integration"
    )
    _threadid: int = PrivateAttr(default_factory=threading.get_ident)
    _memory: InMemorySaver = checkpointer
    _agent: CompiledStateGraph = PrivateAttr()

    def model_post_init(self, _context: Any) -> None:
        """Instantiate the chat model with tools and memory."""

        api_key = self.llm.api_key.get_secret_value()

        llm_instance = init_chat_model(
            model=self.llm.model,
            model_provider=self.llm.provider,
            temperature=self.llm.temperature,
            max_tokens=self.llm.max_tokens,
            api_key=api_key,
        )

        # Create the retrieve_context tool with RAGService dependency
        retrieve_context_tool = create_retrieve_context_tool(self.rag_service)
        
        logger.info(f"🔧 AIAgent initialized with tools: {[retrieve_context_tool.name]}")

        # Create agent with tools
        # We pass the raw llm_instance as create_agent handles tool binding
        self._agent = create_agent(
            llm_instance,
            tools=[retrieve_context_tool],
            system_prompt=self.systemprompt,
            checkpointer=self._memory,
            middleware=middleware,
        )


    @property
    def agent(self) -> CompiledStateGraph:
        """Get the instantiated chat model."""
        return self._agent

    def respond(self, prompt: HumanMessage, ragcontext: HumanMessage) -> str:
        """Send prompt and optional RAG context to the agent and return the parsed response as plain text.

        The agent expects a list of message dicts with 'role' and 'content'. We build that
        list from conversation history, optional RAG context, and the current user prompt.
        """

        messages = []

        if ragcontext and getattr(ragcontext, "content", ""):
            messages.append({"role": "system", "content": ragcontext.content})

        messages.append({"role": "user", "content": prompt.content})
        logger.debug(f"Agent messages: {messages}")

        # Invoke the agent graph
        # The result 'raw' is the final state (dict) of the graph
        raw = self._agent.invoke(
            {"messages": messages}, {"configurable": {"thread_id": self._threadid}}
        )

        def _normalize_parsed(obj: Any) -> str:
            """Normalize parser/agent outputs to plain string."""
            if isinstance(obj, str):
                return obj
            if isinstance(obj, dict):
                # Check for direct output keys
                for k in ("text", "output", "content"):
                    v = obj.get(k)
                    if v:
                        return str(v)
                
                # Check for messages list in LangGraph state
                msgs = obj.get("messages")
                if isinstance(msgs, list) and msgs:
                    last = msgs[-1]
                    # If the last message is an AIMessage, extract its content
                    if hasattr(last, "content"):
                        return str(last.content)
                    if isinstance(last, dict):
                        return str(last.get("content") or last.get("text") or str(last))
                    return str(last)
                
                return str(obj)
            return str(obj)

        final_text = _normalize_parsed(raw)

        ai_msg = AIMessage(content=final_text)
        # try:
        #     self.memory.add_assistant_message([ai_msg])
        # except Exception:
        #     pass

        return final_text
