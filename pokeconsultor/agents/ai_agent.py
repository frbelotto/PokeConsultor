import threading
from typing import Any, cast

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from pokeconsultor.config import settings
from pokeconsultor.llm.base import LLMProfile
from pokeconsultor.services.logger import logger
from pokeconsultor.services.memory import checkpointer, middleware


_BASE_RECURSION_LIMIT = 20
_PER_MIDDLEWARE_RECURSION_COST = 4


class AIAgent(BaseModel):
    """Thin wrapper that instantiates a LangChain chat model with memory."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")
    systemprompt: SystemMessage
    llm: LLMProfile
    tools: list[Any] = Field(default_factory=list)
    _threadid: int = PrivateAttr(default_factory=threading.get_ident)
    _memory: InMemorySaver = checkpointer
    _agent: CompiledStateGraph = PrivateAttr()

    def model_post_init(self, _context: Any) -> None:
        """Instantiate the chat model passing the provider API key when available."""

        llm_instance = init_chat_model(
            model=self.llm.model,
            model_provider=self.llm.provider,
            temperature=self.llm.temperature,
            max_tokens=self.llm.max_tokens,
            api_key=self.llm.api_key.get_secret_value(),
        )

        self._agent = create_agent(
            llm_instance,
            system_prompt=self.systemprompt,
            checkpointer=self._memory,
            middleware=middleware,
            tools=self.tools,
        )

    @property
    def agent(self) -> CompiledStateGraph:
        """Get the instantiated chat model."""
        return self._agent

    def respond(self, prompt: HumanMessage | str) -> str:
        """Send a user prompt to the agent and return plain text response."""

        user_text = prompt.content if isinstance(prompt, HumanMessage) else str(prompt)
        messages = [{"role": "user", "content": user_text}]
        logger.debug(f"Agent messages: {messages}")

        # Each middleware adds graph steps around the model/tool execution cycle.
        # Keep a minimum safe recursion budget to avoid premature GraphRecursionError
        # when multiple middlewares are enabled simultaneously.
        effective_recursion_limit = _BASE_RECURSION_LIMIT + (
            len(middleware) * _PER_MIDDLEWARE_RECURSION_COST
        )

        logger.debug(
            "Agent invocation limits | recursion_limit=%s",
            effective_recursion_limit,
        )

        invoke_config: RunnableConfig = cast(
            RunnableConfig,
            {
                "recursion_limit": effective_recursion_limit,
                "configurable": {
                    "thread_id": str(self._threadid),
                },
            },
        )

        raw = self._agent.invoke(
            {"messages": messages},
            invoke_config,
        )

        return self._normalize_output(raw)

    def get_state_history(self) -> list[Any]:
        """Return persisted graph state history for current thread."""
        history = self._agent.get_state_history(
            {"configurable": {"thread_id": str(self._threadid)}}
        )
        return list(history)

    def clear_thread_memory(self) -> None:
        """Clear persisted graph state for the current conversation thread."""
        self._memory.delete_thread(str(self._threadid))

    def get_latest_interaction_tool_usage(self) -> dict[str, Any]:
        """Return tool usage summary for the latest user interaction."""
        history = self.get_state_history()
        if not history:
            return {"used": False, "tool_names": [], "tool_calls": 0}

        latest_state = history[0]
        values = getattr(latest_state, "values", {})
        messages = values.get("messages", []) if isinstance(values, dict) else []
        if not isinstance(messages, list) or not messages:
            return {"used": False, "tool_names": [], "tool_calls": 0}

        tool_names: list[str] = []
        tool_calls = 0

        for message in reversed(messages):
            if isinstance(message, HumanMessage):
                break

            message_tool_name = getattr(message, "name", None)
            if message_tool_name and message_tool_name not in tool_names:
                tool_names.append(message_tool_name)
                tool_calls += 1

            llm_tool_calls = getattr(message, "tool_calls", None)
            if isinstance(llm_tool_calls, list) and llm_tool_calls:
                for call in llm_tool_calls:
                    if not isinstance(call, dict):
                        continue
                    name = call.get("name")
                    if name and name not in tool_names:
                        tool_names.append(name)
                    tool_calls += 1

        return {
            "used": bool(tool_names),
            "tool_names": tool_names,
            "tool_calls": tool_calls,
        }

    @staticmethod
    def _normalize_output(raw: Any) -> str:
        """Normalize graph output structures into final assistant text."""
        if isinstance(raw, str):
            return raw

        if isinstance(raw, dict):
            messages = raw.get("messages")
            if isinstance(messages, list) and messages:
                last = messages[-1]
                if isinstance(last, dict):
                    return str(last.get("content", "")).strip()
                content = getattr(last, "content", None)
                if content is not None:
                    return str(content).strip()

            for key in ("output", "content", "text"):
                value = raw.get(key)
                if value:
                    return str(value).strip()

        return str(raw).strip()
