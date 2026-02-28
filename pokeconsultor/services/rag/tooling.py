"""Tool adapters for integrating RAGService with LangGraph/LangChain agents."""

from __future__ import annotations

import json
from threading import Lock
from typing import Any

from langchain.tools import tool
from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from pokeconsultor.services.logger import logger
from pokeconsultor.services.rag.service import RAGService

_TOOL_CALL_COUNTS: dict[tuple[str, str, str], int] = {}
_TOOL_CALL_COUNTS_LOCK = Lock()
_MAX_TRACKED_TOOL_CALL_KEYS = 10_000


def _consume_tool_call_budget(
    *,
    tool_name: str,
    configurable: dict[str, Any],
) -> tuple[bool, int, int]:
    """Consume one tool-call slot for the current interaction.

    Returns:
        Tuple containing:
        - allowed: whether this call is within the configured limit
        - used_calls: calls already used in this turn after processing this request
        - max_calls: configured maximum calls for this turn
    """
    max_calls_raw = configurable.get("max_tool_calls")
    turn_id = configurable.get("turn_id")

    if max_calls_raw is None or turn_id is None:
        return True, 0, 0

    try:
        max_calls = int(max_calls_raw)
    except (TypeError, ValueError):
        return True, 0, 0

    if max_calls <= 0:
        return True, 0, max_calls

    thread_id = str(configurable.get("thread_id", "unknown"))
    turn_key = (thread_id, str(turn_id), tool_name)

    with _TOOL_CALL_COUNTS_LOCK:
        used_calls = _TOOL_CALL_COUNTS.get(turn_key, 0)
        if used_calls >= max_calls:
            return False, used_calls, max_calls

        updated_calls = used_calls + 1
        _TOOL_CALL_COUNTS[turn_key] = updated_calls

        if len(_TOOL_CALL_COUNTS) > _MAX_TRACKED_TOOL_CALL_KEYS:
            _TOOL_CALL_COUNTS.clear()

    return True, updated_calls, max_calls


class RetrieveContextInput(BaseModel):
    """Input schema for retrieve_context tool."""

    query: str = Field(description="User query to retrieve relevant local context")


def build_rag_context_tool(rag_service: RAGService) -> Any:
    """Build a LangChain tool that returns formatted RAG context for a query."""

    def _source_metadata(doc: Document) -> dict[str, Any]:
        """Extract source metadata from a retrieved document."""
        metadata = doc.metadata or {}
        return {
            "file_path": metadata.get("file_path") or metadata.get("source"),
            "page_number": metadata.get("page_number"),
            "row_number": metadata.get("row_number"),
            "temp_id": metadata.get("_temp_id"),
        }

    @tool(
        "retrieve_context",
        args_schema=RetrieveContextInput,
        return_direct=False,
        description=(
            "Retrieve and return relevant local knowledge context for the user query. "
            "Always use this tool before answering knowledge questions."
        ),
    )
    def retrieve_context(query: str, config: RunnableConfig | None = None) -> str:
        """Retrieve context chunks and format them for LLM consumption."""

        configurable = {}
        if isinstance(config, dict):
            maybe_configurable = config.get("configurable", {})
            if isinstance(maybe_configurable, dict):
                configurable = maybe_configurable

        allowed, used_calls, max_calls = _consume_tool_call_budget(
            tool_name="retrieve_context",
            configurable=configurable,
        )

        if not allowed:
            logger.warning(
                "[RAG TOOL] retrieve_context blocked by max_tool_calls | used=%s | max=%s",
                used_calls,
                max_calls,
            )
            payload = {
                "query": query,
                "retrieved_docs": 0,
                "sources": [],
                "context": "",
                "message": (
                    "Tool call limit reached for this interaction. "
                    "Answer with the context already available."
                ),
            }
            return json.dumps(payload, ensure_ascii=False)

        logger.info("[RAG TOOL] retrieve_context called | query=%s", query)

        results = rag_service.retrieve(query)
        if not results:
            logger.info("[RAG TOOL] no context found")
            payload = {
                "query": query,
                "retrieved_docs": 0,
                "sources": [],
                "context": "",
                "message": "No relevant local context was found for this query.",
            }
            return json.dumps(payload, ensure_ascii=False)

        logger.info("[RAG TOOL] retrieved_docs=%d", len(results))
        payload = {
            "query": query,
            "retrieved_docs": len(results),
            "sources": [_source_metadata(doc) for doc in results],
            "context": rag_service.format_results(results),
            "message": "Relevant local context retrieved successfully.",
        }
        return json.dumps(payload, ensure_ascii=False)

    return retrieve_context
