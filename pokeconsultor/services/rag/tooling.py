"""Tool adapters for integrating RAGService with LangGraph/LangChain agents."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from langchain.tools import tool
from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from pokeconsultor.config import settings
from pokeconsultor.services.logger import logger
from pokeconsultor.services.rag.service import RAGService


def _json_payload(**payload: Any) -> str:
    """Serialize tool payload preserving UTF-8 characters."""
    return json.dumps(payload, ensure_ascii=False)


class RetrieveContextInput(BaseModel):
    """Input schema for retrieve_context tool."""

    query: str = Field(description="User query to retrieve relevant local context")


def _build_mcp_connections(server_url: str) -> dict[str, Any]:
    """Build MCP connection mapping for MultiServerMCPClient."""
    token = settings.POKEAPI_MCP_AUTH_TOKEN
    token_value = token.get_secret_value() if token is not None else ""

    connection: dict[str, Any] = {
        "transport": "http",
        "url": server_url,
    }

    if token_value:
        connection["headers"] = {
            "Authorization": f"Bearer {token_value}",
            "MCP-Proxy-Auth-Token": token_value,
        }

    return {"pokeapi": connection}


async def _load_mcp_tools_async(server_url: str) -> list[Any]:
    """Load MCP tools asynchronously using LangChain MCP adapters."""
    from langchain_mcp_adapters.client import MultiServerMCPClient

    client = MultiServerMCPClient(
        _build_mcp_connections(server_url),
        tool_name_prefix=True,
    )
    return await client.get_tools()


def _load_mcp_tools(server_url: str) -> list[Any]:
    """Load MCP tools with graceful fallback when MCP is unavailable."""
    try:
        tools = asyncio.run(_load_mcp_tools_async(server_url))
        logger.info("[MCP TOOLING] loaded_mcp_tools=%d", len(tools))
        return tools
    except ImportError as exc:
        logger.warning(
            "[MCP TOOLING] langchain-mcp-adapters not available; using RAG only | error=%s",
            exc,
        )
        return []
    except (RuntimeError, ConnectionError, TimeoutError) as exc:
        logger.warning(
            "[MCP TOOLING] MCP unavailable; using RAG only | error=%s",
            exc,
        )
        return []


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

        logger.info("[RAG TOOL] retrieve_context called | query=%s", query)

        results = rag_service.retrieve(query)
        if not results:
            logger.info("[RAG TOOL] no context found")
            return _json_payload(
                query=query,
                retrieved_docs=0,
                sources=[],
                context="",
                message="No relevant local context was found for this query.",
            )

        logger.info("[RAG TOOL] retrieved_docs=%d", len(results))
        return _json_payload(
            query=query,
            retrieved_docs=len(results),
            sources=[_source_metadata(doc) for doc in results],
            context=rag_service.format_results(results),
            message="Relevant local context retrieved successfully.",
        )

    return retrieve_context


def build_agent_tools(
    rag_service: RAGService,
    *,
    mcp_enabled: bool,
    mcp_server_url: str,
) -> list[Any]:
    """Compose tools for the agent while keeping RAG always enabled."""
    rag_tool = build_rag_context_tool(rag_service)
    tools: list[Any] = [rag_tool]

    if not mcp_enabled:
        return tools

    mcp_tools = _load_mcp_tools(mcp_server_url)
    tools.extend(mcp_tools)
    return tools
