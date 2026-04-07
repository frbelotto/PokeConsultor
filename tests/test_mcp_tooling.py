"""Unit tests for MCP tooling behavior and graceful fallback."""

from __future__ import annotations

from typing import Any, cast

from langchain_core.documents import Document

from pokeconsultor.services.rag import tooling


class FakeRAGService:
    """Minimal fake RAG service used to build tools in tests."""

    def retrieve(self, query: str) -> list[Document]:
        return [Document(page_content=f"Local result for: {query}")]

    def format_results(self, results: list[Document]) -> str:
        return "\n".join(doc.page_content for doc in results)


def test_build_agent_tools_keeps_rag_when_mcp_disabled() -> None:
    """RAG tool must always be present when MCP is disabled."""
    tools = tooling.build_agent_tools(
        cast(Any, FakeRAGService()),
        mcp_enabled=False,
        mcp_server_url="http://localhost:9999",
    )

    assert len(tools) == 1
    assert tools[0].name == "retrieve_context"


def test_build_agent_tools_adds_mcp_when_enabled() -> None:
    """MCP tool should be appended as complementary tool when enabled."""
    fake_mcp_tool = cast(Any, type("FakeMCPTool", (), {"name": "pokeapi_search"})())
    original_loader = tooling._load_mcp_tools
    tooling._load_mcp_tools = lambda server_url: [fake_mcp_tool]

    try:
        tools = tooling.build_agent_tools(
            cast(Any, FakeRAGService()),
            mcp_enabled=True,
            mcp_server_url="http://localhost:9999",
        )
    finally:
        tooling._load_mcp_tools = original_loader

    names = [tool.name for tool in tools]
    assert names == ["retrieve_context", "pokeapi_search"]


def test_load_mcp_tools_returns_empty_list_on_runtime_error(monkeypatch) -> None:
    """MCP loader must fail gracefully and keep startup resilient."""

    async def fake_async_loader(server_url: str) -> list[Any]:
        del server_url
        raise RuntimeError("offline")

    monkeypatch.setattr(tooling, "_load_mcp_tools_async", fake_async_loader)

    tools = tooling._load_mcp_tools("http://localhost:9999")
    assert tools == []


def test_load_mcp_tools_returns_tools_on_success(monkeypatch) -> None:
    """MCP loader should return adapter tools when server is reachable."""
    fake_tool = cast(Any, type("FakeMCPTool", (), {"name": "pokeapi_lookup"})())

    async def fake_async_loader(server_url: str) -> list[Any]:
        del server_url
        return [fake_tool]

    monkeypatch.setattr(tooling, "_load_mcp_tools_async", fake_async_loader)

    tools = tooling._load_mcp_tools("http://localhost:9999")
    assert [tool.name for tool in tools] == ["pokeapi_lookup"]


def test_load_mcp_tools_returns_empty_list_on_import_error(monkeypatch) -> None:
    """Missing adapter dependency must not break agent startup."""

    async def fake_async_loader(server_url: str) -> list[Any]:
        del server_url
        raise ImportError("langchain_mcp_adapters not installed")

    monkeypatch.setattr(tooling, "_load_mcp_tools_async", fake_async_loader)

    tools = tooling._load_mcp_tools("http://localhost:9999")
    assert tools == []
