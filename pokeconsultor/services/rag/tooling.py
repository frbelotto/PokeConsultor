"""Tool adapters for integrating RAGService with LangGraph/LangChain agents."""

from __future__ import annotations

import json
from typing import Any

from langchain.tools import tool
from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from pokeconsultor.services.logger import logger
from pokeconsultor.services.rag.service import RAGService


def _json_payload(**payload: Any) -> str:
    """Serialize tool payload preserving UTF-8 characters."""
    return json.dumps(payload, ensure_ascii=False)


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
