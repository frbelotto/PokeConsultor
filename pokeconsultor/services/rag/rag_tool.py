"""LangChain tool for RAG retrieval.

Exposes the RAG system as a tool that the LLM can call to retrieve
relevant context from the knowledge base.
"""

from typing import TYPE_CHECKING
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from pokeconsultor.services.logger import logger

if TYPE_CHECKING:
    from pokeconsultor.services.rag.service import RAGService


class RetrieveContextInput(BaseModel):
    """Input schema for retrieve_context tool.
    
    Optimized for speed - encourages focused queries over multiple broad queries.
    """

    query: str = Field(
        description=(
            "Search query for hybrid retrieval (semantic + lexical). "
            "IMPORTANT: The search combines SEMANTIC (meaning-based) and LEXICAL (exact term matching). "
            "Use EXACT TERMS when searching for specific names, technical terms, or unique identifiers. "
            "Use DESCRIPTIVE CONCEPTS for broader topics. "
            "Examples: 'Harry Potter casamento' (exact name + concept), "
            "'Pikachu evolução pedra trovão' (name + exact terms), "
            "'personagem cicatriz relâmpago' (semantic description)"
        )
    )
    k: int = Field(
        default=3,
        ge=1,
        le=10,
        description=(
            "Number of documents to retrieve (default: 3 for speed). "
            "Use 3-5 for most queries. Only increase if you need more coverage."
        ),
    )


def create_retrieve_context_tool(rag_service: "RAGService"):
    """Factory function to create the retrieve_context tool with RAGService dependency.
    
    Args:
        rag_service: Initialized RAGService instance
        
    Returns:
        LangChain tool bound to the RAGService
    """

    @tool(args_schema=RetrieveContextInput)
    def retrieve_context(query: str, k: int = 3) -> str:
        """Retrieve relevant context from the knowledge base using HYBRID SEARCH.

        SEARCH STRATEGY - Combines TWO complementary approaches:
        1. SEMANTIC SEARCH (vector): Finds documents by MEANING and concepts
           → Good for: synonyms, paraphrases, conceptual matches
           → Example: "personagem cicatriz" finds "Harry Potter" content
        
        2. LEXICAL SEARCH (BM25): Finds documents by EXACT TERM matching
           → Good for: specific names, technical terms, unique identifiers
           → Example: "Pikachu" finds exact mentions of "Pikachu"
        
        3. FUSION & RERANKING: Merges both results using RRF + cross-encoder
        
        QUERY FORMULATION TIPS:
        - Use EXACT NAMES when searching for specific entities
        - Include DESCRIPTIVE TERMS to leverage semantic search
        - Combine both: "Harry Potter casamento" (name + concept)
        - Be specific and focused for best results
        
        PERFORMANCE: Prefer one well-crafted query over multiple broad queries.
        Only make multiple calls if you need fundamentally different information.

        Args:
            query: Search query (combines exact terms + concepts for hybrid search)
            k: Number of documents to retrieve (default: 3, max: 10)

        Returns:
            Formatted context string with retrieved documents and metadata
        """
        logger.info(
            f"🔧 Tool 'retrieve_context' called | query='{query}' | k={k}"
        )

        try:
            # Retrieve documents using hybrid search (no expansion)
            # We pass expand=False to disable automatic query expansion
            results = rag_service.retrieve(query, expand=False)

            if not results:
                logger.warning(f"No results found for query: {query}")
                return "Nenhum contexto relevante encontrado na base de conhecimento."

            # Limit to requested k
            results = results[:k]

            # Format results for LLM consumption
            formatted_context = rag_service.format_results(
                results, max_tokens=None, compact=True
            )

            # Log retrieval stats
            context_tokens = rag_service.count_tokens(formatted_context)
            logger.info(
                f"✅ Retrieved {len(results)} documents | ~{context_tokens} tokens"
            )

            return formatted_context

        except Exception as e:
            logger.exception(f"Error in retrieve_context tool: {e}")
            return f"Erro ao buscar contexto: {str(e)}"

    return retrieve_context
